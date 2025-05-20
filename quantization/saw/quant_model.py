import torch.nn as nn
from .quant_config import Quantize_config, QuantizeModel_config
from .quant_block import get_specials, BaseQuantBlock
from .quant_block import QuantBasicTransformerBlock, QuantResBlock
from .quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock
from .quant_layer import QuantLayer, StraightThrough, BasicQuantLayer, Basic2DquantLayer, get_layer_type
from .saw_layer import saw_Linear_QuantLayer, saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import MemoryEfficientCrossAttention
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm, trange
import torch
import torch.optim.lr_scheduler as lr_scheduler
from quantization.saw.saw_layer import get_loss_function
import copy

unet_first_layer_name = ["time_embed.0", "out.2"]
vae_first_layer_name = ["quant_conv", "post_quant_conv", "encoder.conv_out", "decoder.conv_out"]
first_layer_name = unet_first_layer_name + vae_first_layer_name

class QuantWrapper(nn.Module):
    def __init__(self, model: nn.Module, quant_config, device="cuda", exclude=None):
        nn.Module.__init__(self)
        self.model = QuantModel(model, quant_config, device, exclude)

    def forward(self, x, timesteps=None, encoder_hidden_states=None):
        return self.model(x, timesteps, encoder_hidden_states)
    
    def __getattr__(self, name):
        if name == "model":
            return super().__getattr__(name)
        
        if hasattr(self.model, name):
            return getattr(self.model, name)
        
        return getattr(self.model.model, name)

def contains_any(input_string: str, string_list: list) -> bool:
    return any(substring in input_string for substring in string_list)

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, quant_config: QuantizeModel_config, device="cuda", exclude=None):
        super().__init__()
        self.model = model
        self.device = device
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        if hasattr(model, 'in_channels'):
            self.in_channels = model.in_channels
        self.specials = get_specials(True)

        self.exclude = None
        self.quant_config = quant_config
        self.first_layer_quant_config = copy.deepcopy(quant_config)
        self.first_layer_quant_config.weight_config.quant_bits = 8
        self.first_layer_quant_config.act_config.quant_bits = 8
        self.quant_module_refactor(self.model, quant_config)
        self.quant_block_refactor(self.model, quant_config)
        self.quant_layer_type = get_layer_type(quant_config)

    def quant_module_refactor(self, module: nn.Module, quant_config: QuantizeModel_config, module_name="model"):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """

        for name, child_module in module.named_children():
            whole_name = module_name + "." + name
            if self.exclude is not None and self.exclude in name:
                    continue
            elif isinstance(child_module, nn.Linear):
                if any(substring in whole_name for substring in first_layer_name):
                    print(f"{whole_name} is first layer, set precision to 8")
                    setattr(module, name, saw_Linear_QuantLayer(
                        child_module, self.first_layer_quant_config, self.device))
                else:
                    setattr(module, name, saw_Linear_QuantLayer(
                        child_module, quant_config, self.device))
            elif isinstance(child_module, nn.Conv1d):
                if any(substring in whole_name for substring in first_layer_name):
                    print(f"{whole_name} is first layer, set precision to 8")
                    setattr(module, name, saw_Conv1d_QuantLayer(
                        child_module, self.first_layer_quant_config, self.device))
                else:
                    setattr(module, name, saw_Conv1d_QuantLayer(
                        child_module, quant_config, self.device))
            elif isinstance(child_module, nn.Conv2d):
                if any(substring in whole_name for substring in first_layer_name):
                    print(f"{whole_name} is first layer, set precision to 8")
                    setattr(module, name, saw_Conv2d_QuantLayer(
                        child_module, self.first_layer_quant_config, self.device))
                else:
                    setattr(module, name, saw_Conv2d_QuantLayer(
                        child_module, quant_config, self.device))
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, quant_config, module_name=whole_name)

    def quant_block_refactor(self, module: nn.Module, quant_config: QuantizeModel_config):
        for name, child_module in module.named_children():
            if isinstance(child_module, MemoryEfficientCrossAttention):
                setattr(module, name, self.specials[type(child_module)](child_module,
                        quant_config))
            else:
                self.quant_block_refactor(child_module, quant_config)
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        quant_config))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        quant_config))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        quant_config))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        quant_config))
            else:
                self.quant_block_refactor(child_module, quant_config)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantLayer, QuantAttnBlock,
                              saw_Linear_QuantLayer, QuantBasicTransformerBlock,
                              saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)

    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)
            if isinstance(m, (saw_Linear_QuantLayer, saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                m.set_running_stat(running_stat)
            if isinstance(m, QuantAttnBlock):
                m.set_running_stat(running_stat)

    def set_record(self, record: bool):
        for m in self.model.modules():
            if isinstance(m, saw_Linear_QuantLayer):
                m.set_record(record)

    def get_all_norm_param(self):
        norm_list = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    norm_list.append(m.weight.requires_grad_(True))
                if hasattr(m, 'bias') and m.bias is not None:
                    norm_list.append(m.bias.requires_grad_(True))
        return norm_list

    def get_all_scale_factor(self, pick=None):
        scale_list = []
        if pick is None:
            for name, m in self.model.named_modules():
                if isinstance(m, (saw_Linear_QuantLayer, saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                    scale_list.append(m.scale_factor.requires_grad_(True))
                    scale_list.append(m.offset.requires_grad_(True))
                elif isinstance(m, QuantAttnBlock):
                    scale_list.append(m.scale_factor_qk.requires_grad_(True))
                    scale_list.append(m.scale_factor_vw.requires_grad_(True))
        else:
            if pick=="encoder":
                token = ["encoder", "quant_conv"]
            else:
                token = ["decoder", "post_quant_conv"]
            for name, m in self.model.named_modules():
                if any([t in name for t in token]):
                    if isinstance(m, (saw_Linear_QuantLayer, saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                        scale_list.append(m.scale_factor.requires_grad_(True))
                        scale_list.append(m.offset.requires_grad_(True))
                    elif isinstance(m, QuantAttnBlock):
                        scale_list.append(m.scale_factor_qk.requires_grad_(True))
                        scale_list.append(m.scale_factor_vw.requires_grad_(True))
        return scale_list

    def get_all_quant_param(self, user="all", pick=None):
        param_list = []
        if pick is None:
            for name, m in self.model.named_modules():
                if isinstance(m, BasicQuantLayer):
                    if user == "all" or m.quantize_config.user == user:
                        param_list.append(m.delta)
                        if not m.sym:
                            param_list.append(m.zero_point)
                elif isinstance(m, Basic2DquantLayer):
                    if user == "all" or m.quantize_config.user == user:
                        param_list.append(m.x_max)
                        param_list.append(m.x_min)
        else:
            if pick=="encoder":
                token = ["encoder", "quant_conv"]
            else:
                token = ["decoder", "post_quant_conv"]
            for name, m in self.model.named_modules():
                # print(name)
                if any([t in name for t in token]):
                    if isinstance(m, BasicQuantLayer):
                        if user == "all" or m.quantize_config.user == user:
                            param_list.append(m.delta)
                            if not m.sym:
                                param_list.append(m.zero_point)
                    elif isinstance(m, Basic2DquantLayer):
                        if user == "all" or m.quantize_config.user == user:
                            param_list.append(m.x_max)
                            param_list.append(m.x_min)
        return param_list

    def get_all_lora_param(self):
        lora_params = []
        for name, m in self.model.named_modules():
            if isinstance(m, (saw_Linear_QuantLayer, QuantLayer,
                              saw_Conv2d_QuantLayer, saw_Conv1d_QuantLayer)):
                if m.use_lora:
                    lora_params.append(m.lora_layer.lora_A)
                    lora_params.append(m.lora_layer.lora_B)
        return lora_params

    def set_all_init(self, flag, user="all"):
        for name, child_module in self.model.named_modules():
            if user == "all" or user == 'quant':
                if isinstance(child_module, self.quant_layer_type):
                    if isinstance(child_module, BasicQuantLayer):
                        if child_module.delta.numel() == 1:
                            if child_module.delta.data.item() == 1.0:
                                child_module.set_init_state(False)
                        else:
                            child_module.set_init_state(flag)
                    elif isinstance(child_module, Basic2DquantLayer):
                        if child_module.x_max.numel() == 1:
                            if child_module.x_max.data.item() == 0.5 and child_module.x_min.data.item() == -0.5:
                                child_module.set_init_state(False)
                                print(name)
                        else:
                            child_module.set_init_state(flag)
                    child_module.set_init_state(flag)
            # if user == "all" or user == 'attn':
            #     if isinstance(child_module, QuantAttnBlock):
            #         child_module.scale_init_flag = flag

    def set_all_recon_init(self, flag):
        for name, child_module in self.model.named_modules():
            if isinstance(child_module, (saw_Linear_QuantLayer,
                            saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                child_module.recon_inited = flag
            if isinstance(child_module, QuantAttnBlock):
                child_module.scale_init_flag = flag

    def set_all_recon(self, flag):
        for name, child_module in self.model.named_modules():
            if isinstance(child_module, (saw_Linear_QuantLayer,
                                        saw_Conv1d_QuantLayer, saw_Conv2d_QuantLayer)):
                child_module.recon_flag = flag
            if isinstance(child_module, QuantAttnBlock):
                child_module.recon_flag = flag
                
def configure_train_component(quant_config, train_params):
    optimizer = torch.optim.Adam(train_params, lr=float(quant_config["cali_learning_rate"]))
    criterion = get_loss_function(quant_config["loss_function"])
    if "scheduler" in quant_config.keys():
        milestones = quant_config["scheduler"]["milestones"]
        gamma = quant_config["scheduler"]["gamma"]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        scheduler = None
    psnr_loss = get_loss_function("psnr")
    return optimizer, criterion, scheduler, psnr_loss