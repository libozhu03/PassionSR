import torch
import torch.nn as nn
from .quant_config import Quantize_config, QuantizeModel_config
from .quant_layer import BasicQuantLayer, QuantLayer
from tqdm import tqdm, trange
import numpy as np
from typing import Union
import torch.nn.functional as F

class saw_Conv1d_QuantLayer(QuantLayer):
    """
    A quantized 1D convolutional layer with scale and offset adaptation for quantization-aware training.

    This layer extends a base QuantLayer to support quantization of Conv1d operations, including adaptive scaling and offsetting
    of inputs and weights for improved quantization performance. It supports optional LoRA (Low-Rank Adaptation), activation and 
    weight quantization, and running statistics for scale updating.

    Args:
        org_module (nn.Conv2d): The original convolutional module to be quantized.
        quantize_config: Configuration object for quantization parameters.
        device (str, optional): Device to place the layer parameters on. Default is "cuda".
        s_alpha (float, optional): Exponent for scale adaptation. Default is 0.5.

    Attributes:
        quant_config: Stores the quantization configuration.
        input_embedd: Placeholder for input embedding (if used).
        record_flag (bool): Flag to enable input recording.
        device (str): Device for computation.
        scale_factor (nn.Parameter): Learnable scale factor for input normalization.
        offset (nn.Parameter): Learnable offset for input normalization.
        recon_flag (bool): Flag to enable reconstruction mode.
        recon_inited (bool): Indicates if reconstruction has been initialized.
        s_alpha (float): Exponent for scale adaptation.
        running_stat (bool): Flag to enable running statistics for scale updating.

    Methods:
        init_scale(input): Initializes the scale factor based on input and weight statistics.
        set_running_stat(running_stat): Sets the running statistics flag.
        update_scale(input, update_rate=0.1): Updates the scale factor using running statistics.
        set_record(flag): Sets the input recording flag.
        forward(input): Forward pass with quantization, scaling, offsetting, and optional LoRA and reconstruction.
    """
    def __init__(self,
                 org_module: nn.Conv2d,
                 quantize_config,
                 device="cuda",
                 s_alpha = 0.5
                 ):
        super().__init__(org_module, quantize_config, device)
        self.quant_config = quantize_config
        self.input_embedd = None
        self.record_flag = False
        self.device = device
        scale_factor = torch.ones(1, self.org_weight.shape[1], 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        offset = torch.zeros(1, self.org_weight.shape[1], 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.offset = nn.Parameter(offset, requires_grad=True)
        self.recon_flag = False
        self.recon_inited = False
        if hasattr(quantize_config, 's_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False

    def init_scale(self, input):
        self.recon_inited = True
        print("Input shape: ", input.shape)
        print("weight shape:", self.weight.shape)
        weight_max = torch.amax(torch.abs(self.weight), dim=(0,2), keepdim=True).clamp(min=1e-5)
        input_max = torch.amax(torch.abs(input), dim=(0,2), keepdim=True).clamp(min=1e-5)
        self.scale_factor.data = input_max ** self.s_alpha / weight_max ** self.s_alpha
        return self.scale_factor

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def update_scale(self, input, update_rate=0.1):
        weight_max = torch.amax(torch.abs(self.weight), dim=(0,2), keepdim=True).clamp(min=1e-5)
        input_max = torch.amax(torch.abs(input), dim=(0,2), keepdim=True).clamp(min=1e-5)
        tmp = input_max ** self.s_alpha / weight_max ** self.s_alpha
        self.scale_factor.data = self.scale_factor.data * (1 - update_rate) + tmp * update_rate

    def set_record(self, flag):
        self.record_flag = flag

    def forward(self, input):
        if self.record_flag:
            self.record_x(input)
        if self.recon_flag:
            if not self.recon_inited:
                self.init_scale(input)
            elif self.running_stat:
                self.update_scale(input)
            offset = self.offset.expand_as(input)
            input = (input - offset) / self.scale_factor
            recond_bias = self.fwd_func(offset, self.weight, None, **self.fwd_kwargs)
            recond_bias = recond_bias.squeeze(1)
            # weight = nn.Parameter(self.weight / self.scale_factor)
            weight = self.weight * self.scale_factor
        else:
            input = input
            weight = self.weight

        if self.use_act_quant:
            if self.split:
                split_divide_line = input.shape[1] // 2
                input_0 = self.act_quantizer_0(input[:, :split_divide_line, ...])
                input_1 = self.act_quantizer_1(input[:, split_divide_line:, ...])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                input = self.act_quantizer_0(input)
        else:
            input = input

        if self.use_weight_quant:
            if self.split:
                split_divide_line = self.weight.shape[1] // 2
                weight_0 = self.weight_quantizer_0(weight[:, :split_divide_line, ...])
                weight_1 = self.weight_quantizer_1(weight[:, split_divide_line:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer_0(weight)
            # bias = self.bias
            if self.org_bias is not None:
                bias = self.bias_quantizer(self.bias)
            else:
                bias = self.bias
        else:
            # weight = self.org_weight
            weight = weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        if self.recon_flag:
            out = out + recond_bias
        out = self.activation_function(out)
        return out

class saw_Conv2d_QuantLayer(QuantLayer):
    """
    A quantized 2D convolutional layer with scale and offset adjustment for input reconstruction.

    This layer extends a standard quantized Conv2d layer by introducing learnable scale and offset parameters
    for each input channel, enabling input reconstruction and improved quantization performance. It supports
    both activation and weight quantization, as well as optional input recording for analysis.

    Args:
        org_module (nn.Conv2d): The original Conv2d module to be quantized.
        quantize_config: Configuration object for quantization parameters.
        device (str, optional): Device to place the layer parameters on. Default is "cuda".
        s_alpha (float, optional): Exponent for scale factor calculation. Default is 0.5.

    Attributes:
        quant_config: Stores the quantization configuration.
        input_embedd: Stores recorded input embeddings if recording is enabled.
        record_flag (bool): Flag to enable/disable input recording.
        device (str): Device for computation.
        scale_factor (nn.Parameter): Learnable scale factor for each input channel.
        offset (nn.Parameter): Learnable offset for each input channel.
        recon_flag (bool): Flag to enable/disable input reconstruction.
        recon_inited (bool): Indicates if reconstruction parameters have been initialized.
        s_alpha (float): Exponent used in scale factor calculation.
        running_stat (bool): Flag to enable/disable running statistics for scale update.

    Methods:
        init_scale(input): Initializes the scale factor based on input and weight statistics.
        set_running_stat(running_stat): Sets the running statistics flag.
        update_scale(input, update_rate=0.1): Updates the scale factor using running statistics.
        set_record(flag): Enables or disables input recording.
        forward(input): Forward pass with quantization, optional reconstruction, and activation.
        record_x(x): Records input embeddings for analysis.
    """
    def __init__(self,
                org_module: nn.Conv2d,
                quantize_config,
                device="cuda",
                s_alpha = 0.5,
                ):
        super().__init__(org_module, quantize_config, device)
        self.quant_config = quantize_config
        self.input_embedd = None
        self.record_flag = False
        self.device = device
        scale_factor = torch.ones(1, self.org_weight.shape[1], 1, 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        offset = torch.zeros(1, self.org_weight.shape[1], 1, 1, requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.offset = nn.Parameter(offset, requires_grad=True)
        self.recon_flag = False
        self.recon_inited = False
        if hasattr(quantize_config, 's_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False

    def init_scale(self, input):
        self.recon_inited = True
        weight_max = torch.amax(torch.abs(self.weight), dim=(0,2,3), keepdim=True).clamp(min=1e-5)
        input_max = torch.amax(torch.abs(input), dim=(0,2,3), keepdim=True).clamp(min=1e-5)
        self.scale_factor.data = input_max ** self.s_alpha / weight_max ** self.s_alpha
        return self.scale_factor

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def update_scale(self, input, update_rate=0.1):
        weight_max = torch.amax(torch.abs(self.weight), dim=(0,2,3), keepdim=True).clamp(min=1e-5)
        input_max = torch.amax(torch.abs(input), dim=(0,2,3), keepdim=True).clamp(min=1e-5)
        tmp = input_max ** self.s_alpha / weight_max ** self.s_alpha
        self.scale_factor.data = self.scale_factor.data * (1 - update_rate) + tmp * update_rate

    def set_record(self, flag):
        self.record_flag = flag

    def forward(self, input):
        if self.record_flag:
            self.record_x(input)
        if self.recon_flag:
            if not self.recon_inited:
                self.init_scale(input)
            elif self.running_stat:
                self.update_scale(input)
            offset = self.offset.expand_as(input)
            input = (input - offset) / self.scale_factor
            recond_bias = self.fwd_func(offset, self.weight, None, **self.fwd_kwargs)
            recond_bias = recond_bias.squeeze(1)
            # weight = nn.Parameter(self.weight / self.scale_factor)
            weight = self.weight * self.scale_factor
        else:
            input = input
            weight = self.weight
        if self.use_act_quant:
            if self.split:
                split_divide_line = input.shape[1] // 2
                input_0 = self.act_quantizer_0(input[:, :split_divide_line, ...])
                input_1 = self.act_quantizer_1(input[:, split_divide_line:, ...])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                input = self.act_quantizer_0(input)
        else:
            input = input

        if self.use_weight_quant:
            if self.split:
                split_divide_line = self.weight.shape[1] // 2
                weight_0 = self.weight_quantizer_0(weight[:, :split_divide_line, ...])
                weight_1 = self.weight_quantizer_1(weight[:, split_divide_line:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer_0(weight)
            # bias = self.bias
            if self.org_bias is not None:
                bias = self.bias_quantizer(self.bias)
            else:
                bias = self.bias
        else:
            weight = weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        if self.recon_flag:
            out = out + recond_bias
        out = self.activation_function(out)
        return out

    def record_x(self, x):
        x_clone = x.clone().detach().cpu()
        if self.input_embedd is None:
            self.input_embedd = x_clone
        else:
            self.input_embedd = torch.cat((self.input_embedd, x_clone), dim=0)

class saw_Linear_QuantLayer(QuantLayer):
    """
    A quantized linear layer with scale and offset adaptation for weight and activation quantization.

    This layer wraps a standard nn.Linear module and applies quantization to its weights and activations,
    supporting adaptive scaling and offset for improved quantization accuracy. It provides mechanisms for
    recording input embeddings, updating scale factors based on running statistics, and reconstructing outputs
    with learned scale and offset parameters.

    Args:
        org_module (nn.Linear): The original linear module to be quantized.
        quantize_config (QuantizeModel_config): Configuration object specifying quantization parameters.
        device (str, optional): Device to place the quantized parameters on. Default is "cuda".
        s_alpha (float, optional): Exponent for scale factor computation. Default is 0.5.

    Attributes:
        quant_config: Stores the quantization configuration.
        input_embedd: Buffer for recorded input embeddings.
        record_flag (bool): Flag to enable input recording.
        device (str): Device for parameters.
        scale_factor (nn.Parameter): Learnable scale factor for input normalization.
        offset (nn.Parameter): Learnable offset for input normalization.
        recon_flag (bool): Flag to enable reconstruction mode.
        recon_inited (bool): Indicates if reconstruction parameters are initialized.
        s_alpha (float): Exponent for scale factor computation.
        running_stat (bool): Flag to enable running statistics for scale update.

    Methods:
        init_scale(input): Initializes the scale factor based on input and weight statistics.
        set_running_stat(running_stat): Sets the running statistics flag.
        update_scale(input, update_rate): Updates the scale factor using running statistics.
        set_record(flag): Enables or disables input recording.
        forward(input): Forward pass with quantization, scaling, and optional reconstruction.
        record_x(x): Records input embeddings for analysis or calibration.
    """
    def __init__(self,
                org_module: nn.Linear,
                quantize_config : QuantizeModel_config,
                device = "cuda",
                s_alpha = 0.5,
                ):
        super(saw_Linear_QuantLayer, self).__init__(org_module, quantize_config)
        self.quant_config = quantize_config
        self.input_embedd = None
        self.record_flag = False
        self.device = device
        scale_factor = torch.ones(1, self.org_weight.shape[1], requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        offset = torch.zeros(1, self.org_weight.shape[1], requires_grad=True,
                                       dtype=torch.float32, device=self.device)
        self.offset = nn.Parameter(offset, requires_grad=True)
        self.recon_flag = False
        self.recon_inited = False
        if hasattr(quantize_config, 's_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False

    def init_scale(self, input):
        self.recon_inited = True
        weight_max = torch.amax(torch.abs(self.weight), dim=(0), keepdim=True).clamp(min=1e-5)
        input_max = torch.amax(torch.abs(input), dim=(0,1), keepdim=True).clamp(min=1e-5)
        self.scale_factor.data = input_max ** self.s_alpha / weight_max ** self.s_alpha
        if len(self.scale_factor.shape) == 3:
            self.scale_factor.data = self.scale_factor.data.squeeze(0)
        return self.scale_factor

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def update_scale(self, input, update_rate=0.1):
        weight_max = torch.amax(torch.abs(self.weight), dim=(0), keepdim=True).clamp(min=1e-5)
        input_max = torch.amax(torch.abs(input), dim=(0,1), keepdim=True).clamp(min=1e-5)
        tmp = input_max ** self.s_alpha / weight_max ** self.s_alpha
        self.scale_factor.data = self.scale_factor.data * (1 - update_rate) + tmp * update_rate
        if len(self.scale_factor.shape) == 3:
            self.scale_factor.data = self.scale_factor.data.squeeze(0)

    def set_record(self, flag):
        self.record_flag = flag

    def forward(self, input):
        if self.record_flag:
            self.record_x(input)
        if self.recon_flag:
            if not self.recon_inited:
                self.init_scale(input)
            elif self.running_stat:
                self.update_scale(input)
            offset = self.offset.expand_as(input)
            input = (input + offset) / self.scale_factor
            recond_bias = self.weight @ self.offset.T
            recond_bias = recond_bias.squeeze(1)
            weight = self.weight * self.scale_factor
        else:
            input = input
            weight = self.weight
        if self.use_act_quant:
            if self.split:
                split_divide_line = input.shape[1] // 2
                input_0 = self.act_quantizer_0(input[:, :split_divide_line, ...])
                input_1 = self.act_quantizer_1(input[:, split_divide_line:, ...])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                input = self.act_quantizer_0(input)
        else:
            input = input

        if self.use_weight_quant:
            if self.split:
                split_divide_line = self.weight.shape[1] // 2
                weight_0 = self.weight_quantizer_0(weight[:, :split_divide_line, ...])
                weight_1 = self.weight_quantizer_1(weight[:, split_divide_line:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer_0(weight)
            if self.org_bias is not None:
                bias = self.bias_quantizer(self.bias)
            else:
                bias = self.bias
        else:
            weight = weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        if self.recon_flag:
            out = out + recond_bias.T
        out = self.activation_function(out)
        return out

    def record_x(self, x):
        x_clone = x.clone().detach().cpu()
        if self.input_embedd is None:
            self.input_embedd = x_clone
        else:
            self.input_embedd = torch.cat((self.input_embedd, x_clone), dim=0)

class saw_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(saw_Loss, self).__init__()

    def forward(self, pred, target):
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        cov = torch.mean((pred - pred_mean) * (target - target_mean))
        correlation = cov / (pred_std * target_std + 1e-8)  # 避免除以0
        loss = 1 - correlation
        return loss
    
class PSNRLoss(nn.Module):
    def __init__(self, max_pixel_value=1.0, eps=1e-10):
        super(PSNRLoss, self).__init__()
        self.max_pixel_value = max_pixel_value
        self.eps = eps

    def forward(self, img1, img2):
        mse_loss = nn.functional.mse_loss(img1, img2)
        mse_loss = torch.clamp(mse_loss, min=self.eps)
        psnr = 10 * torch.log10(self.max_pixel_value ** 2 / mse_loss)
        return psnr

def get_loss_function(loss_name):
    loss_dict = {
        'mse': nn.MSELoss(),
        'cross_entropy': nn.CrossEntropyLoss(),
        'l1': nn.L1Loss(),
        'bce': nn.BCELoss(),
        'bce_with_logits': nn.BCEWithLogitsLoss(),
        'psnr': PSNRLoss()
    }
    
    if loss_name in loss_dict:
        return loss_dict[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

