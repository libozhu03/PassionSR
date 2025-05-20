import torch
import torch.nn as nn
from .quant_config import Quantize_config, QuantizeModel_config
from typing import Union
import torch.nn.functional as F
from torch.autograd import Function

class Basic2DquantLayer(nn.Module):
    """
    A basic 2D quantization layer for PyTorch models.
    This layer supports both symmetric and asymmetric quantization, with configurable quantization bits,
    signed/unsigned quantization, and optional channel-wise quantization for weights. It maintains running
    statistics for activation quantization and supports initialization and momentum-based updates of quantization
    parameters.
    Args:
        quantize_config (Quantize_config): Configuration object specifying quantization parameters such as
            number of bits, signed/unsigned, symmetric/asymmetric, and user type ("weight" or "activation").
        delta (float, optional): Initial quantization scale (step size). Default is 1.
        zero_point (float, optional): Initial zero point for quantization. Default is 0.
    Attributes:
        quantize_config (Quantize_config): Stores the quantization configuration.
        quant_bits (int): Number of bits for quantization.
        sign (bool): Whether to use signed quantization.
        sym (bool): Whether to use symmetric quantization.
        delta (torch.Tensor): Quantization scale (step size).
        zero_point (torch.Tensor): Zero point for quantization.
        init_flag (bool): Indicates if the layer has been initialized.
        running_stat (bool): Indicates if running statistics are being used for activation quantization.
        x_min (nn.Parameter): Minimum value observed (per-channel or global).
        x_max (nn.Parameter): Maximum value observed (per-channel or global).
    Methods:
        set_init_state(flag=True): Set the initialization flag.
        set_running_stat(flag): Enable or disable running statistics for activation quantization.
        forward(x): Quantize and dequantize the input tensor `x`.
        Init_basicinfo(x): Initialize quantization parameters based on input tensor `x`.
        init_quantization_scale(x, channel_wise=False): Initialize quantization scale and zero point.
        get_tensor_maxmin(x, type="max"): Get per-channel or global max/min values from tensor `x`.
        channel_wise_reshape(value, shape): Reshape per-channel statistics to match input tensor shape.
        set_param(): Compute quantization scale (delta) and zero point based on current min/max.
        quantize(x, delta, zero_point): Quantize input tensor `x` using given scale and zero point.
        dequantize(x, delta, zero_point): Dequantize quantized tensor `x` using given scale and zero point.
        act_momentum_update(x, act_range_momentum=0.95): Update running min/max statistics for activations.
    Note:
        - This layer assumes the existence of a custom `Round` function for STE rounding.
        - The `Quantize_config` class must provide attributes: `quant_bits`, `sign`, `sym`, and `user`.
    """
    def __init__(self,quantize_config: Quantize_config, delta=1, zero_point=0,):
        super().__init__()
        self.quantize_config = quantize_config
        self.quant_bits = quantize_config.quant_bits
        self.sign = quantize_config.sign
        self.sym = quantize_config.sym
        self.delta, self.zero_point = torch.tensor(delta, dtype=float),torch.tensor(zero_point, dtype=float)
        self.init_flag = False
        self.running_stat = False
        self.x_min, self.x_max =  nn.Parameter(torch.tensor(-0.5, dtype=float)), nn.Parameter(torch.tensor(0.5, dtype=float))

    def set_init_state(self, flag=True):
        self.init_flag = flag

    def set_running_stat(self, flag):
        self.running_stat = flag

    def forward(self, x: torch.Tensor):
        if not self.init_flag:
            self.Init_basicinfo(x)
        if self.running_stat:
            self.act_momentum_update(x)
        delta, zero_point = self.set_param()
        self.delta = delta.detach().clone()
        self.zero_point = zero_point.detach().clone()
        x = self.quantize(x, delta, zero_point)
        x = self.dequantize(x, delta, zero_point)
        return x

    def Init_basicinfo(self, x):
        # print("Init_basicinfo")
        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            channel_wise = True
        else:
            channel_wise = False
        self.init_quantization_scale(x_clone, channel_wise)
        self.set_init_state()

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        x_clone = x.clone().detach()
        if channel_wise:
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()
        self.x_max.data, self.x_min.data = x_max, x_min
        self.delta, self.zero_point = self.set_param()

    def get_tensor_maxmin(self, x, type="max"):
        x_clone = x.clone().detach()
        if type == 'max':
            if len(x.shape) == 4:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.max(dim=-1)[0]
            else:
                x_max = x_clone.max()
            return x_max
        elif type == 'min':
            if len(x.shape) == 4:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                x_min = x_clone.min(dim=-1)[0]
            else:
                x_min = x_clone.min()
            return x_min
        else:
            return None

    def channel_wise_reshape(self, value, shape):
        if len(shape) == 4:
            value = value.view(-1, 1, 1, 1)
        elif len(shape) == 3:
            value = value.view(-1, 1, 1)
        elif len(shape) == 2:
            value = value.view(-1, 1)
        else:
            pass
        return value

    def set_param(self):
        delta, zero_point = None, None
        if self.sym:
            delta = torch.max(torch.abs(self.x_max), torch.abs(self.x_min)) / (2 ** (self.quant_bits - 1))
            zero_point = torch.zeros_like(self.delta)
        else:
            delta = (self.x_max - self.x_min) / (2 ** (self.quant_bits) - 1)
            if self.sign == True:
                # self.zero_point.data = round_ste((self.x_max + self.x_min) / ( 2 * self.delta))
                zero_point = Round.apply((self.x_max + self.x_min) / ( 2 * self.delta))
            else:
                # self.zero_point.data = round_ste(self.x_min / self.delta)
                zero_point = Round.apply(self.x_min / self.delta)
        return delta, zero_point

    def quantize(self, x: torch.Tensor, delta: torch.Tensor, zero_point: torch.Tensor):
        x = x / delta
        # x = torch.round(x)
        # x = round_ste(x)
        x = Round.apply(x)
        if not self.sym:
            x = x - zero_point
        if self.sign == True:
            x = torch.clamp(x, -2 ** (self.quant_bits - 1), 2 ** (self.quant_bits - 1) - 1)
        else:
            x = torch.clamp(x, 0, 2 ** self.quant_bits - 1)
        return x
    
    def dequantize(self, x: torch.Tensor, delta: torch.Tensor, zero_point: torch.Tensor):
        if not self.sym:
            x = x + zero_point
        x = x * delta
        return x

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.init_flag)

        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()

        self.x_min.data = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max.data = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        self.delta, self.zero_point = self.set_param()

class BasicQuantLayer(nn.Module):
    """
    A basic quantization layer for neural networks, supporting both symmetric and asymmetric quantization,
    as well as channel-wise and tensor-wise quantization. This layer can be used for quantizing weights or activations.
    Args:
        quantize_config (Quantize_config): Configuration object specifying quantization parameters such as
            number of bits, signed/unsigned, symmetric/asymmetric, and user type ("weight" or "activation").
        delta (float, optional): Initial quantization scale (step size). Default is 1.
        zero_point (float, optional): Initial zero point for quantization. Default is 0.
        load (bool, optional): If True, skips initialization of quantization parameters. Default is False.
    Attributes:
        quantize_config (Quantize_config): Stores the quantization configuration.
        quant_bits (int): Number of bits for quantization.
        sign (bool): Whether quantization is signed.
        sym (bool): Whether quantization is symmetric.
        delta (nn.Parameter): Learnable quantization scale.
        zero_point (nn.Parameter): Learnable zero point for asymmetric quantization.
        init_flag (bool): Indicates if the layer has been initialized.
        running_stat (bool): If True, updates quantization range using running statistics (momentum).
        x_min (nn.Parameter): Minimum value observed (used for scale calculation).
        x_max (nn.Parameter): Maximum value observed (used for scale calculation).
    Methods:
        set_init_state(flag=True): Sets the initialization flag.
        set_running_stat(flag): Enables or disables running statistics for activation range.
        forward(x): Applies quantization and dequantization to the input tensor.
        Init_basicinfo(x): Initializes quantization parameters based on input tensor statistics.
        init_quantization_scale(x, channel_wise=False): Initializes scale and zero point, optionally channel-wise.
        get_tensor_maxmin(x, type="max"): Computes channel-wise or tensor-wise min/max.
        channel_wise_reshape(value, shape): Reshapes min/max values for channel-wise quantization.
        set_param(): Updates delta and zero_point based on current min/max and quantization config.
        quantize(x): Quantizes the input tensor.
        dequantize(x): Dequantizes the quantized tensor.
        act_momentum_update(x, act_range_momentum=0.95): Updates min/max using exponential moving average.
    """
    def __init__(self, quantize_config: Quantize_config, delta=1, zero_point=0, load=False):
        super(BasicQuantLayer, self).__init__()
        self.quantize_config = quantize_config
        self.quant_bits = quantize_config.quant_bits
        self.sign = quantize_config.sign
        self.sym = quantize_config.sym
        self.delta, self.zero_point = nn.Parameter(torch.tensor(delta, dtype=float)), nn.Parameter(torch.tensor(zero_point, dtype=float))
        self.init_flag = load
        # self.x_min, self.x_max = None, None
        self.running_stat = False
        self.x_min, self.x_max =  nn.Parameter(torch.tensor(-0.5, dtype=float)), nn.Parameter(torch.tensor(0.5, dtype=float))

    def set_init_state(self, flag=True):
        self.init_flag = flag

    def set_running_stat(self, flag):
        self.running_stat = flag

    def forward(self, x: torch.Tensor):
        if not self.init_flag:
            self.Init_basicinfo(x)
        if self.running_stat:
            self.act_momentum_update(x)
        x = self.quantize(x)
        x = self.dequantize(x)
        return x

    def Init_basicinfo(self, x):
        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            channel_wise = True
        else:
            channel_wise = False
        self.init_quantization_scale(x, channel_wise)
        self.set_init_state()

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        x_clone = x.clone().detach()
        if channel_wise:
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()
        self.x_max.data, self.x_min.data = x_max, x_min
        self.set_param()

    def get_tensor_maxmin(self, x, type="max"):
        x_clone = x.clone().detach()
        if type == 'max':
            if len(x.shape) == 4:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.max(dim=-1)[0]
            else:
                x_max = x_clone.max()
            return x_max
        elif type == 'min':
            if len(x.shape) == 4:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_min = x_clone.min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                x_min = x_clone.min(dim=-1)[0]
            else:
                x_min = x_clone.min()
            return x_min
        else:
            return None

    def channel_wise_reshape(self, value, shape):
        if len(shape) == 4:
            value = value.view(-1, 1, 1, 1)
        elif len(shape) == 3:
            value = value.view(-1, 1, 1)
        elif len(shape) == 2:
            value = value.view(-1, 1)
        else:
            pass
        return value

    def set_param(self):
        if self.sym:
            self.delta.data = torch.max(torch.abs(self.x_max), torch.abs(self.x_min)) / (2 ** (self.quant_bits - 1))
            self.zero_point.data = nn.Parameter(torch.zeros_like(self.delta))
        else:
            self.delta.data = (self.x_max - self.x_min)
            self.delta.data = self.delta / (2 ** (self.quant_bits) - 1)
            if self.sign == True:
                # self.zero_point.data = round_ste((self.x_max + self.x_min) / ( 2 * self.delta))
                self.zero_point.data = Round.apply((self.x_max + self.x_min) / ( 2 * self.delta))
            else:
                # self.zero_point.data = round_ste(self.x_min / self.delta)
                self.zero_point.data = Round.apply(self.x_min / self.delta)

    def quantize(self, x: torch.Tensor):
        x = x / self.delta
        # x = torch.round(x)
        # x = round_ste(x)
        x = Round.apply(x)
        if not self.sym:
            x = x - self.zero_point
        if self.sign == True:
            x = torch.clamp(x, -2 ** (self.quant_bits - 1), 2 ** (self.quant_bits - 1) - 1)
        else:
            x = torch.clamp(x, 0, 2 ** self.quant_bits - 1)
        return x
    
    def dequantize(self, x: torch.Tensor):
        if not self.sym:
            x = x + self.zero_point
        x = x * self.delta
        return x

    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.init_flag)

        x_clone = x.clone().detach()
        if self.quantize_config.user == "weight":
            x_clone = x.clone().detach()
            x_max = self.get_tensor_maxmin(x_clone, type="max")
            x_max = self.channel_wise_reshape(x_max, x.shape)
            x_min = self.get_tensor_maxmin(x_clone, type="min")
            x_min = self.channel_wise_reshape(x_min, x.shape)
        else:
            x_max = x_clone.max().clone()
            x_min = x_clone.min().clone()

        self.x_min.data = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max.data = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        self.set_param()

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class QuantLayer(nn.Module):
    """
    A quantization wrapper layer for PyTorch modules (Conv2d, Conv1d, Linear).
    This layer replaces the original module's forward pass with a quantized version,
    supporting both weight and activation quantization, as well as optional channel splitting
    for separate quantization of input/weight channels.
    Args:
        org_module (Union[nn.Conv2d, nn.Linear, nn.Conv1d]): The original PyTorch module to be quantized.
        quantize_config (QuantizeModel_config): Configuration object specifying quantization parameters.
        device (str, optional): Device to use for computation. Default is "cuda".
    Attributes:
        quantize_config (QuantizeModel_config): Stores quantization configuration.
        device (str): Device for computation.
        quant_layer_type (type): Quantizer class determined by configuration.
        fwd_kwargs (dict): Forward function keyword arguments (stride, padding, etc.).
        fwd_func (callable): Forward function (F.conv2d, F.conv1d, or F.linear).
        weight (torch.Tensor): Quantized weight tensor.
        org_weight (torch.Tensor): Original weight tensor (for dequantization or reference).
        bias (torch.Tensor or None): Quantized bias tensor.
        org_bias (torch.Tensor or None): Original bias tensor.
        bias_quantizer (callable or None): Quantizer for bias.
        use_weight_quant (bool): Whether to apply weight quantization.
        use_act_quant (bool): Whether to apply activation quantization.
        weight_quantizer_0 (callable): Quantizer for weights (or first split).
        act_quantizer_0 (callable): Quantizer for activations (or first split).
        split (bool): Whether to split channels for separate quantization.
        weight_quantizer_1 (callable): Quantizer for second split of weights (if split enabled).
        act_quantizer_1 (callable): Quantizer for second split of activations (if split enabled).
        activation_function (callable): Activation function (StraightThrough).
    Methods:
        set_split(split): Enable or disable channel splitting for quantization.
        forward(input): Forward pass with quantization applied to weights and activations.
        set_quant_state(weight_quant, act_quant): Enable/disable weight and activation quantization.
        set_running_stat(running_stat): Set running statistics mode for activation quantizers.
    """
    def __init__(
        self,
        org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
        quantize_config : QuantizeModel_config,
        device = "cuda",
        # split : bool = False,
    ) -> None:
        super(QuantLayer, self).__init__()
        self.quantize_config = quantize_config
        self.device = device
        self.quant_layer_type = get_layer_type(quantize_config)

        # self.org_module = org_module
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight.data
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias.data
            self.org_bias = org_module.bias.data.clone()
            self.bias_quantizer = self.quant_layer_type(self.quantize_config.weight_config)
        else:
            self.bias = None
            self.org_bias = None
            self.bias_quantizer = None

        self.use_weight_quant = True
        self.use_act_quant = True
        self.weight_quantizer_0 = self.quant_layer_type(self.quantize_config.weight_config)
        self.act_quantizer_0 = self.quant_layer_type(self.quantize_config.act_config)
        self.set_split(quantize_config.split)

        self.activation_function = StraightThrough()

    def set_split(self, split):
        self.split = split
        if self.split:
            self.weight_quantizer_1 = self.quant_layer_type(self.quantize_config.weight_config)
            self.act_quantizer_1 = self.quant_layer_type(self.quantize_config.act_config)

    def forward(self, input: torch.Tensor):
        if self.use_act_quant:
            if self.split:
                split_divide_line = input.shape[1] // 2
                input_0 = self.act_quantizer_0(input[:, :split_divide_line, ...])
                input_1 = self.act_quantizer_1(input[:, split_divide_line:, ...])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                input = self.act_quantizer_0(input)
        if self.use_weight_quant:
            if self.split:
                split_divide_line = self.weight.shape[1] // 2
                weight_0 = self.weight_quantizer_0(self.weight[:, :split_divide_line, ...])
                weight_1 = self.weight_quantizer_1(self.weight[:, split_divide_line:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer_0(self.weight)
            # bias = self.bias
            if self.org_bias is not None:
                bias = self.bias_quantizer(self.bias)
            else:
                bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def set_running_stat(self, running_stat: bool):
        self.act_quantizer_0.set_running_stat(running_stat)
        if self.split != 0:
            self.act_quantizer_1.set_running_stat(running_stat)

def get_layer_type(quant_config: QuantizeModel_config):
    if quant_config.layer_type == "normal_quant":
        return BasicQuantLayer
    elif quant_config.layer_type == "2Dquant":
        return Basic2DquantLayer
    else:
        raise NotImplementedError(f"Quantization layer {quant_config.layer_type} not implemented")