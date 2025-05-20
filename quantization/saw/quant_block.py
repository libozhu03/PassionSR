import torch
import torch.nn as nn
from einops import rearrange, repeat
from types import MethodType
from .quant_config import Quantize_config, QuantizeModel_config
from .quant_layer import BasicQuantLayer, QuantLayer, StraightThrough, get_layer_type, Basic2DquantLayer
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import exists, default
from ldm.modules.attention import MemoryEfficientCrossAttention
from ldm.modules.diffusionmodules.model import ResnetBlock, AttnBlock, nonlinearity

class BaseQuantBlock(nn.Module):
    """
    BaseQuantBlock is a base class for quantization-aware neural network blocks.

    Args:
        quantize_config (QuantizeModel_config): Configuration object specifying quantization parameters.

    Attributes:
        use_weight_quant (bool): Flag indicating whether weight quantization is enabled.
        use_act_quant (bool): Flag indicating whether activation quantization is enabled.
        recon_flag (bool): Flag for reconstruction mode (default: False).
        quantize_config (QuantizeModel_config): Stores the quantization configuration.
        quant_layer_type (type): The quantization layer type determined by the configuration.
        activation_function (nn.Module): The activation function used, typically a straight-through estimator.

    Methods:
        set_quant_state(weight_quant: bool = False, act_quant: bool = False):
            Sets the quantization state for weights and activations for all QuantLayer modules in the block.

        set_running_stat(running_stat):
            Sets the running statistics for all QuantLayer modules in the block.
    """
    def __init__(self, quantize_config: QuantizeModel_config):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.recon_flag = False
        self.quantize_config = quantize_config
        self.quant_layer_type = get_layer_type(quantize_config)
        # self.act_quantizer = self.quant_layer_type(self.quantize_config.act_config)
        self.activation_function = StraightThrough()

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(weight_quant, act_quant)

    def set_running_stat(self, running_stat):
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)

def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)
    # print(q.shape)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    scale = 1.0 / q.shape[-1] ** 0.5
    if self.use_act_quant:
        if self.recon_flag:
            if not self.scale_init_flag:
                self.init_scale_factor_qk(q, k, q.device)
            elif self.running_stat:
                self.unpdate_scale_factor_qk(q, k)
            q = torch.einsum("b i d, d -> b i d", q, self.scale_factor_qk)
            k = torch.einsum("b j d, d -> b j d", k, torch.reciprocal(self.scale_factor_qk))
        quant_q = self.act_quantizer_q(q)
        quant_k = self.act_quantizer_k(k)
        # sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
        sim = torch.einsum('b i d, b j d -> b i j', quant_q, quant_k) * scale
    else:
        # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)


    if self.use_act_quant:
        if self.recon_flag:
            if not self.scale_init_flag:
                self.init_scale_factor_vw(v, attn, v.device)
                self.scale_init_flag = True
            elif self.running_stat:
                self.unpdate_scale_factor_vw(v, attn)
            attn = torch.einsum("b i j, j -> b i j", attn, self.scale_factor_vw)
            v = torch.einsum("b j d, j -> b j d", v,  torch.reciprocal(self.scale_factor_vw) )
        out = torch.einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
    else:
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

class QuantBasicTransformerBlock(BaseQuantBlock):
    """
    A quantized version of a basic transformer block, enabling activation and weight quantization
    for attention and feed-forward layers.
    Args:
        tran (BasicTransformerBlock): The original transformer block to be quantized.
        quantize_config (QuantizeModel_config): Configuration for quantization, including
            activation and weight quantization settings.
    Attributes:
        attn1: The first attention layer, with quantization hooks.
        attn2: The second attention layer (cross-attention), with quantization hooks.
        ff: The feed-forward layer.
        norm1, norm2, norm3: Normalization layers.
        checkpoint (bool): Whether to use gradient checkpointing for memory efficiency.
        quant_layer_type: The quantization layer class/type, determined by config.
        use_weight_quant (bool): Whether to enable weight quantization.
        use_act_quant (bool): Whether to enable activation quantization.
    Methods:
        forward(x, context=None):
            Forward pass with optional context, using checkpointing if enabled.
        _forward(x, context=None):
            Internal forward pass: applies attention, cross-attention, and feed-forward layers
            with residual connections.
        set_quant_state(weight_quant=False, act_quant=False):
            Sets the quantization state for weights and activations for all quantized layers
            within the block.
    """
    def __init__(
        self, tran: BasicTransformerBlock,
        quantize_config: QuantizeModel_config,
        ):
        super().__init__(quantize_config)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2

        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        # self.checkpoint = False
        self.quant_layer_type = get_layer_type(quantize_config)
        # logger.info(f"quant attn matmul")
        self.attn1.act_quantizer_q = self.quant_layer_type(self.quantize_config.act_config)
        self.attn1.act_quantizer_k = self.quant_layer_type(self.quantize_config.act_config)
        self.attn1.act_quantizer_v = self.quant_layer_type(self.quantize_config.act_config)

        self.attn2.act_quantizer_q = self.quant_layer_type(self.quantize_config.act_config)
        self.attn2.act_quantizer_k = self.quant_layer_type(self.quantize_config.act_config)
        self.attn2.act_quantizer_v = self.quant_layer_type(self.quantize_config.act_config)

        self.attn1.act_quantizer_w = self.quant_layer_type(self.quantize_config.act_config)
        self.attn2.act_quantizer_w = self.quant_layer_type(self.quantize_config.act_config)

        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False

        self.attn1.scale_factor_qk = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn1.scale_factor_vw = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn1.scale_init_flag = False

        self.attn2.scale_factor_qk = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn2.scale_factor_vw = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.attn2.scale_init_flag = False

    def forward(self, x, context=None):
        # print(f"x shape {x.shape} context shape {context.shape}")
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(weight_quant, act_quant)

class QuantAttnBlock(BaseQuantBlock):
    """
    Quantized Attention Block for neural network quantization.

    This class wraps an attention block with quantization-aware operations, enabling quantization of activations
    and scaling factors for efficient inference. It supports initialization and updating of scale factors for
    query-key and value-attention computations, and applies quantization to the attention mechanism.

    Args:
        attn (AttnBlock): The original attention block to be quantized.
        quantize_config (QuantizeModel_config): Configuration for quantization, including activation quantizer settings.
        s_alpha (float, optional): Exponent for scale factor computation. Defaults to 0.5.

    Attributes:
        in_channels (int): Number of input channels.
        heads (int): Number of attention heads.
        norm (nn.Module): Normalization layer (uses StraightThrough for quantization).
        to_q, to_k, to_v, to_out (nn.Module): Linear layers for query, key, value, and output projections.
        quant_layer_type (type): Quantization layer type from configuration.
        act_quantizer_q, act_quantizer_k, act_quantizer_v, act_quantizer_w (nn.Module): Activation quantizers for Q, K, V, and attention weights.
        scale_factor_qk (nn.Parameter): Learnable scale factor for query-key.
        scale_factor_vw (nn.Parameter): Learnable scale factor for value-attention.
        scale_init_flag (bool): Flag indicating if scale factors are initialized.
        s_alpha (float): Exponent for scale factor computation.
        running_stat (bool): Whether to use running statistics for quantization.

    Methods:
        set_running_stat(running_stat): Set the running statistics flag.
        init_scale_factor_qk(q, k, device): Initialize scale factor for query-key.
        unpdate_scale_factor_qk(q, k, update_rate=0.1): Update scale factor for query-key.
        init_scale_factor_vw(v, attn, device): Initialize scale factor for value-attention.
        unpdate_scale_factor_vw(v, attn, update_rate=0.1): Update scale factor for value-attention.
        forward(x): Forward pass with quantized attention computation.

    Forward Pass:
        - Applies normalization and projects input to Q, K, V.
        - Optionally applies scaling and quantization to Q, K, V, and attention weights.
        - Computes attention weights and applies them to values.
        - Projects output and adds residual connection.
    """
    def __init__(
        self,
        attn: AttnBlock,
        quantize_config:QuantizeModel_config,
        s_alpha = 0.5
        ):
        super().__init__(quantize_config)
        self.in_channels = attn.query_dim
        self.heads = attn.heads
        # self.norm = attn.norm
        self.norm = StraightThrough()
        self.to_q = attn.to_q
        self.to_k = attn.to_k
        self.to_v = attn.to_v
        self.to_out = attn.to_out

        self.quant_layer_type = get_layer_type(quantize_config)
        self.act_quantizer_q = self.quant_layer_type(self.quantize_config.act_config)
        self.act_quantizer_k = self.quant_layer_type(self.quantize_config.act_config)
        self.act_quantizer_v = self.quant_layer_type(self.quantize_config.act_config)
        self.act_quantizer_w = self.quant_layer_type(self.quantize_config.act_config)
        self.scale_factor_qk = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.scale_factor_vw = nn.Parameter(torch.ones(1, requires_grad=True, \
                                    dtype=torch.float32))
        self.scale_init_flag = False
        if hasattr(quantize_config,'s_alpha'):
            self.s_alpha = quantize_config.s_alpha
        else:
            self.s_alpha = s_alpha
        self.running_stat = False

    def set_running_stat(self, running_stat):
        self.running_stat = running_stat

    def init_scale_factor_qk(self, q, k, device):
        self.scale_factor_qk.data = torch.ones(q.shape[2], requires_grad=True, \
                                    dtype=torch.float32, device=device)
        q_max = torch.amax(torch.abs(q), dim=(0,1)).clamp(min=1e-5).squeeze()
        k_max = torch.amax(torch.abs(k), dim=(0,1)).clamp(min=1e-5).squeeze()
        tmp = k_max ** self.s_alpha / q_max ** self.s_alpha
        self.scale_factor_qk.data = tmp

    def unpdate_scale_factor_qk(self, q, k, update_rate=0.1):
        q_max = torch.amax(torch.abs(q), dim=(0,1)).clamp(min=1e-5).squeeze()
        k_max = torch.amax(torch.abs(k), dim=(0,1)).clamp(min=1e-5).squeeze()
        tmp = k_max ** self.s_alpha / q_max ** self.s_alpha
        self.scale_factor_qk.data =  self.scale_factor_qk.data * (1 - update_rate) + tmp * update_rate

    def init_scale_factor_vw(self, v, attn, device):
        self.scale_factor_vw.data = torch.ones(v.shape[1], requires_grad=True, \
                                    dtype=torch.float32, device=device)
        v_max = torch.amax(torch.abs(v), dim=(0,2)).clamp(min=1e-5).squeeze()
        attn_max = torch.amax(torch.abs(attn), dim=(0,1)).clamp(min=1e-5).squeeze()
        self.scale_factor_vw.data =  v_max ** self.s_alpha / attn_max ** self.s_alpha
        # self.scale_init_flag = True

    def unpdate_scale_factor_vw(self, v, attn, update_rate=0.1):
        v_max = torch.amax(torch.abs(v), dim=(0,2)).clamp(min=1e-5).squeeze()
        attn_max = torch.amax(torch.abs(attn), dim=(0,1)).clamp(min=1e-5).squeeze()
        tmp = v_max ** self.s_alpha / attn_max ** self.s_alpha
        self.scale_factor_vw.data =  self.scale_factor_vw.data * (1 - update_rate) + tmp * update_rate

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)

        # compute attention
        print(f"q shape {q.shape} k shape {k.shape} v shape {v.shape}")
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        if self.recon_flag:
            q = torch.einsum("b h c, c -> b h c", q, self.scale_factor_qk)
            k = torch.einsum("b c h, c -> b c h", k, self.scale_factor_qk)
            # q = q * self.scale_factor
            # k = k / self.scale_factor
        if self.use_act_quant:
            q = self.act_quantizer_q(q)
            k = self.act_quantizer_k(k)
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.recon_flag:
            v = torch.einsum("b c h, h -> b c h", v, self.scale_factor_vw)
            w_ = torch.einsum("b m n, m -> b m n", w_, self.scale_factor_vw)
        if self.use_act_quant:
            v = self.act_quantizer_v(v)
            w_ = self.act_quantizer_w(w_)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.to_out(h_)

        out = x + h_
        return out


def get_specials(quant_act=False):
    specials = {
        BasicTransformerBlock: QuantBasicTransformerBlock,
        MemoryEfficientCrossAttention: QuantAttnBlock,
    }
    return specials


