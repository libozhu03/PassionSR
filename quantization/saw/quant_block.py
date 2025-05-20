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
    Base implementation of block structures for all networks.
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

    # def set_split(self, split):
    #     for m in self.modules():
    #         if isinstance(m, QuantLayer):
    #             m.set_split(split)

class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self,
        res: ResBlock,
        quantize_config: QuantizeModel_config
        ):
        super().__init__(quantize_config)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and self.skip_connection.split == 0:
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb, split=0):
        # print(f"x shape {x.shape} emb shape {emb.shape}")
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        # assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMul(BaseQuantBlock):
    def __init__(
        self,
        quantize_config: QuantizeModel_config
        ):
        super().__init__(quantize_config)
        # self.scale = None
        self.scale = nn.Parameter(torch.ones(1))
        self.use_act_quant = False
        self.quant_layer_type = get_layer_type(quantize_config)
        # self.scaling_factor = nn.Parameter(torch.ones(1))
        self.act_quantizer_q = self.quant_layer_type(self.quantize_config.act_config)
        self.act_quantizer_k = self.quant_layer_type(self.quantize_config.act_config)

    def forward(self, q, k):
        if self.use_act_quant:
            quant_q = self.act_quantizer_q(q * self.scale)
            quant_k = self.act_quantizer_k(k * self.scale)
            weight = torch.einsum(
                "bct,bcs->bts", quant_q, quant_k
            )
        else:
            weight = torch.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(
        self,
        quantize_config: QuantizeModel_config,
        ):
        super().__init__(quantize_config)
        self.use_act_quant = False
        self.quant_layer_type = get_layer_type(quantize_config)
        self.act_quantizer_v = self.quant_layer_type(self.quantize_config.act_config)
        self.act_quantizer_w = self.quant_layer_type(self.quantize_config.act_config)

    def forward(self, weight, v):
        if self.use_act_quant:
            a = torch.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
        else:
            a = torch.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttentionBlock,
        quantize_config: QuantizeModel_config,
        ):
        super().__init__(quantize_config)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv

        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)
    # print(q.shape)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # if self.recon_flag and not self.scale_init_flag:
    #     # if torch.all(self.scale_factor_qk.data == 1) \
    #     #     and torch.all(self.scale_factor_vw.data == 1):
    #     #     # self.init_scale_factor(q.shape[2], v.shape[1], q.device)
    #     #     self.init_scale_factor(q.shape[2], v.shape[1], q.device)
    #     # else:
    #     #     self.scale_init_flag = True
    #     self.init_scale_factor_qk(q, k, q.device)

    scale = 1.0 / q.shape[-1] ** 0.5
    if self.use_act_quant:
        if self.recon_flag:
            if not self.scale_init_flag:
                self.init_scale_factor_qk(q, k, q.device)
            elif self.running_stat:
                self.unpdate_scale_factor_qk(q, k)
            # print(f"q shape {q.shape} k shape {k.shape}")
            # print(f"scale factor qk shape {self.scale_factor_qk.shape}")
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
        # if context is None:
        #     assert(len(x) == 2)
        #     x, context = x
        
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


# the two classes below are for DDIM CIFAR
class QuantResnetBlock(BaseQuantBlock):
    def __init__(
        self, res: ResnetBlock,
        quantize_config: QuantizeModel_config
        ):
        super().__init__(quantize_config)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut


    def forward(self, x, temb=None, split=0):
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x, split=split)
        out = x + h
        return out


class QuantAttnBlock(BaseQuantBlock):
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
        # print("to_k weight size:")
        # print(self.to_k.weight.size())
        # self.device = attn.deivce
        # self.recon_flag = False
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
        # print(f"q shape {q.shape} k shape {k.shape}")
        self.scale_factor_qk.data = torch.ones(q.shape[2], requires_grad=True, \
                                    dtype=torch.float32, device=device)
        q_max = torch.amax(torch.abs(q), dim=(0,1)).clamp(min=1e-5).squeeze()
        k_max = torch.amax(torch.abs(k), dim=(0,1)).clamp(min=1e-5).squeeze()
        tmp = k_max ** self.s_alpha / q_max ** self.s_alpha
        # print(f"tmp shape {tmp.shape}")
        self.scale_factor_qk.data = tmp
        # print(f"scale factor qk shape {self.scale_factor_qk.shape}")
        # self.scale_init_flag = True

    def unpdate_scale_factor_qk(self, q, k, update_rate=0.1):
        # print(f"q shape {q.shape} k shape {k.shape}")
        q_max = torch.amax(torch.abs(q), dim=(0,1)).clamp(min=1e-5).squeeze()
        k_max = torch.amax(torch.abs(k), dim=(0,1)).clamp(min=1e-5).squeeze()
        tmp = k_max ** self.s_alpha / q_max ** self.s_alpha
        # print(f"tmp shape {tmp.shape}")
        # print(f"scale factor qk shape {self.scale_factor_qk.shape}")
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

    # def init_scale_factor(self, shape1, shape2, device):
    #     self.scale_factor_qk.data = torch.ones(shape1, requires_grad=True, \
    #                                 dtype=torch.float32, device=device)
    #     self.scale_factor_vw.data = torch.ones(shape2, requires_grad=True, \
    #                                 dtype=torch.float32, device=device)
    #     self.scale_init_flag = True

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)

        # if self.recon_flag and not self.scale_init_flag:
        #     if torch.all(self.self.scale_factor_qk.data == 1) \
        #         and torch.all(self.self.scale_factor_vw.data == 1):
        #         self.init_scale_factor(q.shape[1], q.shape[2]*q.shape[3], q.device)
        #     else:
        #         self.scale_init_flag = True

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


