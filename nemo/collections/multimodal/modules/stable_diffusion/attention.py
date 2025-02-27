# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from apex.contrib.group_norm import GroupNorm
from einops import rearrange
from torch import einsum, nn
from torch._dynamo import disable

from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.util import checkpoint


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    cur_device = torch.cuda.current_device()
    dprops = torch.cuda.get_device_properties(cur_device)

    is_sm75 = dprops.major == 7 and dprops.minor == 5
    is_sm8x = dprops.major == 8 and dprops.minor >= 0
    is_sm90 = dprops.major == 9 and dprops.minor >= 0

    return is_sm8x or is_sm75 or is_sm90


try:
    import torch.nn as nn
    from flash_attn.modules.mha import FlashCrossAttention, FlashSelfAttention

    flash_attn_installed = check_cuda()
    print("FlashAttention Installed")

    # Disable TorchDynamo on FlashAttention
    FlashSelfAttention.forward = disable(FlashSelfAttention.forward)
    FlashCrossAttention.forward = disable(FlashCrossAttention.forward)
except ImportError:
    flash_attn_installed = False


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    if isinstance(d, (torch.Tensor, float, int)):
        return d
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels, num_groups=32, act=""):
    return GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, act=act)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


# b n (h d) -> (b h) n d
def rearrange_heads_outer(t: torch.Tensor, h: int) -> torch.Tensor:
    b, n, ch = t.shape
    return t.view(b, n, h, -1).transpose(1, 2).reshape(b * h, n, -1)


# (b h) n d -> b n (h d)
def rearrange_heads_inner(t: torch.Tensor, h: int) -> torch.Tensor:
    b = t.shape[0] // h
    n = t.shape[1]
    return t.view(b, h, n, -1).transpose(1, 2).reshape(b, n, -1)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, use_flash_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        # make attention part be aware of self-attention/cross-attention
        self.context_dim = context_dim
        self.query_dim = query_dim
        self.dim_head = dim_head

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.use_flash_attention = use_flash_attention

        if dim_head <= 160 and (dim_head % 8) == 0 and flash_attn_installed:
            if context_dim == query_dim:
                self.flash_attn = FlashSelfAttention(softmax_scale=self.scale)
            else:
                self.flash_attn = FlashCrossAttention(softmax_scale=self.scale)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        out = self._attention(q, k, v, mask)

        return self.to_out(out)

    def _attention(self, q, k, v, mask=None):
        h = self.heads

        if (
            not flash_attn_installed
            or not self.use_flash_attention
            or q.dtype == torch.float32
            or (self.dim_head > 160 or (self.dim_head % 8) != 0)
            or mask is not None
        ):
            # original implementation
            # b n (h d) -> (b h) n d
            q = rearrange_heads_outer(q, h)
            k = rearrange_heads_outer(k, h)
            v = rearrange_heads_outer(v, h)

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                # standard stable diffusion does not run into here
                mask = mask.view(mask.shape[0], -1)
                b, j = mask.shape
                mask = mask.unsqueeze(1).expand(b, h, j).reshape(b * h, 1, j)  # b j -> (b h) () j
                sim.masked_fill_(~mask, self.max_neg[sim.dtype])

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)

            # (b h) n d -> b n (h d)
            out = rearrange_heads_inner(out, h)
        elif self.context_dim == self.query_dim:
            # self-attention
            qkv = torch.stack([q, k, v], dim=2)
            b, s, t, hd = qkv.shape
            d = hd // h
            qkv = qkv.view(b, s, t, h, d)

            out = self.flash_attn(qkv)
            out = out.view(b, s, hd)
        else:
            # cross-attention
            kv = torch.stack([k, v], dim=2)

            s_q = q.shape[1]
            b, s_kv, t, hd = kv.shape
            d = hd // h

            q = q.view(b, s_q, h, d)
            kv = kv.view(b, s_kv, t, h, d)

            out = self.flash_attn(q, kv)
            out = out.view(b, s_q, hd)

        return out


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=False,
        use_flash_attention=False,
        disable_self_attn=False,
    ):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=False,
        use_flash_attention=False,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    use_checkpoint=use_checkpoint,
                    use_flash_attention=use_flash_attention,
                    disable_self_attn=disable_self_attn,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.view(b, c, -1).transpose(1, 2)  # b c h w -> b (h w) c
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = x.transpose(1, 2).view(b, c, h, w)  # b (h w) c -> b c h w
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
