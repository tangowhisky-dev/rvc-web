# CrossAttention, ConvNeXtBlock, ConvNeXtStack.
import gc
import gzip
import json
import math
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from functools import partial
from pathlib import Path
from pprint import pprint
from random import Random
from typing import BinaryIO, Literal, Optional, Union, Sequence, Iterable, Callable

import numpy as np
import pyworld
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    assert "soundfile" in torchaudio.list_audio_backends()
except AttributeError:
    import soundfile as _sf  # noqa: F401

if not hasattr(torch.amp, "GradScaler"):
    class GradScaler(torch.cuda.amp.GradScaler):
        def __init__(self, _, *args, **kwargs):
            super().__init__(*args, **kwargs)
    torch.amp.GradScaler = GradScaler

PARAPHERNALIA_VERSION = "2.0.0-rc.0"
from .conv import CausalConv1d, WSConv1d, WSLinear
from ..io import dump_params, dump_layer

class CrossAttention(nn.Module):
    def __init__(
        self,
        qk_channels: int,
        vo_channels: int,
        num_heads: int,
        in_q_channels: int,
        in_kv_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert qk_channels % num_heads == 0
        self.qk_channels = qk_channels
        self.vo_channels = vo_channels
        self.num_heads = num_heads
        self.in_q_channels = in_q_channels
        self.in_kv_channels = in_kv_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.head_qk_channels = qk_channels // num_heads
        self.head_vo_channels = vo_channels // num_heads
        self.q_projection = nn.Linear(in_q_channels, qk_channels)
        self.q_projection.weight.data.normal_(0.0, math.sqrt(1.0 / in_q_channels))
        self.q_projection.bias.data.zero_()
        self.kv_projection = nn.Linear(in_kv_channels, qk_channels + vo_channels)
        self.kv_projection.weight.data.normal_(0.0, math.sqrt(1.0 / in_kv_channels))
        self.kv_projection.bias.data.zero_()
        self.out_projection = nn.Linear(vo_channels, out_channels)
        self.out_projection.weight.data.normal_(0.0, math.sqrt(1.0 / vo_channels))
        self.out_projection.bias.data.zero_()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        # q: [batch_size, q_length, in_q_channels]
        # kv: [batch_size, kv_length, in_kv_channels]
        batch_size, q_length, _ = q.size()
        _, kv_length, _ = kv.size()
        # [batch_size, q_length, qk_channels]
        q = self.q_projection(q)
        # [batch_size, kv_length, qk_channels + vo_channels]
        kv = self.kv_projection(kv)
        # [batch_size, kv_length, qk_channels], [batch_size, kv_length, vo_channels]
        k, v = kv.split([self.qk_channels, self.vo_channels], dim=2)
        q = q.view(
            batch_size, q_length, self.num_heads, self.head_qk_channels
        ).transpose(1, 2)
        k = k.view(
            batch_size, kv_length, self.num_heads, self.head_qk_channels
        ).transpose(1, 2)
        v = v.view(
            batch_size, kv_length, self.num_heads, self.head_vo_channels
        ).transpose(1, 2)
        # [batch_size, num_heads, q_length, head_vo_channels]
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        # [batch_size, q_length, vo_channels]
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_length, self.vo_channels)
        )
        # [batch_size, q_length, out_channels]
        attn_out = self.out_projection(attn_out)
        return attn_out

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        q_projection_weight = self.q_projection.weight.data.clone()
        q_projection_bias = self.q_projection.bias.data.clone()
        q_projection_weight *= 1.0 / math.sqrt(math.sqrt(self.head_qk_channels))
        q_projection_bias *= 1.0 / math.sqrt(math.sqrt(self.head_qk_channels))
        dump_params(q_projection_weight, f)
        dump_params(q_projection_bias, f)
        dump_layer(self.out_projection, f)

    def dump_kv(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump_kv(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        kv_projection_weight = self.kv_projection.weight.data.clone()
        kv_projection_bias = self.kv_projection.bias.data.clone()
        k_projection_weight, v_projection_weight = kv_projection_weight.split(
            [self.qk_channels, self.vo_channels]
        )
        k_projection_bias, v_projection_bias = kv_projection_bias.split(
            [self.qk_channels, self.vo_channels]
        )
        k_projection_weight *= 1.0 / math.sqrt(math.sqrt(self.head_qk_channels))
        k_projection_bias *= 1.0 / math.sqrt(math.sqrt(self.head_qk_channels))
        # [qk_channels, in_kv_channels] -> [num_heads, head_qk_channels, in_kv_channels]
        k_projection_weight = k_projection_weight.view(
            self.num_heads, self.head_qk_channels, self.in_kv_channels
        )
        # [qk_channels] -> [num_heads, head_qk_channels]
        k_projection_bias = k_projection_bias.view(
            self.num_heads, self.head_qk_channels
        )
        # [vo_channels, in_kv_channels] -> [num_heads, head_vo_channels, in_kv_channels]
        v_projection_weight = v_projection_weight.view(
            self.num_heads, self.head_vo_channels, self.in_kv_channels
        )
        # [vo_channels] -> [num_heads, head_vo_channels]
        v_projection_bias = v_projection_bias.view(
            self.num_heads, self.head_vo_channels
        )
        for i in range(self.num_heads):
            # [head_qk_channels, in_kv_channels]
            dump_params(k_projection_weight[i], f)
            # [head_vo_channels, in_kv_channels]
            dump_params(v_projection_weight[i], f)
        for i in range(self.num_heads):
            # [head_qk_channels]
            dump_params(k_projection_bias[i], f)
            # [head_vo_channels]
            dump_params(v_projection_bias[i], f)


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        intermediate_channels: int,
        layer_scale_init_value: float,
        kernel_size: int = 7,
        use_weight_standardization: bool = False,
        enable_scaling: bool = False,
        pre_scale: float = 1.0,
        post_scale: float = 1.0,
        use_mha: bool = False,
        cross_attention: bool = False,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        attention_channels: Optional[int] = None,
        kv_channels: Optional[int] = None,
    ):
        super().__init__()
        self.use_weight_standardization = use_weight_standardization
        self.enable_scaling = enable_scaling
        self.use_mha = use_mha
        self.cross_attention = cross_attention
        if use_mha:
            self.attn_norm = nn.LayerNorm(channels)
            if cross_attention:
                self.mha = CrossAttention(
                    qk_channels=attention_channels,
                    vo_channels=attention_channels,
                    num_heads=num_heads,
                    in_q_channels=channels,
                    in_kv_channels=kv_channels,
                    out_channels=channels,
                    dropout=attention_dropout,
                )
            else:  # self-attention
                assert attention_channels is None
                assert kv_channels is None
                self.mha = nn.MultiheadAttention(
                    embed_dim=channels,
                    num_heads=num_heads,
                    dropout=attention_dropout,
                    batch_first=True,
                )
        self.dwconv = CausalConv1d(
            channels, channels, kernel_size=kernel_size, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, intermediate_channels)
        self.pwconv2 = nn.Linear(intermediate_channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale_init_value))
        self.dwconv.weight.data.normal_(0.0, math.sqrt(1.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.pwconv1.weight.data.normal_(0.0, math.sqrt(2.0 / channels))
        self.pwconv1.bias.data.zero_()
        self.pwconv2.weight.data.normal_(0.0, math.sqrt(1.0 / intermediate_channels))
        self.pwconv2.bias.data.zero_()
        if use_weight_standardization:
            self.norm = nn.Identity()
            self.dwconv = WSConv1d(channels, channels, kernel_size, groups=channels)
            self.pwconv1 = WSLinear(channels, intermediate_channels)
            self.pwconv2 = WSLinear(intermediate_channels, channels)
            del self.gamma
        if enable_scaling:
            self.register_buffer("pre_scale", torch.tensor(pre_scale))
            self.register_buffer("post_scale", torch.tensor(post_scale))
            self.post_scale_weight = nn.Parameter(torch.ones(()))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_mha:
            batch_size, channels, length = x.size()
            if self.cross_attention:
                assert kv is not None
            else:
                assert kv is None
                assert length % 4 == 0
            identity = x
            if self.cross_attention:
                # kv: [batch_size, kv_length, kv_channels]
                x = x.transpose(1, 2)
                x = self.attn_norm(x)
                x = self.mha(x, kv)
                x = x.transpose(1, 2)
            else:
                x = x.view(batch_size, channels, length // 4, 4)
                x = x.permute(0, 3, 2, 1)
                x = x.reshape(batch_size * 4, length // 4, channels)
                x = self.attn_norm(x)
                x, _ = self.mha(
                    x, x, x, attn_mask=attn_mask, is_causal=True, need_weights=False
                )
                x = x.view(batch_size, 4, length // 4, channels)
                x = x.permute(0, 3, 2, 1)
                x = x.reshape(batch_size, channels, length)
            x += identity

        identity = x
        if self.enable_scaling:
            x = x * self.pre_scale
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.pwconv2(x)
        if not self.use_weight_standardization:
            x *= self.gamma
        if self.enable_scaling:
            x *= self.post_scale * self.post_scale_weight
        x = x.transpose(1, 2)
        x += identity
        return x

    def merge_weights(self):
        if self.use_mha:
            if self.cross_attention:
                assert isinstance(self.mha, CrossAttention)
                self.mha.q_projection.bias.data += torch.mv(
                    self.mha.q_projection.weight.data, self.attn_norm.bias.data
                )
                self.mha.q_projection.weight.data *= self.attn_norm.weight.data[None, :]
                self.attn_norm.bias.data[:] = 0.0
                self.attn_norm.weight.data[:] = 1.0
            else:  # self-attention
                assert isinstance(self.mha, nn.MultiheadAttention)
                self.mha.in_proj_bias.data += torch.mv(
                    self.mha.in_proj_weight.data, self.attn_norm.bias.data
                )
                self.mha.in_proj_weight.data *= self.attn_norm.weight.data[None, :]
                self.attn_norm.bias.data[:] = 0.0
                self.attn_norm.weight.data[:] = 1.0
        if self.use_weight_standardization:
            self.dwconv.merge_weights()
            self.pwconv1.merge_weights()
            self.pwconv2.merge_weights()
        else:
            self.pwconv1.bias.data += torch.mv(
                self.pwconv1.weight.data, self.norm.bias.data
            )
            self.pwconv1.weight.data *= self.norm.weight.data[None, :]
            self.norm.bias.data[:] = 0.0
            self.norm.weight.data[:] = 1.0
            self.pwconv2.weight.data *= self.gamma.data[:, None]
            self.pwconv2.bias.data *= self.gamma.data
            self.gamma.data[:] = 1.0
        if self.enable_scaling:
            self.dwconv.weight.data *= self.pre_scale.data
            self.pre_scale.data.fill_(1.0)
            self.pwconv2.weight.data *= (
                self.post_scale.data * self.post_scale_weight.data
            )
            self.pwconv2.bias.data *= self.post_scale.data * self.post_scale_weight.data
            self.post_scale.data.fill_(1.0)
            self.post_scale_weight.data.fill_(1.0)

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        if self.use_mha:
            dump_layer(self.mha, f)
        dump_layer(self.dwconv, f)
        dump_layer(self.pwconv1, f)
        dump_layer(self.pwconv2, f)


class ConvNeXtStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        intermediate_channels: int,
        n_blocks: int,
        delay: int,
        embed_kernel_size: int,
        kernel_size: int,
        use_weight_standardization: bool = False,
        enable_scaling: bool = False,
        use_mha: bool = False,
        cross_attention: bool = False,
        kv_channels: Optional[int] = None,
    ):
        super().__init__()
        assert delay * 2 + 1 <= embed_kernel_size
        assert not (use_weight_standardization and use_mha)  # 未対応
        self.use_weight_standardization = use_weight_standardization
        self.use_mha = use_mha
        self.cross_attention = cross_attention
        self.embed = CausalConv1d(in_channels, channels, embed_kernel_size, delay=delay)
        self.norm = nn.LayerNorm(channels)
        self.convnext = nn.ModuleList()
        for i in range(n_blocks):
            pre_scale = 1.0 / math.sqrt(1.0 + i / n_blocks) if enable_scaling else 1.0
            post_scale = 1.0 / math.sqrt(n_blocks) if enable_scaling else 1.0
            block = ConvNeXtBlock(
                channels=channels,
                intermediate_channels=intermediate_channels,
                layer_scale_init_value=1.0 / n_blocks,
                kernel_size=kernel_size,
                use_weight_standardization=use_weight_standardization,
                enable_scaling=enable_scaling,
                pre_scale=pre_scale,
                post_scale=post_scale,
                use_mha=use_mha,
                cross_attention=cross_attention,
                num_heads=4,
                attention_dropout=0.1,
                attention_channels=kv_channels,
                kv_channels=kv_channels,
            )
            self.convnext.append(block)
        self.final_layer_norm = nn.LayerNorm(channels)
        self.embed.weight.data.normal_(
            0.0, math.sqrt(0.5 / (embed_kernel_size * in_channels))
        )
        self.embed.bias.data.zero_()
        if use_weight_standardization:
            self.embed = WSConv1d(in_channels, channels, embed_kernel_size, delay=delay)
            self.norm = nn.Identity()
            self.final_layer_norm = nn.Identity()

    def forward(
        self, x: torch.Tensor, kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if self.use_mha and not self.cross_attention:
            pad_length = -x.size(2) % 4
            if pad_length:
                x = F.pad(x, (0, pad_length))
            t40 = x.size(2) // 4
            attn_mask = torch.ones((t40, t40), dtype=torch.bool, device=x.device).triu(
                1
            )
        else:
            attn_mask = None
        for conv_block in self.convnext:
            x = conv_block(x, attn_mask=attn_mask, kv=kv)
        if self.use_mha and not self.cross_attention and pad_length:
            x = x[:, :, :-pad_length]
        x = self.final_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return x

    def merge_weights(self):
        if self.use_weight_standardization:
            self.embed.merge_weights()
        for conv_block in self.convnext:
            conv_block.merge_weights()

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.embed, f)
        if not self.use_weight_standardization:
            dump_layer(self.norm, f)
        dump_layer(self.convnext, f)
        if not self.use_weight_standardization:
            dump_layer(self.final_layer_norm, f)

    def dump_kv(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump_kv(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        assert self.use_mha and self.cross_attention
        for conv_block in self.convnext:
            if not conv_block.use_mha or not conv_block.cross_attention:
                continue
            assert isinstance(conv_block, ConvNeXtBlock)
            assert hasattr(conv_block, "mha")
            assert isinstance(conv_block.mha, CrossAttention)
            conv_block.mha.dump_kv(f)

