# https://github.com/ShoufaChen/CycleMLP/blob/main/cycle_mlp.py
import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import to_2tuple, to_3tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

if __name__ != "__main__":
    from tabular_data import get_module_from_config
else:
    from src.tabular_data import get_module_from_config, get_tabular_config


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, F, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3)
            x = super().forward(x)
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(0, 2, 1)
            x = super().forward(x)
            x = x.permute(0, 2, 1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CycleFC(nn.Module):
    """
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class GTM(nn.Module):
    def __init__(self, dim, group_size=2, mixing_type="short_range"):
        super().__init__()
        self.S = group_size
        self.ty = mixing_type
        self.C = dim
        if self.ty == 'shift_token':
            self.linear = nn.Linear(self.S * self.C, self.C)
        else:
            self.linear = nn.Linear(self.S * self.C, self.S * self.C)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1, 4)
        x = self.grouped_time_mixing(x)
        return x.permute(0, 3, 1, 2, 4)

    def grouped_time_mixing(self, x):
        B, H, W, T, C = x.shape
        if self.ty == 'short_range':
            x = self.linear(x.reshape(B, H, W, -1, self.S * C))
            x = x.reshape(B, H, W, T, C)
        elif self.ty == 'long_range':
            x = x.reshape(B, H, W, self.S, -1, C).transpose(3, 4)
            x = self.linear(x.reshape(B, H, W, -1, self.S * C))
            x = x.reshape(B, H, W, -1, self.S, C).transpose(3, 4)
            x = x.reshape(B, H, W, T, C)
        elif self.ty == 'shift_window':
            x = torch.roll(x, self.S // 2)
            x = self.linear(x.reshape(B, H, W, -1, self.S * C))
            x = torch.roll(x.reshape(B, H, W, T, C), -self.S // 2)
        elif self.ty == 'shift_token':
            x = [torch.roll(x, i) for i in range(self.S)]
            x = self.linear(torch.cat(x, dim=4))
        return x


class TokenMixing3DMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, proj_drop=0., mixing_type="short_range"):
        super().__init__()
        self.sfc_h = CycleFC(dim, dim, (1, 7), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (7, 1), 1, 0)
        self.gtm = GTM(dim, mixing_type=mixing_type)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x2d = x.reshape(B * T, H, W, C)
        h = self.sfc_h(x2d.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, T, H, W, C)
        w = self.sfc_w(x2d.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, T, H, W, C)
        t = self.gtm(x)
        a = (h + w + t).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + t * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block3D(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, drop_path=0., act_layer=nn.GELU, norm_layer=GroupNorm,
                 skip_lam=1.0, mlp_fn=TokenMixing3DMLP, mixing_type="short_range"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, mixing_type=mixing_type)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 3D Image to Patch Embedding with overlapping
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        stride = to_3tuple(stride)
        padding = to_3tuple(padding)
        self.patch_size = patch_size
        # remove image_size in model init to support dynamic image size

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding,
                              groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """ Downsample transition stage
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv3d(in_embed_dim, out_embed_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)  # B, C, T, H, W
        x = x.permute(0, 2, 3, 4, 1)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False,
                 drop_path_rate=0., skip_lam=1.0, mlp_fn=TokenMixing3DMLP, mixing_type="short_range", **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(Block3D(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path=block_dpr, skip_lam=skip_lam,
                              mlp_fn=mlp_fn, mixing_type=mixing_type))
    blocks = nn.Sequential(*blocks)

    return blocks


class MLP3D(nn.Module):
    """ MLP3D Network """

    def __init__(self, layers, patch_size=7, in_chans=3, num_classes=1000, embed_dims=None, transitions=None,
                 mlp_ratios=None, skip_lam=1.0, qkv_bias=False, drop_path_rate=0., norm_layer=GroupNorm,
                 mlp_fn=TokenMixing3DMLP, mixing_type="short_range", tabular_config=None, tab_concat_dim=None, ):

        super().__init__()
        self.use_tabular = tabular_config is not None
        self.use_concat = tab_concat_dim is not None
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedOverlapping(patch_size=patch_size, stride=4, padding=2, in_chans=in_chans,
                                                 embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn,
                                 mixing_type=mixing_type)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))

        self.network = nn.ModuleList(network)

        if self.use_tabular:
            if len(tabular_config) == 1:
                self.tab1 = None
                self.tab2 = None
                self.tab3 = None
                self.tab4 = None
                self.tab5 = get_module_from_config(tabular_config[0])
            else:
                self.tab1 = get_module_from_config(tabular_config[0])
                self.tab2 = get_module_from_config(tabular_config[1])
                self.tab3 = get_module_from_config(tabular_config[2])
                self.tab4 = get_module_from_config(tabular_config[3])
                self.tab5 = get_module_from_config(tabular_config[4])
        else:
            self.tab1 = None
            self.tab2 = None
            self.tab3 = None
            self.tab4 = None
            self.tab5 = None

        # Classifier head
        self.norm = norm_layer(embed_dims[-1])

        if self.use_concat:
            self.head = nn.Linear(in_features=embed_dims[-1] + tab_concat_dim,
                                  out_features=num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(in_features=embed_dims[-1],
                                  out_features=num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,T,H,W-> B,T,H,W,C
        x = x.permute(0, 2, 3, 4, 1)
        return x

    def forward_tokens(self, x, tab=None):
        for i, block in enumerate(self.network):
            x = block(x)
            if i == 0 and self.tab2 is not None:
                x = x.permute(0, 4, 1, 2, 3)
                x = self.tab2(x, tab)
                x = x.permute(0, 2, 3, 4, 1)
            if i == 2 and self.tab3 is not None:
                x = x.permute(0, 4, 1, 2, 3)
                x = self.tab3(x, tab)
                x = x.permute(0, 2, 3, 4, 1)
            if i == 4 and self.tab4 is not None:
                x = x.permute(0, 4, 1, 2, 3)
                x = self.tab4(x, tab)
                x = x.permute(0, 2, 3, 4, 1)
            if i == 6 and self.tab5 is not None:
                x = x.permute(0, 4, 1, 2, 3)
                x = self.tab5(x, tab)
                x = x.permute(0, 2, 3, 4, 1)

        B, T, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x, tab=None):
        x = self.forward_embeddings(x)

        if self.use_tabular and self.tab1 is not None:
            x = x.permute(0, 4, 1, 2, 3)
            x = self.tab1(x, tab)
            x = x.permute(0, 2, 3, 4, 1)
        # B, T, H, W, C -> B, N, C
        x = self.forward_tokens(x, tab)

        x = self.norm(x).mean(1)
        if self.use_concat:
            y = tab.flatten(1)
            x = torch.cat((x, y), dim=1)
        cls_out = self.head(x).squeeze(1)
        return cls_out


def MLP3D_T(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = MLP3D(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                  mlp_ratios=mlp_ratios, mlp_fn=TokenMixing3DMLP, **kwargs)
    return model


def MLP3D_M(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = MLP3D(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                  mlp_ratios=mlp_ratios, mlp_fn=TokenMixing3DMLP, **kwargs)
    return model


if __name__ == '__main__':
    x = torch.randn(2, 1, 16, 192, 192)
    model = MLP3D_T(in_chans=1, num_classes=1, mixing_type='shift_token')
    print(model.forward(x).shape)
