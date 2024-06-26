from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from monai.utils.module import look_up_option
from monai.utils.enums import SkipMode
import math
from einops import rearrange


class TabularModule(nn.Module):
    def __init__(self, tab_dim=6,
                 channel_dim=2,
                 frame_dim=None,
                 hw_size=None,
                 module=None):
        super(TabularModule, self).__init__()
        self.channel_dim = channel_dim
        self.tab_dim = tab_dim
        self.frame_dim = frame_dim
        self.hw_size = hw_size


# DAFT based on: https://github.com/ai-med/DAFT/blob/master/daft/networks/vol_blocks.py
class DAFT(TabularModule):
    def __init__(self,
                 bottleneck_dim=7,
                 **kwargs
                 ):
        super(DAFT, self).__init__(**kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        layers = [
            ("aux_base", nn.Linear(self.tab_dim + self.channel_dim, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, 2 * self.channel_dim, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, feature_map, x_aux):
        x_aux = x_aux.squeeze(dim=1)
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        v_scale, v_shift = torch.split(attention, self.channel_dim, dim=1)
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        out = (v_scale * feature_map) + v_shift

        return out


# FiLM based on: https://github.com/ai-med/DAFT/blob/master/daft/networks/vol_blocks.py
class FiLM(TabularModule):
    def __init__(self,
                 bottleneck_dim=7,
                 **kwargs
                 ):
        super(FiLM, self).__init__(**kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        layers = [
            ("aux_base", nn.Linear(self.tab_dim, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, 2 * self.channel_dim, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, feature_map, x_aux):
        attention = self.aux(x_aux)
        attention = attention.squeeze(dim=1)
        v_scale, v_shift = torch.split(attention, self.channel_dim, dim=1)
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        out = (v_scale * feature_map) + v_shift
        return out


# _______________ TabAttention ____________________________________________________________________
# _______________ CBAM based on: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py ___________________

class TabAttention(TabularModule):
    def __init__(self, channel_dim, frame_dim, hw_size, tab_dim=6, tabattention=True, cam_sam=True,
                 temporal_attention=True, **kwargs):
        """
        TabAttention module for integrating attention learning conditionally on tabular data within CNNs.

        @param tab_dim: Number of tabular data features
        @param tabattention: Turn on/off tabular embeddings (plain CBAM with TAM)
        @param cam_sam: Turn on/off Channel and Spatial Attention Modules
        @param temporal_attention: Turn on/off Temporal Attention Moudule
        """
        super(TabAttention, self).__init__(channel_dim=channel_dim, frame_dim=frame_dim, tab_dim=tab_dim,
                                           hw_size=hw_size)
        self.tabattention = tabattention
        self.temporal_attention = temporal_attention
        self.cam_sam = cam_sam
        if self.cam_sam:
            self.channel_gate = ChannelGate(channel_dim, tabattention=tabattention, tab_dim=tab_dim)
            self.spatial_gate = SpatialGate(tabattention=tabattention, tab_dim=tab_dim, input_size=hw_size)
        if temporal_attention:
            self.temporal_gate = TemporalGate(frame_dim, tabattention=tabattention, tab_dim=tab_dim)

    def forward(self, x, tab=None):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'n c f h w -> n f c h w')
        x_in = torch.reshape(x, (b * f, c, h, w))
        if self.tabattention:
            tab_rep = tab.repeat(f, 1, 1)
        else:
            tab_rep = None

        if self.cam_sam:
            x_out = self.channel_gate(x_in, tab_rep)
            x_out = self.spatial_gate(x_out, tab_rep)
        else:
            x_out = x_in

        x_out = torch.reshape(x_out, (b, f, c, h, w))

        if self.temporal_attention:
            x_out = self.temporal_gate(x_out, tab)

        x_out = rearrange(x_out, 'n f c h w -> n c f h w')

        return x_out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, tabattention=True, tab_dim=6, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.tabattention = tabattention
        self.tab_dim = tab_dim
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        if tabattention:
            self.pool_types = ['avg', 'max', 'tab']
            self.tab_embedding = nn.Sequential(
                nn.Linear(tab_dim, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

    def forward(self, x, tab=None):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            elif pool_type == 'tab':
                embedded = self.tab_embedding(tab)
                embedded = torch.reshape(embedded, (-1, self.gate_channels))
                pool = self.mlp(embedded)
                channel_att_raw = pool

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class TemporalMHSA(nn.Module):
    def __init__(self, input_dim=2, seq_len=16, heads=2):
        super(TemporalMHSA, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = 4
        self.head_dim = self.embedding_dim // heads
        self.heads = heads
        self.qkv = nn.Linear(self.input_dim, self.embedding_dim * 3)
        self.rel = nn.Parameter(torch.randn([1, 1, seq_len, 1]), requires_grad=True)
        self.o_proj = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, self.heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        d_k = q.size()[-1]
        k = k + self.rel.expand_as(k)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)  # [Batch, SeqLen, EmbeddingDim]
        x_out = self.o_proj(values)

        return x_out


class TemporalGate(nn.Module):
    def __init__(self, gate_frames, pool_types=['avg', 'max'], tabattention=True, tab_dim=6):
        super(TemporalGate, self).__init__()
        self.tabattention = tabattention
        self.tab_dim = tab_dim
        self.gate_frames = gate_frames
        self.pool_types = pool_types
        if tabattention:
            self.pool_types = ['avg', 'max', 'tab']
            self.tab_embedding = nn.Sequential(
                nn.Linear(tab_dim, gate_frames // 2),
                nn.ReLU(),
                nn.Linear(gate_frames // 2, gate_frames)
            )
        if tabattention:
            self.mhsa = TemporalMHSA(input_dim=3, seq_len=self.gate_frames)
        else:
            self.mhsa = TemporalMHSA(input_dim=2, seq_len=self.gate_frames)

    def forward(self, x, tab=None):
        avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4))).reshape(-1, self.gate_frames, 1)
        max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4))).reshape(-1, self.gate_frames, 1)

        if self.tabattention:
            embedded = self.tab_embedding(tab)
            tab_embedded = torch.reshape(embedded, (-1, self.gate_frames, 1))
            concatenated = torch.cat((avg_pool, max_pool, tab_embedded), dim=2)
        else:
            concatenated = torch.cat((avg_pool, max_pool), dim=2)

        scale = torch.sigmoid(self.mhsa(concatenated)).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, tabattention=True, tab_dim=6, input_size=(8, 8)):
        super(SpatialGate, self).__init__()
        self.tabattention = tabattention
        self.tab_dim = tab_dim
        self.input_size = input_size
        kernel_size = 7
        self.compress = ChannelPool()
        in_planes = 3 if tabattention else 2
        self.spatial = BasicConv(in_planes, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        if self.tabattention:
            self.tab_embedding = nn.Sequential(
                nn.Linear(tab_dim, input_size[0] * input_size[1] // 2),
                nn.ReLU(),
                nn.Linear(input_size[0] * input_size[1] // 2, input_size[0] * input_size[1])
            )

    def forward(self, x, tab=None):
        x_compress = self.compress(x)
        if self.tabattention:
            embedded = self.tab_embedding(tab)
            embedded = torch.reshape(embedded, (-1, 1, self.input_size[0], self.input_size[1]))
            x_compress = torch.cat((x_compress, embedded), dim=1)

        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


# _______________ TabAttention ____________________________________________________________________

class MLP(nn.Module):
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


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class TabMixer(TabularModule):
    def __init__(self,
                 norm=Affine,
                 use_tabular_data=True,
                 spatial_first=True,
                 use_spatial=True,
                 use_temporal=True,
                 use_channel=True,
                 **kwargs
                 ):
        super(TabMixer, self).__init__(**kwargs)

        self.use_tabular_data = use_tabular_data
        self.spatial_first = spatial_first
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_channel = use_channel

        if not self.use_tabular_data:
            self.tab_dim = 0
        else:
            self.tab_embedding = MLP(in_features=self.tab_dim, hidden_features=self.tab_dim // 2,
                                     out_features=self.tab_dim)

        if self.use_spatial:
            self.tab_mlp_s = MLP(in_features=(self.hw_size[0] * self.hw_size[1]) // 4 + self.tab_dim,
                                 hidden_features=(self.hw_size[0] * self.hw_size[1]) // 8,
                                 out_features=self.hw_size[0] * self.hw_size[1] // 4)
            self.norm_s = norm((self.hw_size[0] * self.hw_size[1]) // 4)

        if self.use_temporal:
            self.tab_mlp_t = MLP(in_features=self.frame_dim + self.tab_dim, hidden_features=self.frame_dim // 2,
                                 out_features=self.frame_dim)
            self.norm_t = norm(self.frame_dim)

        if self.use_channel:
            self.tab_mlp_c = MLP(in_features=self.channel_dim + self.tab_dim, hidden_features=self.channel_dim // 2,
                                 out_features=self.channel_dim)
            self.norm_c = norm(self.channel_dim)

        self.pool = nn.AvgPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x, tab=None):
        B, C_in, F_in, H_in, W_in = x.shape
        x = self.pool(x)
        _, C, F, H, W = x.shape

        x = x.reshape(B, C, F, (H * W))

        if self.use_tabular_data:
            tab = torch.unsqueeze(tab, 1).unsqueeze(1)
            tab_emb = self.tab_embedding(tab)

            tab_emb_spatial = tab_emb.repeat(1, C, F, 1)
            tab_emb_temporal = tab_emb.repeat(1, C, H * W, 1)
            tab_emb_channel = tab_emb.repeat(1, H * W, F, 1)

        if self.spatial_first:
            if self.use_spatial:
                x_s = self.norm_s(x)
                if self.use_tabular_data:
                    x_s = torch.concat([x_s, tab_emb_spatial], dim=3)
                x_s = self.tab_mlp_s(x_s)
                x = x + x_s
            x = rearrange(x, 'b c f hw -> b c hw f')

            if self.use_temporal:
                x_t = self.norm_t(x)
                if self.use_tabular_data:
                    x_t = torch.concat([x_t, tab_emb_temporal], dim=3)
                x_t = self.tab_mlp_t(x_t)
                x = x + x_t
            x = rearrange(x, 'b c hw f -> b hw f c')
        else:
            x = rearrange(x, 'b c f hw -> b c hw f')
            if self.use_temporal:
                x_t = self.norm_t(x)
                if self.use_tabular_data:
                    x_t = torch.concat([x_t, tab_emb_temporal], dim=3)
                x_t = self.tab_mlp_t(x_t)
                x = x + x_t
            x = rearrange(x, 'b c hw f -> b c f hw ')

            if self.use_spatial:
                x_s = self.norm_s(x)
                if self.use_tabular_data:
                    x_s = torch.concat([x_s, tab_emb_spatial], dim=3)
                x_s = self.tab_mlp_s(x_s)
                x = x + x_s
            x = rearrange(x, 'b c f hw -> b hw f c')

        if self.use_channel:
            x_c = self.norm_c(x)
            if self.use_tabular_data:
                x_c = torch.concat([x_c, tab_emb_channel], dim=3)
            x_c = self.tab_mlp_c(x_c)
            x = x + x_c
        x = rearrange(x, 'b hw f c -> b c f hw')

        x = x.reshape(B, C, F, H, W)
        x = nn.functional.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        return x


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: str = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value

    def forward(self, x: torch.Tensor, tab=None) -> torch.Tensor:
        y = self.submodule(x, tab)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class DoubleInputSequential(nn.Module):
    def __init__(self, *layers):
        super(DoubleInputSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y):
        for l in self.layers:
            if isinstance(l, (SkipConnection, TabularModule, DoubleInputSequential)):
                x = l(x, y)
            else:
                x = l(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 8, 16, 6, 6)
    y = torch.randn(2, 6)
    model = TabMixer(channel_dim=8, frame_dim=16, hw_size=(6, 6), tab_dim=6)
    print(model.forward(x, y).shape)
