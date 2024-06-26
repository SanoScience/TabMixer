import torch
from einops import rearrange
from torch import nn


class TabularModule(nn.Module):
    def __init__(self, tab_dim=6,
                 channel_dim=2,
                 frame_dim=None,
                 hw_size=None,
                 module=None):
        """

        Args:
            tab_dim: Number of tabular features
            channel_dim: Number of channels
            frame_dim: Number of frames
            hw_size: Spatial dimensions (height and width)
            module: Module name (optional)
        """
        super(TabularModule, self).__init__()
        self.channel_dim = channel_dim
        self.tab_dim = tab_dim
        self.frame_dim = frame_dim
        self.hw_size = hw_size


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
        """
        TabMixer module for integrating tabular data in vision models via mixing operations.

        Args:
            norm: Normalization layer
            use_tabular_data: Do not use tabular data for mixing operations
            spatial_first:  Apply spatial mixing as first mixing operation
            use_spatial: Apply spatial mixing
            use_temporal: Apply temporal mixing
            use_channel: Apply channel mixing
            **kwargs:
        """
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


if __name__ == '__main__':
    x = torch.randn(2, 8, 16, 6, 6)
    y = torch.randn(2, 6)
    model = TabMixer(channel_dim=8, frame_dim=16, hw_size=(6, 6), tab_dim=6)
    print(model.forward(x, y).shape)
