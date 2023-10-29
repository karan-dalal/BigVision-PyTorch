import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import models.configs as configs

def posemb_sincos_2d(h, w, width, device, temperature=10000., dtype=torch.float32):
    y_coords = torch.arange(h)
    x_coords = torch.arange(w)  
    y, x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    omega = torch.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature**omega)

    y = torch.einsum('m,d->md', y.flatten(), omega)
    x = torch.einsum('m,d->md', x.flatten(), omega)
    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)

    return pe[None, :, :].type(dtype).to(device)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),

                nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                      batch_first=True),

                nn.LayerNorm(dim),

                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):

        for norm1, attn, norm2, ff in self.layers:
            y = norm1(x)
            y = attn(y, y, y)[0]
            x = x + y

            y = norm2(x)
            y = ff(y)
            x = x + y

        return x

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size, num_classes):
        super(VisionTransformer, self).__init__()

        self.to_patch_embedding = nn.Conv2d(in_channels=3, out_channels=config.hidden_size,
                                            kernel_size=config.patch_size, stride=config.patch_size)

        self.transformer = Transformer(config.hidden_size, config.transformer.num_layers,
                                       config.transformer.num_heads, config.transformer.mlp_dim)

        self.norm = nn.LayerNorm(config.hidden_size)

        self.mlp_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, num_classes)
            )

        nn.init.xavier_uniform_(self.mlp_head[0].weight)
        nn.init.zeros_(self.mlp_head[0].bias)
        nn.init.constant_(self.mlp_head[2].weight, 0)
        nn.init.zeros_(self.mlp_head[2].bias)

    def forward(self, img):
        x = self.to_patch_embedding(img)    

        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n, h * w, c)     

        x = posemb_sincos_2d(h, w, c, device=x.device) + x

        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        return self.mlp_head(x)

CONFIGS = {
    'ViT-S_16': configs.get_s16_config(),
    'ViT-T_16': configs.get_t16_config(),
}
