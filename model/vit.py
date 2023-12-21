"""
Vision Transformer Architecture
by: Anthony Givans

TODO: Add more comments, use a config file
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """
    Patch Embedding
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_size: int = 768,
        img_size: int = 224,
        dropout: float = 1e-3,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        # number of patches = num_patches_h * num_patches_w, since height = width this reduces to num_patches ** 2
        num_patches = (img_size // patch_size) ** 2
        # after getting the pacthes from a c x img_size x img_size image, the dimensionality of each patch is patch_size ** 2 x patch_size x patch_size x c
        # this is num_patches x patch_size x patch_size x c, which means each patch is patch_size x patch_size x c
        patch_dim = in_channels * patch_size**2

        # LayerNorm speeds up convergence
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, embed_size), nn.LayerNorm(embed_size)
        )

        self.cls_token = nn.Parameter(torch.randn(embed_size))
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # x has shape batch_size x in_channels x img_size x img_size but we need -> batch_size x num_patches x patch_dim
        # b, 3, 224, 224 -> b, s, 14*16, 14*16 -> b, 3, 14, 16, 14, 16
        # b, 14*14, 16*16*3 -> b, num_patches, patch_dim
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        cls_token = repeat(self.cls_token, "d -> b 1 d", b=batch_size)
        x = torch.cat([cls_token, x], dim=1)  # prepend the cls token to the input
        # add position embedding and do dropout
        x += self.positions
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    MultiHead Attention
    """

    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int = 768, expansion: int = 4, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()

        self.att_norm = nn.LayerNorm(emb_size)
        self.ff_norm = nn.LayerNorm(emb_size)
        self.att = MultiHeadAttention(emb_size=emb_size, num_heads=num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # using += results in the following error:
        """
        RuntimeError: one of the variables needed for gradient computation has been modified by an
        inplace operation: [torch.cuda.FloatTensor [64, 65, 768]], which is output 0 of AddBackward0,
        is at version 24; expected version 23 instead. Hint: enable anomaly detection to find the
        operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
        """
        x = x + self.att(self.att_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        emb_size: int = 768,
        img_size: int = 224,
        depth: int = 12,
        n_classes: int = 1000,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embed_size=emb_size, img_size=img_size
        )
        self.layers = nn.ModuleList([TransformerBlock(emb_size=emb_size) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.fc(x[:, 0])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT().to(device)
    x = torch.torch.randn(1, 3, 224, 224).to(device=device)
    print(model(x).shape)
    summary(ViT(), (3, 224, 224), device="cuda")
