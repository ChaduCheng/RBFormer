import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from functools import partial


from einops.layers.torch import Rearrange, Reduce


from .Attention_block.Vit_var_ele import Attention, Attention_conv, FeedForward, FeedForward_conv, FeedForward_MLP

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn, parts):
        super().__init__()
        if parts['norm'] == 'LN':
            self.norm = nn.LayerNorm(dim)
        elif parts['norm'] == 'LN_channel':
            self.norm = LayerNorm(dim)
        elif parts['norm'] == 'BN':
            self.norm = nn.BatchNorm2d(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)




class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, parts, expansion_factor, num_patches):
        super().__init__()
        self.layers = nn.ModuleList([])

        if parts['Trans_block'] == 'ori_att':
            attention_block = Attention(dim, heads=heads, dropout=dropout)
        elif parts['Trans_block'] == 'conv_att':
            attention_block = Attention_conv(dim, heads=heads, dropout=dropout)
        elif parts['Trans_block'] == 'MLP_att':
            attention_block = FeedForward_MLP(num_patches, expansion_factor, dropout, partial(nn.Conv1d, kernel_size = 1))


        if parts['Trans_MLP'] == 'ori_MLP':
            ffn_block = FeedForward(dim, mlp_dim, dropout=dropout)
        elif parts['Trans_MLP'] == 'conv_MLP':
            ffn_block = FeedForward_conv(dim, mlp_dim, dropout=dropout)
        elif parts['Trans_MLP'] == 'no_MLP':
            ffn_block = nn.Identity()



        if parts['norm'] == 'none':
            attention_norm = attention_block
            ffn_norm = ffn_block
        else:
            attention_norm = PreNorm(dim, attention_block, parts)
            ffn_norm = PreNorm(dim, ffn_block, parts)


        if parts['skipconnect'] == 'none':
            attention_skipconnect = attention_norm
            ffn_skipconnect = ffn_norm
        elif parts['skipconnect'] == 'residual':
            attention_skipconnect = Residual(attention_norm)
            ffn_skipconnect = Residual(ffn_norm)

        attention = attention_skipconnect
        if parts['Trans_MLP'] == 'no_MLP':
            ffn = ffn_block
        else:
            ffn = ffn_skipconnect

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                attention,
                ffn,
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class Robust_ViTmodel(nn.Module):
    def __init__(
            self, *,
            image_size,
            patch_size,
            num_classes,
            dim=96,
            depth,
            heads,   ## swin
            mlp_dim,
            channels = 3,   ## vit, CoAtNet swin
            dropout = 0.,
            emb_dropout = 0.,
            dim_head = 64,   ## local_vit, nest
            parts,  ## vit structure dict, parts= {embedding: * ; transformer_block: * ; skip-connection: *; Layer-norm: *; }
            expansion_factor = 4   ### MixerMLP
    ):
        super().__init__()
        ### embedding input    Part 1
        self.type = type
        self.part = parts



        #### for patch embedding: Part 1.1 patch embedding part   different types of conv input

        if self.part['embedding'] == 'img_input':
            assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
            patch_dim = channels * patch_size ** 2
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )
        elif self.part['embedding'] == 'conv_input':
            dim = 96
            depth = 12
            heads = 3
            in_channels = 3
            out_channels = 96
            dropout = 0.
            emb_dropout = 0.
            conv_kernel = 7
            stride = 2
            pool_kernel = 3
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, conv_kernel, stride, 4),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(pool_kernel, stride)
            )
            feature_size = image_size // 4

            assert feature_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
            num_patches = (feature_size // patch_size) ** 2
            patch_dim = out_channels * patch_size ** 2
            self.to_patch_embedding = nn.Sequential(
                #self.conv,
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )
        elif self.part['embedding'] == 'attconv_input':
            assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
            patch_dim = channels * patch_size ** 2
            # self.to_patch_embedding = nn.Sequential(
            #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            #     nn.Linear(patch_dim, dim),
            # )
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
                nn.Conv2d(patch_dim, dim, 1),
            )




        if self.part['Trans_block'] == 'MLP_FFN':
            num_patches = (image_size // patch_size) ** 2
        else:
            ## position embedding
            num_patches = (image_size // patch_size) ** 2
            # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  ## add pos embedding
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  ## add cls token
            ## cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            ## others parameters:
            self.dropout = nn.Dropout(emb_dropout)


        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, self.part, expansion_factor, num_patches)

        ### MLP input  Part 3

        if self.type == 'MixerMLP':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                Reduce('b n c -> b c', 'mean'),
                nn.Linear(dim, num_classes)
            )
        else:
            self.to_cls_token = nn.Identity()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes)
            )


    def forward(self, img):


        ##### embedding part;

        # p = self.patch_size

        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # x = self.conv(img)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # cls_tokens = self.cls_token.expand(b, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)



        #### transformer block part:

        x = self.transformer(x)


        #### CLS token part output

        # x = self.to_cls_token(x[:, 0])


        ### MLP output

        # x = Reduce(x, 'b n c -> b c', 'mean'),

        x = x.mean(dim=1)


        out = self.mlp_head(x)

        return out