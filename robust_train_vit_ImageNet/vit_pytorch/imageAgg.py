from functools import partial
import torch
from torch import nn, einsum
from math import sqrt

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# helpers

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b



class Attention(nn.Module):    ## basic attention code
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads   ## b batchsize; n:patchnumber
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale   ## simiar to matrix multiply

        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value = True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, float('-inf'))
        #     del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class FeedForward_conv(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Hardswish(),
            DepthWiseConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        # x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.net(x)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        return x



class Attention_conv(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))
        ## img conv
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, Attention_conv(dim, heads = heads, dropout = dropout)),
        #         PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
        #     ]))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))

        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         FeedForward(dim, mlp_mult, dropout = dropout),
        #         FeedForward(dim, mlp_mult, dropout = dropout)
        #     ]))

    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ImageAgg(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in hierarchies]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, dim),
        # )

        ###### conv input

        # dim = 96
        # depth = 12
        # heads = 3
        in_channels = 3
        out_channels = 32
        # dropout = 0.
        # emb_dropout = 0.
        conv_kernel = 3
        stride = 1
        pool_kernel = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_kernel, stride)
        )
        feature_size = image_size // 2

        assert feature_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size // patch_size) ** 2
        patch_dim = out_channels * patch_size ** 2
        self.to_patch_embedding2 = nn.Sequential(
            self.conv,
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, layer_dims[0]),
        )

        ## conv input


        block_repeats = cast_tuple(block_repeats, num_hierarchies)

        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))
        #
        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

        ### MLP input  Part 3
        # self.mlp_head = nn.Sequential(
        #     LayerNorm(dim),
        #     Reduce('b c h w -> b c', 'mean'),
        #     nn.Linear(dim, mlp_mult),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(mlp_mult, num_classes)
        # )

    def forward(self, img):
        x1 = self.to_patch_embedding(img)
        # b, c, h, w = x.shape
        x = self.to_patch_embedding2(img)

        num_hierarchies = len(self.layers)



        # for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
        #     block_size = 2 ** level
        #     x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
        #     x = transformer(x)
        #     x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
        #     x = aggregate(x)

        h = w = int(sqrt(x.shape[-2]))

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            # x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)


            x = rearrange(x, 'b c (b1 h1) (b2 w1) -> (b b1 b2) c h1 w1', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            # x = rearrange(x, '(b b1 b2) a c -> b (a b1 b2) c  ', b1 = block_size, b2 = block_size)
            # h = w = int(sqrt(x.shape[-2]))
            # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = aggregate(x)
            # x = rearrange(x, 'b c h w -> b (h w) c')


        return self.mlp_head(x)
