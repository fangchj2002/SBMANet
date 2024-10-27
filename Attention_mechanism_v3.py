import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HPA import HPA


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, in1, in2):
        b, c, h, w = in1.shape

        in1 = rearrange(in1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        #               head=self.num_heads)
        in2 = rearrange(in2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        in1 = torch.nn.functional.normalize(in1, dim=-1)
        in2 = torch.nn.functional.normalize(in2, dim=-1)

        attn = (in1 @ in2.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ in2)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features*2, dim, kernel_size=1, bias=bias)

        self.hadamad = Grouped_multi_axis_Hadamard_Product_Attention(hidden_features*2,hidden_features*2)

    def forward(self, x):
        x = self.project_in(x)
        x = self.hadamad(x)
        x = self.project_out(x)
        return x

class HHMF(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(HHMF, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )
        self.act = nn.LeakyReLU(inplace=False)

    def forward(self, in1, in2):
        b, c, d, h, w = in1.shape
        in1 = in1.reshape(b, c, d, h * w)
        in2 = in2.reshape(b, c, d, h * w)

        in1 = self.norm1(in1)
        in2 = self.norm1(in2)

        atten = self.attn(in1, in2)
        out = in1 + in2 + atten
        out = out + self.ffn(self.norm2(out))
        out = out.reshape(b, c, d, h, w)

        return out

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, in1, in2):
        b, c, h, w = in1.shape
        in1 = rearrange(in1, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        in2 = rearrange(in2, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        in1 = torch.nn.functional.normalize(in1, dim=-1)
        in2 = torch.nn.functional.normalize(in2, dim=-1)
        attn1 = (in1 @ in2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ in2)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out1 = self.proj(out1)

        return out1

class Mlp(nn.Module):
    def __init__(self,
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Crossmodal_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False,):
        super(Crossmodal_Attention, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        self.act = nn.LeakyReLU(inplace=False)
    def forward(self, in1, in2):
        b, c, d, h, w = in1.shape
        in1 = self.norm1(in1.reshape(b, c, d, h*w))
        in2 = self.norm1(in2.reshape(b, c, d, h*w))

        atten = self.attn(in1, in2)
        out = in1 + in2 + atten

        out = out + self.mlp(self.norm2(out))

        out = out.reshape(b, c, d, h, w)

        return self.act(out)




