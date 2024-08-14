import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


#
# 这两个类分别定义了没有偏置和有偏置的 Layer Normalization。Layer Normalization 用于归一化输入.
#
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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# 该类通过选择使用有偏置或没有偏置的 Layer Normalization，对输入进行标准化.
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


# 这个类定义模型中用于每个 Transformer 块的前馈神经网络（FeedForward）。该前馈网络包含一个卷积层
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x



class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, qk_norm=1):
        super(Cross_Attention, self).__init__()
        self.norm = qk_norm
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                         bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        
        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                         bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, feat):
        # 输入
        feat_A, feat_B = feat.chunk(2, dim=1)
        b, c, h, w = feat_A.shape
        qkv1 = self.qkv_dwconv1(self.qkv1(feat_A))
        qkv2 = self.qkv_dwconv2(self.qkv2(feat_B))

        q1, k1, v1 = qkv1.chunk(3, dim=1)
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        q_norm1 = torch.norm(q1, p=2, dim=-1, keepdim=True) / self.norm + 1e-6
        q1 = torch.div(q1, q_norm1)
        k_norm1 = torch.norm(k1, p=2, dim=-2, keepdim=True) / self.norm + 1e-6
        k1 = torch.div(k1, k_norm1)
        attn1 = k1 @ v1
        q_norm2 = torch.norm(q2, p=2, dim=-1, keepdim=True) / self.norm + 1e-6
        q2 = torch.div(q2, q_norm2)
        k_norm2 = torch.norm(k2, p=2, dim=-2, keepdim=True) / self.norm + 1e-6
        k2 = torch.div(k2, k_norm2)
        attn2 = k2 @ v2
        out_numerator1 = torch.sum(v2, dim=-2).unsqueeze(2) + (q1 @ attn2)
        out_denominator1 = torch.full((h * w, c // self.num_heads), h * w).to(q1.device) \
                               + q1 @ torch.sum(k2, dim=-1).unsqueeze(3).repeat(1, 1, 1, c // self.num_heads) + 1e-6

        out1 = torch.div(out_numerator1, out_denominator1) * self.temperature1
        # out1 = out1 * refine_weight1
        out1 = rearrange(out1, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        out1 = self.project_out1(out1)

        out_numerator2 = torch.sum(v1, dim=-2).unsqueeze(2) + (q2 @ attn1)
        out_denominator2 = torch.full((h * w, c // self.num_heads), h * w).to(q2.device) \
                               + q2 @ torch.sum(k1, dim=-1).unsqueeze(3).repeat(1, 1, 1, c // self.num_heads) + 1e-6

        out2 = torch.div(out_numerator2, out_denominator2) * self.temperature2
        # out2 = out2 * refine_weight2
        out2 = rearrange(out2, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)

        out2 = self.project_out2(out2)
        out = torch.cat((out1, out2), dim=1)
        return out

# 这是 Diffusion Transformer 模块的主要构建块: Layer Normalization、自注意力机制和前馈神经网络ss DTB(nn.Module):
class DTB(nn.Module):
    def __init__(self, dim, num_heads, ffn_factor, bias, LayerNorm_type):
        super(DTB, self).__init__()

        self.norm1 = LayerNorm(dim*2, LayerNorm_type)
        self.attn = Cross_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim*2, LayerNorm_type)
        self.ffn = FeedForward(dim*2, ffn_factor, bias)


    def forward(self, feat):
        feat = feat + self.attn(self.norm1(feat))
        feat = feat + self.ffn(self.norm2(feat))
        return feat



# handle multiple input 处理多个input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


#####################################Diffusion Transformer DFT################################
class Fusion_head(nn.Module):
    def __init__(self,
                 out_channels=1,  # 输出的通道数.
                 dim=128,  # Transformer 模块的维度。
                 num_blocks=[2, 2, 2, 2],  # 每个级别的 Transformer 模块块的数量。
                 heads=[1, 2, 4, 8],  # 每个级别的 Transformer 模块中注意力头的数量
                 ffn_factor=2.66,  # 前馈神经网络隐藏层维度相对于输入维度的倍数
                 bias=False,  # 是否包含卷积层的偏置。
                 LayerNorm_type='WithBias',  # Layer Normalization 的类型，可以是 'WithBias' 或 'BiasFree'。
                 ):

        super(Fusion_head, self).__init__()

        # 解码器
        self.decoders = nn.ModuleList()

        self.aggregate= nn.ModuleList()
        # 解码器 ：upsample+多个NAFBlock + upsample+多个NAFBlock
        for i in range(3, -1, -1):
            self.decoders.append(MySequential(*[DTB(dim=dim, num_heads=heads[i], ffn_factor=ffn_factor,
                                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[i])]))
            self.aggregate.append(MySequential(*[SKFF(dim, height=2)]))
            dim = dim // 2
        self.last_fuse=SKFF(8, height=2)
            # 输出层：16->1
        self.ending = HeadTanh2d(24, out_channels)
        #self.ending = nn.Conv2d(24, out_channels, 1)

    def forward(self, feat,latent_result):  # inp_img：输入图像，t：时间步长
        latent_A,latent_B=latent_result.chunk(2, dim=1)
        # 解码器
        for idx, (decoder, fusion, feat_in) in enumerate(zip(self.decoders, self.aggregate,feat)):
            
            if idx!=0:
                if x.shape[2:] == feat_in.shape[2:]:
                    feat_in=feat_in+x
                else:
                    x = F.interpolate(x, size=feat_in.shape[2:], mode='bilinear', align_corners=False)
                    feat_in=feat_in+x
            feat_in = decoder(feat_in)
            feat_A,feat_B=feat_in.chunk(2, dim=1)
            feat_in=fusion([feat_A,feat_B])
            x = F.interpolate(feat_in, scale_factor=2, mode="bilinear", align_corners=True)
        latent=self.last_fuse([latent_A,latent_B])
        # 输出
        x = self.ending(torch.cat((feat_in,latent), dim=1))
        return x


if __name__ == '__main__':
    from DFT_only import DFT
    num=8
    test_sample1 = torch.randn(1, num, 64, 64).cuda(0)
    test_sample2 = torch.randn(1, num, 64, 64).cuda(0)
    model = DFT(inp_channels=num * 2,
                out_channels=num * 2,
                dim=32,
                num_blocks=[4, 4, 4, 4],
                heads=[1, 2, 4, 8],
                ffn_factor=4.0,
                bias=False,
                LayerNorm_type='WithBias',
                num_channel=[16, 32, 64, 128]).cuda(0)


    fusion = Fusion_head(
                out_channels=8,
                dim=128,
                num_blocks=[2, 2, 2, 2],
                heads=[1, 2, 4, 8],
                ffn_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias',
                ).cuda(0).eval()

    total1 = sum([param.nelement() for param in model.parameters()])
    total2 = sum([param.nelement() for param in fusion.parameters()])
    total=total1+total2
    print("Number of parameters: %.2fM" % (total / 1e6))

    # 定义采样器
    latent = torch.cat((test_sample1, test_sample2), dim=1)

    t = 950
    a,middle= model(latent, t, 'cuda')

    x=fusion(middle)
    print(x.shape)
'''
torch.Size([1, 16, 256, 256])
torch.Size([1, 32, 128, 128])
torch.Size([1, 64, 64, 64])
torch.Size([1, 128, 32, 32])
'''