import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
from mmcv.cnn import build_norm_layer


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


'''
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
'''



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2 * (i // 2) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[:, np.newaxis, :]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)



class DTB(nn.Module):

    def __init__(self, dim, num_heads, ffn_factor, bias, LayerNorm_type):
        super(DTB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)
        self.MLP_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 4 * dim, bias=True)
        )

    def time_forward(self, time, mlp):
        time_emb = mlp(time)

        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def modulate(self,x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, t):

        if len(t.size()) == 1:
            t = nonlinearity(get_timestep_embedding(t, x.size(1)))[:, :]

        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(t,self.MLP_modulation)

        x = x + self.attn(self.modulate(self.norm1(x), shift_att, scale_att))
        x = x + self.ffn(self.modulate(self.norm2(x), shift_ffn, scale_ffn))

        return x, t


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# handle multiple input 
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


#####################################Diffusion Transformer DFT################################
class DFT(nn.Module):
    def __init__(self,
                 inp_channels=1,  
                 out_channels=1,  
                 dim=16,  
                 num_blocks=[4, 6, 6, 8],  
                 heads=[1, 2, 4, 8],  
                 ffn_factor=4.0,  
                 bias=False,  
                 LayerNorm_type='WithBias', 
                 num_channel=[16, 32, 64, 128],
                 ):

        super(DFT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(3):
            self.encoders.append(MySequential(*[
                DTB(dim=dim, num_heads=heads[i], ffn_factor=ffn_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _
                in range(num_blocks[i])]))
            self.downs.append(nn.Conv2d(dim, dim * 2, 2, 2))
            dim = dim * 2



        self.latent = MySequential(*[DTB(dim=dim, num_heads=heads[3], ffn_factor=ffn_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        for i in range(2, -1, -1):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 1, bias=False),
                    nn.PixelShuffle(2)  
                )
            )
            dim = dim // 2
            self.decoders.append(MySequential(*[DTB(dim=dim, num_heads=heads[i], ffn_factor=ffn_factor,
                                                    bias=bias, LayerNorm_type=LayerNorm_type) for _ in
                                                range(num_blocks[i])]))

    
        self.ending = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x, t, device):  
        
        if isinstance(t, int) or isinstance(t, float) or t.dim() == 0:
            t = torch.tensor([t]).to(device)
   
        x = self.patch_embed(x)  #
        B, C, H, W = x.shape
        middle_feat = []
        encs = []
    
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder(x, t)
            encs.append(x)
            x = down(x)

     
        x, _ = self.latent(x, t)
        middle_feat.append(x)

  
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            if x.shape[2:] == enc_skip.shape[2:]:
                x = x + enc_skip  
            else:
                x = F.interpolate(x, size=enc_skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + enc_skip  

            x, _ = decoder(x, t)
            middle_feat.append(x)

        x = self.ending(x)
        x = x[..., :H, :W]

        return x, middle_feat


if __name__ == '__main__':
    num = 8
    test_sample1 = torch.randn(1, num, 64, 64).cuda(1)
    test_sample2 = torch.randn(1, num, 64, 64).cuda(1)
    model = DFT(inp_channels=num * 2,
                out_channels=num * 2,
                dim=32,
                num_blocks=[4, 4, 4, 4],
                heads=[1, 2, 4, 8],
                ffn_factor=4.0,
                bias=False,
                LayerNorm_type='WithBias',
                num_channel=[16, 32, 64, 128]).cuda(1)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    latent = torch.cat((test_sample1, test_sample2), dim=1)
    t = 950
    a = model(latent, t, 'cuda:1')

'''
torch.Size([1, 16, 256, 256])
torch.Size([1, 32, 128, 128])
torch.Size([1, 64, 64, 64])
torch.Size([1, 128, 32, 32])
'''