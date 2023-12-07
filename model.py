
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))

        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False), # channels // 2, 4
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False), # channels * 2, 4
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
  
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # self.ca = MDTA(channel, 4)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=bias, dilation=1)

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)

        self.act1 = nn.ReLU()

        self.conv2 = conv(dim, dim, kernel_size, bias=True)

        self.calayer = CALayer(dim)
        # self.palayer = PALayer(dim)

        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        
        res = self.conv1(x)
        # res = res.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # res = self.norm(res)
        # res = res.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        res = self.act1(res)
        # res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        # res = self.palayer(res)
        res += x

        return res


class ResGroup(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(ResGroup, self).__init__()

        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))

        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
    

class DFPN(nn.Module):
    def __init__(self):
        super(DFPN, self).__init__()

        feature_num = 48
        heads_num = 1
        blocks = 1

        self.embed_conv = nn.Conv2d(3, feature_num, kernel_size=3, padding=1, bias=False)

        # Transformer         
        self.t_block1 = nn.Sequential(
            *[TransformerBlock(channels=feature_num, num_heads=heads_num, expansion_factor=2) for _ in range(blocks)],            
        )
        self.t_block2 = nn.Sequential(
            *[TransformerBlock(channels=feature_num*2, num_heads=heads_num*2, expansion_factor=2) for _ in range(blocks*2)],
        )
        self.t_block3 = nn.Sequential(
            *[TransformerBlock(channels=feature_num*4, num_heads=heads_num*4, expansion_factor=2) for _ in range(blocks*4)],
        )     

        self.down1 = DownSample(feature_num)
        self.down2 = DownSample(feature_num*2)
        self.down3 = DownSample(feature_num*4)
        
        self.t_block4 = nn.Sequential(
            *[TransformerBlock(channels=feature_num*8, num_heads=heads_num*8, expansion_factor=2) for _ in range(blocks*4)],
        )

        self.t_block5 = nn.Sequential(
            nn.Conv2d(feature_num*4, feature_num*4, 1, bias=False),
            *[TransformerBlock(channels=feature_num*4, num_heads=heads_num*4, expansion_factor=2) for _ in range(blocks*4)],
        )
        self.t_block6 = nn.Sequential(
            nn.Conv2d(feature_num*2, feature_num*2, 1, bias=False),
            *[TransformerBlock(channels=feature_num*2, num_heads=heads_num*2, expansion_factor=2) for _ in range(blocks*2)],
        )
        self.t_block7 = nn.Sequential(
            nn.Conv2d(feature_num, feature_num, 1, bias=False),
            *[TransformerBlock(channels=feature_num, num_heads=heads_num, expansion_factor=2) for _ in range(blocks)],
        )

        self.up3 = UpSample(feature_num*8)
        self.up2 = UpSample(feature_num*4)
        self.up1 = UpSample(feature_num*2)

        self.c_up3 = UpSample(feature_num*8)
        self.c_up2 = UpSample(feature_num*4)
        self.c_up1 = UpSample(feature_num*2)

        # Convolution
        self.c_block1 = nn.Sequential(            
            nn.Conv2d(feature_num*4, feature_num*4, 1, bias=True),
            ResGroup(default_conv, feature_num*4, 3, blocks=1),
        )
        self.c_block2 = nn.Sequential(        
            nn.Conv2d(feature_num*2, feature_num*2, 1, bias=True),    
            ResGroup(default_conv, feature_num*2, 3, blocks=1),
        )
        self.c_block3 = nn.Sequential(            
            nn.Conv2d(feature_num, feature_num, 1, bias=True),
            ResGroup(default_conv, feature_num, 3, blocks=1),
        )
       
        self.low_out = nn.Conv2d(feature_num, 3, kernel_size=3, padding=1, bias=False)
        self.high_out = nn.Conv2d(feature_num, 3, kernel_size=3, padding=1, bias=True)

        # self.cat_conv = nn.Conv2d(feature_num*2, feature_num, 3, 1, 1, bias=False)

        
    def forward(self, x):
        
        feat0 = self.embed_conv(x)
        
        # UNet: feature encoder
        t_feat1 = self.t_block1(feat0) 
        t_feat2 = self.t_block2(self.down1(t_feat1))
        t_feat3 = self.t_block3(self.down2(t_feat2))

        t_feat4 = self.t_block4(self.down3(t_feat3))
        
        # Low freqeuncy: transformer (MDTA)
        t_feat5 = self.t_block5(self.up3(t_feat4)) + t_feat3
        t_feat6 = self.t_block6(self.up2(t_feat5)) + t_feat2
        t_feat7 = self.t_block7(self.up1(t_feat6)) + t_feat1

        low = self.low_out(t_feat7) + x

        # High freqeuncy: convolution (ResGroup with layer norm)
        c_feat5 = self.c_block1(self.c_up3(t_feat4)) + t_feat3
        c_feat6 = self.c_block2(self.c_up2(c_feat5)) + t_feat2
        c_feat7 = self.c_block3(self.c_up1(c_feat6)) + t_feat1

        high = self.high_out(c_feat7)

        out = low + high

        # feat_list = [t_feat4, t_feat7, c_feat7]

        return out, low, high#, feat_list