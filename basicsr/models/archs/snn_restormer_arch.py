## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## 脉冲神经元与代理梯度（用于Q/K脉冲化）

class _SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    # ctx = forward 和 backward 之间的中转站，可以用来保存 forward 过程中需要在 backward 中使用的变量
    # x当前膜电位距离阈值还有多远，scale 代理梯度的平滑系数
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (x > 0).to(x.dtype)

    @staticmethod
    # grad_output后面网络”的梯度
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        sig = torch.sigmoid(x * scale)
        grad = grad_output * sig * (1 - sig) * scale
        return grad, None


class SpikeLIF(nn.Module):
    """简单LIF：u = decay*u + x；s = H(u - vth)，并用代理梯度反传"""
    def __init__(self, decay=0.25, vth=0.15, surrogate_scale=10.0, reset="soft"):
        super(SpikeLIF, self).__init__()
        self.decay = decay
        self.vth = vth
        self.surrogate_scale = surrogate_scale
        self.reset = reset
        self.u = None

    def reset_state(self):
        self.u = None

    def forward(self, x):
        if self.u is None or self.u.shape != x.shape or self.u.device != x.device:
            self.u = torch.zeros_like(x)
        self.u = self.u * self.decay + x
        s = _SurrogateHeaviside.apply(self.u - self.vth, self.surrogate_scale)
        if self.reset == "soft":
            self.u = self.u - s * self.vth
        else:
            self.u = self.u * (1 - s)
        return s



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        bias,
        spike_qk=False,
        spike_decay=0.25,
        spike_vth=0.15,
        spike_surrogate_scale=10.0,
        spike_reset="soft",
        spike_log=False,
        spike_log_interval=100,
        fr_ema_momentum=0.99
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.spike_qk = spike_qk
        self.spike_log = spike_log
        self.spike_log_interval = spike_log_interval
        self.fr_ema_momentum = fr_ema_momentum
        self._spike_log_count = 0
        if self.spike_qk:
            self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            self.q_bn = nn.BatchNorm2d(dim)
            self.k_bn = nn.BatchNorm2d(dim)
            self.spike_q = SpikeLIF(decay=spike_decay, vth=spike_vth, surrogate_scale=spike_surrogate_scale, reset=spike_reset)
            self.spike_k = SpikeLIF(decay=spike_decay, vth=spike_vth, surrogate_scale=spike_surrogate_scale, reset=spike_reset)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        if self.spike_qk:
            q = self.q_proj(q)
            q = self.q_bn(q)
            q = self.spike_q(q)
            k = self.k_proj(k)
            k = self.k_bn(k)
            k = self.spike_k(k)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        n = float(q.shape[3])  # N = H*W
        # scale1 on q @ k^T logits: scale1 = 1 / N
        scale1 = 1.0 / (n + 1e-6)
        attn = attn * scale1
        # 强制去均值：让 Attention 有正有负，从而具备“抑制”背景的能力
        attn = attn - attn.mean(dim=-1, keepdim=True)

        # denom before normalization (for stats)
        denom = attn.sum(dim=-1, keepdim=False)

        if self.spike_log and (self._spike_log_count % self.spike_log_interval == 0):
            with torch.no_grad():
                q_rate = q.detach().float().mean().item()
                scale1_min = scale1_mean = scale1_max = float(scale1)

                n_min = denom.min().item()
                n_mean = denom.mean().item()
                n_p1 = (denom > 1.0).float().mean().item()

            print(
                "[spike] q_rate={:.6f} n_min={:.6f} n_p1={:.6f} n_mean={:.6f} "
                "scale1_min/mean/max={:.6f}/{:.6f}/{:.6f} "
                .format(
                    q_rate, n_min, n_p1, n_mean,
                    scale1_min, scale1_mean, scale1_max
                )
            )
        self._spike_log_count += 1

        assert torch.isfinite(attn).all()

        out = (attn @ v)
        c_head = float(q.shape[2])
        # scale2 on attn @ v output: scale2 = 1 / sqrt(c_head + eps)
        scale2 = (c_head + 1e-6) ** -0.5
        out = out * scale2

        if self.spike_log and (self._spike_log_count % self.spike_log_interval == 0):
            with torch.no_grad():
                scale2_min = scale2_mean = scale2_max = float(scale2)

                attn_row_max_mean = attn.detach().max(dim=-1).values.mean().item()
            print(
                "[spike] scale2_min/mean/max={:.6f}/{:.6f}/{:.6f} attn_row_max_mean={:.6f}".format(
                    scale2_min, scale2_mean, scale2_max, attn_row_max_mean
                )
            )
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        assert torch.isfinite(out).all()
        if self.spike_log and (self._spike_log_count % self.spike_log_interval == 0):
            with torch.no_grad():
                attn_out_abs_mean = out.detach().abs().mean().item()
            print(f"[spike] attn_out_abs_mean={attn_out_abs_mean:.6f}")
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        spike_qk=False,
        spike_decay=0.25,
        spike_vth=0.15,
        spike_surrogate_scale=10.0,
        spike_reset="soft",
        spike_log=False,
        spike_log_interval=100
    ):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(
            dim,
            num_heads,
            bias,
            spike_qk=spike_qk,
            spike_decay=spike_decay,
            spike_vth=spike_vth,
            spike_surrogate_scale=spike_surrogate_scale,
            spike_reset=spike_reset,
            spike_log=spike_log,
            spike_log_interval=spike_log_interval
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- SNNRestormer -----------------------
class SNNRestormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        spike_T = 1,                   ## 脉冲时间步数
        spike_qk = False,              ## 是否开启Q/K脉冲化
        spike_decay = 0.25,            ## LIF衰减
        spike_vth = 0.15,              ## LIF阈值
        spike_surrogate_scale = 10.0,  ## 代理梯度平滑系数
        spike_reset = "soft",          ## reset策略：soft/hard
        spike_log = False,             ## 是否打印spike率
        spike_log_interval = 100       ## spike率打印间隔
    ):

        super(SNNRestormer, self).__init__()
        self.spike_T = spike_T
        self.spike_qk = spike_qk

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_decay=spike_decay, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def reset_states(self):
        """清空所有SpikeLIF的膜电位状态"""
        for m in self.modules():
            if hasattr(m, "reset_state"):
                m.reset_state()

    def _forward_impl(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

    def forward(self, inp_img):
        """时间维外层循环：T>1时重复前向并做均值聚合"""
        assert torch.isfinite(inp_img).all()
        if self.spike_T > 1:
            self.reset_states()
            x_seq = inp_img.unsqueeze(0).repeat(self.spike_T, 1, 1, 1, 1)
            out_seq = []
            for t in range(self.spike_T):
                out_t = self._forward_impl(x_seq[t])
                out_seq.append(out_t)
            out = torch.stack(out_seq, dim=0).mean(dim=0)
            assert torch.isfinite(out).all()
            return out
        else:
            if self.spike_qk:
                self.reset_states()
            out = self._forward_impl(inp_img)
            assert torch.isfinite(out).all()
            return out
