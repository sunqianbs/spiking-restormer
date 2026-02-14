## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import numbers
import math

from einops import rearrange
from spikingjelly.activation_based import monitor, neuron, surrogate, layer



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
## 脉冲神经元（spikingjelly实现）

class SpikeLIF(nn.Module):
    """Wrapper over spikingjelly LIFNode to keep existing call sites unchanged."""
    def __init__(self, tau, vth, surrogate_scale=10.0, reset="soft"):
        super(SpikeLIF, self).__init__()
        v_reset = None if reset == "soft" else 0.0
        self.node = neuron.LIFNode(
            tau=float(tau),
            decay_input=False,
            v_threshold=float(vth),
            v_reset=v_reset,
            surrogate_function=surrogate.Sigmoid(alpha=float(surrogate_scale)),
            detach_reset=False,
            step_mode="m",
            backend="torch",
        )

    def reset_state(self):
        self.node.reset()

    def forward(self, x):
        # Accept both 4D and 5D inputs; tdBN/conv may use 4D.
        if x.dim() == 4:
            x_seq = x.unsqueeze(0)
            y = self.node(x_seq)
            return y.squeeze(0)
        return self.node(x)

#把这个类加到你的代码里，替换掉原来那个坏掉的类
class TDBN_Layer(nn.Module):
    def __init__(self, dim, v_th, alpha=1.0):
        super().__init__()
        # 直接调用 PyTorch 原生 BN，绕过那个有毒的库
        self.bn = nn.BatchNorm2d(dim, affine=True)
        
        # 手动实现 tdBN 的核心功能：初始化缩放
        # 这就是 tdBN 论文里唯一的数学改动
        torch.nn.init.constant_(self.bn.weight, alpha * v_th)
        torch.nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        # 无论你传 4D 还是 5D，都能处理
        if x.dim() == 5:
            # 如果是 (T, B, C, H, W)，先压扁再算
            T, B, C, H, W = x.shape
            x_flat = x.reshape(T * B, C, H, W)
            out = self.bn(x_flat)
            return out.reshape(T, B, C, H, W)
        else:
            # 如果是 4D (TB, C, H, W)，直接算
            return self.bn(x)



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
        spike_qk=None,
        spike_tau=None,
        spike_vth=None,
        spike_surrogate_scale=None,
        spike_reset=None,
        spike_log=None,
        spike_log_interval=None
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.scale_factor = nn.Parameter(torch.tensor(1.0))
        # 这是一个针对每个 Channel 的可学习缩放参数
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * 1e-2)
        # 1. 在 __init__ 中增加一个可学习的偏置
        self.v_th = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5) # 初始化为0.5

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.spike_qk = spike_qk
        self.spike_log = spike_log
        self.spike_log_interval = spike_log_interval
        self._spike_log_count = 0
        self._tb_shape = None
        self._stats = {
            'attn_calls': 0,
            'block_calls': 0,
            'n_min_sum': 0.0,
            'n_mean_sum': 0.0,
            'n_p1_sum': 0.0,
            'attn_row_max_mean_sum': 0.0,
            'attn_out_abs_mean_sum': 0.0,
            'xin_abs_mean_sum': 0.0,
            'ratio_sum': 0.0,
        }
        if self.spike_qk:
            self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
            v_th = float(spike_vth) if spike_vth is not None else 1.0
            self.q_pre_tdbn = TDBN_Layer(dim, v_th=v_th)
            self.k_pre_tdbn = TDBN_Layer(dim, v_th=v_th)
            self.v_pre_tdbn = TDBN_Layer(dim, v_th=v_th)
            self.q_tdbn = TDBN_Layer(dim, v_th=v_th)
            self.k_tdbn = TDBN_Layer(dim, v_th=v_th)
            self.v_tdbn = TDBN_Layer(dim, v_th=v_th)
            self.k_bn = nn.BatchNorm2d(dim, affine=False)
            self.spike_q = SpikeLIF(tau=spike_tau, vth=spike_vth, surrogate_scale=spike_surrogate_scale, reset=spike_reset)
            self.spike_k = SpikeLIF(tau=spike_tau, vth=spike_vth, surrogate_scale=spike_surrogate_scale, reset=spike_reset)
            self.spike_v = SpikeLIF(tau=spike_tau, vth=spike_vth, surrogate_scale=spike_surrogate_scale, reset=spike_reset)

    def set_tb_shape(self, t, b):
        if t is None or b is None:
            self._tb_shape = None
        else:
            self._tb_shape = (int(t), int(b))

    def add_block_stats(self, xin_abs_mean, ratio):
        if not self.spike_log:
            return
        self._stats['block_calls'] += 1
        self._stats['xin_abs_mean_sum'] += float(xin_abs_mean)
        self._stats['ratio_sum'] += float(ratio)

    def pop_stats(self):
        attn_calls = self._stats['attn_calls']
        block_calls = self._stats['block_calls']
        out = {}
        if attn_calls > 0:
            out['n_min'] = self._stats['n_min_sum'] / attn_calls
            out['n_mean'] = self._stats['n_mean_sum'] / attn_calls
            out['n_p1'] = self._stats['n_p1_sum'] / attn_calls
            out['attn_row_max_mean'] = self._stats['attn_row_max_mean_sum'] / attn_calls
            out['attn_out_abs_mean'] = self._stats['attn_out_abs_mean_sum'] / attn_calls
        if block_calls > 0:
            out['xin_abs_mean'] = self._stats['xin_abs_mean_sum'] / block_calls
            out['attn_ratio'] = self._stats['ratio_sum'] / block_calls

        for k in self._stats:
            self._stats[k] = 0.0 if k.endswith('_sum') else 0
        return out

    def forward(self, x):
        tb, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if self.spike_qk:
            if self._tb_shape is None:
                t, b = 1, tb
            else:
                t, b = self._tb_shape
                assert t * b == tb, f"tb mismatch: got tb={tb}, but t*b={t*b}"

            def to_time_first_5d(feat):
                return rearrange(feat, '(t b) c h w -> t b c h w', t=t)

            def to_merged_4d(feat):
                return rearrange(feat, 't b c h w -> (t b) c h w')

            q = to_time_first_5d(q)
            q = self.q_pre_tdbn(q)
            q = self.spike_q(q)
            q = to_merged_4d(q)
            # q = self.q_proj(q)
            # q = self.q_tdbn(q)


            # k = to_time_first_5d(k)
            # k = self.k_pre_tdbn(k)
            k = self.k_bn(k)
            # k = self.spike_k(k)
            # k = to_merged_4d(k)
            # k = self.k_proj(k)
            # k = self.k_tdbn(k)


            v = to_time_first_5d(v)
            v = self.v_pre_tdbn(v)
            v = self.spike_v(v)
            v = to_merged_4d(v)
            # v = self.v_proj(v)
            # v = self.v_tdbn(v)
            # v = self.v_bn(v)



        
        q = rearrange(q, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head ch) h w -> b head ch (h w)', head=self.num_heads)

        # q = F.normalize(q, p=2, dim=-1)
        # k = F.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn - self.v_th
        # attn = F.relu(attn)
        attn = F.softmax(attn, dim=-1)
        # ====== 核心结构改造：峰值缩放 (Peak Scaling) ======
        # 找到每一行的最大值，将其作为缩放分母（加上 1e-6 防除零）
        # detach() 是为了防止这个寻找最大值的操作干扰主体梯度流
        # attn_max = attn.amax(dim=-1, keepdim=True).detach()
        # 将整个 Map 线性映射到 [0, 1] 区间，最大值严格等于 1.0
        # attn = attn / (attn_max + 1e-6)

        # attn = attn ** 2
        # denom = attn.sum(dim=-1, keepdim=True)
        # attn = attn / (denom + 1e-6)


        if self.spike_log:
            with torch.no_grad():
                denom_post = attn.sum(dim=-1)
                self._stats['attn_calls'] += 1
                self._stats['n_min_sum'] += denom_post.min().item()
                self._stats['n_mean_sum'] += denom_post.mean().item()
                self._stats['n_p1_sum'] += (denom_post > 1.0).float().mean().item()
                self._stats['attn_row_max_mean_sum'] += attn.detach().max(dim=-1).values.mean().item()

        # out = (attn @ v) * self.scale_factor
        out = (attn @ v)
        out = rearrange(out, 'b head ch (h w) -> b (head ch) h w', head=self.num_heads, h=h, w=w)
        # 放大
        out = self.layer_scale * out
        out = self.project_out(out)
        


        if self.spike_log:
            with torch.no_grad():
                self._stats['attn_out_abs_mean_sum'] += out.detach().abs().mean().item()

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
        spike_qk=None,
        spike_tau=None,
        spike_vth=None,
        spike_surrogate_scale=None,
        spike_reset=None,
        spike_log=None,
        spike_log_interval=None
    ):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.norm1 = nn.BatchNorm2d(dim, affine=True)
        self.attn = Attention(
            dim,
            num_heads,
            bias,
            spike_qk=spike_qk,
            spike_tau=spike_tau,
            spike_vth=spike_vth,
            spike_surrogate_scale=spike_surrogate_scale,
            spike_reset=spike_reset,
            spike_log=spike_log,
            spike_log_interval=spike_log_interval
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        xin = self.norm1(x)
        attn_out = self.attn(xin)
        if self.attn.spike_log:
            with torch.no_grad():
                xin_abs = xin.detach().abs().mean().item()
                attn_out_abs_mean = attn_out.detach().abs().mean().item()
                ratio = attn_out_abs_mean / (xin_abs + 1e-6)
            self.attn.add_block_stats(xin_abs, ratio)
        x = x + attn_out
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
class SpikingRestormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        spike_T = 4,                   ## 脉冲时间步数
        spike_qk = None,               ## 是否开启Q/K脉冲化
        spike_tau = None,              ## LIF时间常数
        spike_vth = None,              ## LIF阈值
        spike_surrogate_scale = None,  ## 代理梯度平滑系数
        spike_reset = None,            ## reset策略：soft/hard
        spike_log = None,              ## 是否打印spike率
        spike_log_interval = None      ## spike率打印间隔
    ):

        super(SpikingRestormer, self).__init__()
        missing = []
        for k, v in {
            "spike_qk": spike_qk,
            "spike_tau": spike_tau,
            "spike_vth": spike_vth,
            "spike_surrogate_scale": spike_surrogate_scale,
            "spike_reset": spike_reset,
            "spike_log": spike_log,
            "spike_log_interval": spike_log_interval,
        }.items():
            if v is None:
                missing.append(k)
        if missing:
            raise ValueError("SpikingRestormer requires spike params in YAML: " + ", ".join(missing))
        self.spike_T = spike_T
        self.spike_qk = spike_qk
        self.spike_log = spike_log

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_blocks[0])])
        
        # self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, spike_qk=spike_qk, spike_tau=spike_tau, spike_vth=spike_vth, spike_surrogate_scale=spike_surrogate_scale, spike_reset=spike_reset, spike_log=spike_log, spike_log_interval=spike_log_interval) for i in range(num_refinement_blocks)])
        self.refinement = nn.Identity()
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.apply(self._init_weights)
        self.spike_monitor = monitor.OutputMonitor(self, SpikeLIF) if self.spike_log else None

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.06, a=-0.04, b=0.04)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, WithBias_LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, BiasFree_LayerNorm):
            nn.init.constant_(m.weight, 1.0)

    def get_spike_stats_and_reset(self):
        stats = {}
        if self.spike_monitor is not None:
            if len(self.spike_monitor.records) > 0:
                rates = [x.detach().float().mean().item() for x in self.spike_monitor.records]
                stats['fr_mean'] = float(sum(rates) / len(rates))
                stats['fr_min'] = float(min(rates))
                stats['fr_max'] = float(max(rates))
            self.spike_monitor.clear_recorded_data()

        attn_stats = []
        for m in self.modules():
            if isinstance(m, Attention):
                s = m.pop_stats()
                if s:
                    attn_stats.append(s)
        if attn_stats:
            keys = attn_stats[0].keys()
            for k in keys:
                stats[k] = float(sum(d[k] for d in attn_stats) / len(attn_stats))
        return stats

    def reset_states(self):
        """清空所有SpikeLIF的膜电位状态"""
        for m in self.modules():
            if isinstance(m, SpikeLIF):
                m.reset_state()

    def _set_attention_tb_shape(self, t=None, b=None):
        for m in self.modules():
            if isinstance(m, Attention):
                m.set_tb_shape(t, b)

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
            out_dec_level1 += self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            # out_dec_level1 = self.output(out_dec_level1)
            feat = out_dec_level1 # (B, C, H, W) 还没过 output 的 feature
            res = self.output(feat)

            out = inp_img + res 


        return out

    def forward(self, inp_img):
        """时间维外层并行：T>1时先构造5D并行批次，再做均值聚合"""
        assert torch.isfinite(inp_img).all()
        self.reset_states()
        if self.spike_T > 1:
            # (B,C,H,W) -> (T,B,C,H,W) -> (T*B,C,H,W)
            inp_seq = inp_img.unsqueeze(0).repeat(self.spike_T, 1, 1, 1, 1)
            t, b, c, h, w = inp_seq.shape
            self._set_attention_tb_shape(t, b)
            inp_tb = inp_seq.reshape(t * b, c, h, w)

            out_tb = self._forward_impl(inp_tb)
            # (T*B,C,H,W) -> (T,B,C,H,W)
            _, out_c, out_h, out_w = out_tb.shape
            out_seq = out_tb.reshape(t, b, out_c, out_h, out_w)
            out = out_seq.mean(dim=0)
            self._set_attention_tb_shape(None, None)
            assert torch.isfinite(out).all()
            return out
        else:
            self._set_attention_tb_shape(None, None)
            out = self._forward_impl(inp_img)
            assert torch.isfinite(out).all()
            return out
