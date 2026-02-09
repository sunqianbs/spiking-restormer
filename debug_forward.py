# debug_forward.py
import torch
import importlib.util
from pathlib import Path

# 直接从文件加载，避免触发 basicsr 包初始化导致 lmdb 依赖
_arch_path = Path(__file__).parent / "basicsr" / "models" / "archs" / "snn_restormer_arch.py"
_spec = importlib.util.spec_from_file_location("snn_restormer_arch", _arch_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SNNRestormer = _mod.SNNRestormer
Attention = _mod.Attention


def _stat(x):
    return (
        f"shape={tuple(x.shape)} min={x.min().item():.6f} "
        f"max={x.max().item():.6f} mean={x.mean().item():.6f} "
        f"finite={bool(torch.isfinite(x).all().item())}"
    )


_MDTA_PRINT_LIMIT = 2  # 只打印前N次MDTA
_mdta_print_count = 0

def debug_attention_forward(self, x):
    """Debug MDTA (Attention) tensors only."""
    global _mdta_print_count
    b, c, h, w = x.shape
    qkv = self.qkv_dwconv(self.qkv(x))
    q, k, v = qkv.chunk(3, dim=1)

    q = q.view(b, self.num_heads, -1, h * w)
    k = k.view(b, self.num_heads, -1, h * w)
    v = v.view(b, self.num_heads, -1, h * w)

    do_print = _mdta_print_count < _MDTA_PRINT_LIMIT
    if do_print:
        print("[MDTA] q", _stat(q))
        print("[MDTA] k", _stat(k))
        print("[MDTA] v", _stat(v))

    if getattr(self, "spike_qk", False):
        q = self.spike_q(q)
        k = self.spike_k(k)
        if do_print:
            print("[MDTA] q_spk", _stat(q))
            print("[MDTA] k_spk", _stat(k))

    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    attn = (q @ k.transpose(-2, -1)) * self.temperature
    attn = torch.relu(attn)
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
    if do_print:
        print("[MDTA] attn", _stat(attn))

    out = (attn @ v)
    out = out.view(b, -1, h, w)
    out = self.project_out(out)
    if do_print:
        print("[MDTA] out", _stat(out))
    _mdta_print_count += 1
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Monkey-patch MDTA for debug printing
    Attention.forward = debug_attention_forward

    net = SNNRestormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        spike_T=4,
        spike_qk=True,
    ).to(device)

    net.eval()

    x = torch.randn(1, 3, 64, 64).to(device)

    with torch.no_grad():
        y = net(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print(
        "Output stats:",
        "min =", y.min().item(),
        "max =", y.max().item(),
        "mean =", y.mean().item(),
    )

    assert y.shape == x.shape, "Output shape mismatch!"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf!"

    print("Forward test passed.")


if __name__ == "__main__":
    main()
