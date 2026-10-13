"""Fused row wise softmax in Triton, benchmarked against PyTorch on a GPU.

Why fuse? An unfused softmax launches separate kernels (max, subtract, exp, sum,
divide), and each one round trips its intermediate through GPU global memory:
roughly 5 reads and 3 writes of the whole tensor. This kernel loads each row
once, does max / exp / sum / divide in registers, and writes once (1 read,
1 write). Softmax is memory bound, so bytes moved is what sets the runtime.

Usage:
  CPU correctness check (no GPU needed):  TRITON_INTERPRET=1 python triton_softmax.py
  GPU correctness + timing (e.g. Colab):  python triton_softmax.py
"""
import json
import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # Padding lanes load -inf so they contribute exp(-inf) = 0 to the row sum.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)                # accumulate in fp32 even for fp16 inputs
    x = x - tl.max(x, axis=0)           # subtract row max for numerical stability
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _num_warps(block: int) -> int:
    # Wider rows need more threads per row, otherwise each thread holds too many
    # values in registers and spills to slow local memory.
    if block >= 4096:
        return 16
    if block >= 2048:
        return 8
    return 4


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_contiguous()
    rows, cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](
        out, x, x.stride(0), out.stride(0), cols, BLOCK=block, num_warps=_num_warps(block)
    )
    return out


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Unfused reference: each op is its own kernel and round trips through GPU memory."""
    z = x - x.max(dim=1, keepdim=True).values
    num = torch.exp(z)
    return num / num.sum(dim=1, keepdim=True)


TOL = {torch.float32: 1e-5, torch.float16: 2e-3}


def check(device: str) -> None:
    torch.manual_seed(0)
    dtypes = [torch.float32] if device == "cpu" else [torch.float32, torch.float16]
    for dtype in dtypes:
        worst = 0.0
        # includes non power of two widths, so the masking path is exercised
        for rows, cols in [(4, 16), (7, 100), (32, 1000), (5, 1024), (3, 3000)]:
            x = torch.randn(rows, cols, device=device, dtype=dtype)
            ref = torch.softmax(x.float(), dim=1)
            err = (fused_softmax(x).float() - ref).abs().max().item()
            worst = max(worst, err)
            assert err < TOL[dtype], f"mismatch for {(rows, cols)} {dtype}: {err}"
        # numerical stability: large values must not overflow to inf/nan
        big_val = 1e4 if dtype == torch.float32 else 6e4
        big = torch.full((2, 64), big_val, device=device, dtype=dtype)
        assert torch.isfinite(fused_softmax(big)).all()
        print(f"[ok] {str(dtype):14} 5 shapes + large value test passed on {device}; max abs error {worst:.2e}")


# Published peak DRAM bandwidth (GB/s) for GPUs Colab hands out, to report % of peak.
PEAK_GBPS = {"T4": 320, "L4": 300, "A100-SXM4-40GB": 1555, "A100-SXM4-80GB": 2039, "V100": 900}


def _peak_for(name: str):
    for key, val in PEAK_GBPS.items():
        if key in name:
            return val
    return None


def _time_ms(fn) -> float:
    from triton.testing import do_bench
    try:
        return float(do_bench(fn, return_mode="median"))
    except TypeError:  # older Triton
        return float(do_bench(fn, quantiles=[0.5]))


def bench(device: str = "cuda") -> None:
    try:  # one compiled graph per (dtype, shape), so raise the recompile cap
        torch._dynamo.config.cache_size_limit = 64
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 64
    except Exception:
        pass
    compiled = torch.compile(naive_softmax, dynamic=False)

    name = torch.cuda.get_device_name(0)
    peak = _peak_for(name)
    rows = 8192
    impls = {
        "naive": naive_softmax,
        "torch_compile": compiled,
        "torch_softmax": lambda t: torch.softmax(t, dim=1),
        "triton": fused_softmax,
    }
    print(f"\nGPU: {name} | {rows} rows | spec peak bandwidth: {peak or 'unknown'} GB/s")
    print("GB/s is effective bandwidth: the ideal 1 read + 1 write of the tensor divided by runtime.\n")
    header = (f"{'dtype':>6} {'cols':>6} {'naive':>8} {'compile':>8} {'torch':>8} {'triton':>8}"
              f" {'GB/s':>6} {'%peak':>6} {'vs naive':>9} {'vs torch':>9}")
    results = []
    for dtype in (torch.float32, torch.float16):
        print(header)
        for cols in (256, 1024, 2048, 4096, 8192):
            x = torch.randn(rows, cols, device=device, dtype=dtype)
            ref = torch.softmax(x.float(), dim=1)
            ms = {}
            for key, fn in impls.items():
                y = fn(x)  # warm up (and compile, for torch.compile) before timing
                torch.cuda.synchronize()
                err = (y.float() - ref).abs().max().item()
                assert err < TOL[dtype], f"{key} wrong for {cols} cols {dtype}: {err}"
                ms[key] = _time_ms(lambda fn=fn: fn(x))
            ideal_bytes = 2 * x.numel() * x.element_size()
            gbps = {k: ideal_bytes / (v * 1e-3) / 1e9 for k, v in ms.items()}
            pct = 100 * gbps["triton"] / peak if peak else None
            row = dict(
                dtype=str(dtype).replace("torch.", ""), rows=rows, cols=cols,
                ms=ms, gb_per_s=gbps, triton_pct_of_peak=pct,
                speedup_vs_naive=ms["naive"] / ms["triton"],
                speedup_vs_torch_softmax=ms["torch_softmax"] / ms["triton"],
                speedup_vs_torch_compile=ms["torch_compile"] / ms["triton"],
            )
            results.append(row)
            print(f"{row['dtype']:>6} {cols:>6} {ms['naive']:>8.4f} {ms['torch_compile']:>8.4f}"
                  f" {ms['torch_softmax']:>8.4f} {ms['triton']:>8.4f} {gbps['triton']:>6.0f}"
                  f" {(f'{pct:.0f}%' if pct else 'n/a'):>6} {row['speedup_vs_naive']:>8.2f}x"
                  f" {row['speedup_vs_torch_softmax']:>8.2f}x")
        print()

    meta = dict(gpu=name, peak_gb_per_s=peak, torch=torch.__version__, triton=triton.__version__,
                timing="triton.testing.do_bench, median ms", results=results)
    with open("gpu_results.json", "w") as f:
        json.dump(meta, f, indent=2)
    write_markdown(meta)
    print("Saved gpu_results.json and results.md")


def write_markdown(meta: dict) -> None:
    lines = [
        f"**GPU:** {meta['gpu']} · {meta['results'][0]['rows']} rows · "
        f"torch {meta['torch']} · triton {meta['triton']} · median of `do_bench`",
        "",
        "| dtype | cols | naive (ms) | torch.compile (ms) | torch.softmax (ms) | Triton fused (ms) "
        "| Triton GB/s | % of peak | vs naive | vs torch.softmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in meta["results"]:
        ms = r["ms"]
        pct = f"{r['triton_pct_of_peak']:.0f}%" if r["triton_pct_of_peak"] else "n/a"
        lines.append(
            f"| {r['dtype']} | {r['cols']} | {ms['naive']:.3f} | {ms['torch_compile']:.3f} | "
            f"{ms['torch_softmax']:.3f} | {ms['triton']:.3f} | {r['gb_per_s']['triton']:.0f} | {pct} | "
            f"{r['speedup_vs_naive']:.2f}x | {r['speedup_vs_torch_softmax']:.2f}x |"
        )
    with open("results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    if torch.cuda.is_available() and not interp:
        check("cuda")
        bench("cuda")
    elif interp:
        check("cpu")
        print("Interpreter mode: correctness only. Run on a GPU for timing.")
    else:
        sys.exit("No GPU found. Run with TRITON_INTERPRET=1 for a CPU correctness check, or use a GPU (e.g. Colab).")
