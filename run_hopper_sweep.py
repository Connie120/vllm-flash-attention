import argparse
import csv
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


ROOT = Path("/home/kang222/flash-attention").resolve()
BENCHMARK_SCRIPT = ROOT / "benchmarks" / "benchmark_hopper_forward.py"

# Data type to use in the benchmark script (matches --dtype argument there).
# Change this to "float16" if you want FP16 instead of BF16.
DTYPE = "bfloat16"

# Attention geometry for all runs
# Matches benchmark_hopper_forward.py CLI: --dim, --headdim, --nheads-q, --nheads-kv
HEAD_DIM = 128
NHEADS_Q = 32
NHEADS_KV = 8
DIM = HEAD_DIM * NHEADS_Q

# Page size for paged KV cache
# NOTE: The benchmark script (benchmark_hopper_forward.py) always uses paged KV format
# for all cases (prefill, decode, and mixed batches). This parameter controls the
# page size used in the paged KV cache format.
# Typical values: 16, 128, etc.
PAGE_SIZE = 16

# Base output directory for all profiles
OUT_BASE = Path("/purdue/kang222/attn_perf/FlashAttention")

# Kernel name prefix to search for in nsys stats output
KERNEL_NAME_PREFIX = (
    "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMa"
)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run a shell command, raising on failure."""
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(cmd)}")


def profile_config(
    mode: str,
    prefill_seqlen: int,
    prefill_batch: int,
    decode_ctx_len: int,
    decode_batch: int,
    repeats: int,
    dtype: str,
    page_size: Optional[int] = None,
    contiguous_blocks: bool = False,
    flush_cache: bool = False,
) -> Tuple[Path, Path, Path]:
    """
    Run nsys profiling for a single configuration and run index.
    
    NOTE: The benchmark script always uses paged KV format. If page_size is None,
    the benchmark script will use its default (16). It's recommended to always
    pass a page_size value explicitly.

    Returns:
        (rep_path, sqlite_path, gpu_trace_path)
    """
    assert mode in {"prefill", "decode", "mix"}

    # Print all parameters for this run
    print("\n" + "=" * 80)
    print(f"Running benchmark: {mode.upper()} mode")
    print("=" * 80)
    print(f"  Mode: {mode}")
    print(f"  Prefill seqlen: {prefill_seqlen}")
    print(f"  Prefill batch: {prefill_batch}")
    print(f"  Decode ctx len: {decode_ctx_len}")
    print(f"  Decode batch: {decode_batch}")
    print(f"  Repeats: {repeats}")
    print(f"  Dtype: {dtype}")
    actual_page_size = page_size if page_size is not None else 16
    print(f"  Page size: {actual_page_size} (paged KV always enabled)")
    print(f"  Head dim: {HEAD_DIM}")
    print(f"  nheads_q: {NHEADS_Q}")
    print(f"  nheads_kv: {NHEADS_KV}")
    print(f"  Total dim: {DIM}")
    print(f"  Contiguous blocks: {contiguous_blocks}")
    print(f"  Flush cache: {flush_cache}")
    print(f"  Output directory: {OUT_BASE / mode / f'pgsz{actual_page_size}'}")
    print("=" * 80)

    # Create subdirectory based on page size
    # Use actual page_size (already computed above)
    page_dir = f"pgsz{actual_page_size}"
    out_dir = OUT_BASE / mode / page_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # File name encodes the active portion(s) of the workload:
    # - prefill: only prefill params
    # - decode: only decode params
    # - mix   : both prefill and decode params
    page_suffix = f"_pgsz{actual_page_size}"
    if mode == "prefill":
        # Example: fa3_prefill_1024_B1_pgsz16.nsys-rep
        base_name = f"fa3_prefill_{prefill_seqlen}_B{prefill_batch}{page_suffix}"
    elif mode == "decode":
        # Example: fa3_decode_1024_B1_pgsz16.nsys-rep
        base_name = f"fa3_decode_{decode_ctx_len}_B{decode_batch}{page_suffix}"
    else:  # mix
        # Example: fa3_mix_prefill_1024_B1_decode_1024_B1_pgsz16.nsys-rep
        base_name = (
            "fa3_mix_"
            f"prefill_{prefill_seqlen}_B{prefill_batch}_"
            f"decode_{decode_ctx_len}_B{decode_batch}{page_suffix}"
        )

    rep_path = out_dir / f"{base_name}.nsys-rep"
    sqlite_path = out_dir / f"{base_name}.sqlite"
    gpu_trace_path = out_dir / f"{base_name}.gpu-trace"

    # nsys profile command
    profile_cmd = [
        "bash",
        "-lc",
        " ".join(
            [
                "CUDA_VISIBLE_DEVICES=0",
                "nsys",
                "profile",
                "--force-overwrite",
                "true",
                "--show-output",
                "true",
                "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node",
                "--sample",
                "process-tree",
                "--cudabacktrace=kernel",
                "-o",
                str(rep_path),
                "python3",
                str(BENCHMARK_SCRIPT),
                f"--dim={DIM}",
                f"--seqlen-prefill={prefill_seqlen}",
                f"--batch-prefill={prefill_batch}",
                f"--seqlen-k-decode={decode_ctx_len}",
                f"--batch-decode={decode_batch}",
                f"--headdim={HEAD_DIM}",
                f"--nheads-q={NHEADS_Q}",
                f"--nheads-kv={NHEADS_KV}",
                f"--repeats={repeats}",
                f"--dtype={dtype}",
            ]
            + (["--page-size", str(page_size)] if page_size is not None else ["--page-size", "16"])  # Always pass page_size (default to 16 if None)
            + (["--contiguous-blocks"] if contiguous_blocks else [])
            + (["--flush-cache"] if flush_cache else [])
        ),
    ]

    run_cmd(profile_cmd, cwd=ROOT / "benchmarks")

    # nsys export to sqlite
    export_cmd = [
        "bash",
        "-lc",
        " ".join(
            [
                "nsys",
                "export",
                "-t",
                "sqlite",
                "--force-overwrite",
                "true",
                "-o",
                str(sqlite_path),
                str(rep_path),
            ]
        ),
    ]
    run_cmd(export_cmd)

    # nsys stats to GPU trace text
    stats_cmd = [
        "bash",
        "-lc",
        f"nsys stats -r cuda_gpu_trace {sqlite_path} > {gpu_trace_path}",
    ]
    run_cmd(stats_cmd)

    return rep_path, sqlite_path, gpu_trace_path


def parse_kernel_times_from_trace(trace_path: Path) -> List[float]:
    """
    From a .gpu-trace file, extract all **Duration (ns)** values associated with the
    FlashAttn kernel (identified by ``KERNEL_NAME_PREFIX``).

    We parse each matching line using gaps of 2+ spaces as column separators, which
    matches the format of ``cuda_gpu_trace`` output (see e.g. ``fa3_test.gpu-trace``).
    The second column is ``Duration (ns)``, which we return as a float.
    """
    times: List[float] = []
    with trace_path.open("r") as f:
        for line in f:
            if KERNEL_NAME_PREFIX in line:
                # Split on 2+ spaces to align with the table-like formatting
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) < 2:
                    continue
                try:
                    duration_ns = float(parts[1])
                except ValueError:
                    continue
                times.append(duration_ns)

    print(f"Found {len(times)} FlashAttn kernel entries in {trace_path}")
    return times


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Run FlashAttention v3 benchmark sweep on Hopper GPUs")
    parser.add_argument('--contiguous-blocks', action='store_true',
                        help='Use contiguous (sequential) block allocation instead of scattered blocks')
    parser.add_argument('--flush-cache', action='store_true',
                        help='Thrash L2 cache before each benchmark run to ensure cold cache')
    args = parser.parse_args()
    
    # Configuration grids
    prefill_seqlens = [1024, 2048, 4096, 8192]
    prefill_batches = [1, 2, 4, 8, 16]

    decode_ctx_lens = [1024, 2048, 4096, 8192]
    decode_batches = [1, 8, 64, 128, 256]

    # Number of benchmark iterations per run inside the benchmark script.
    # The user expects to see 5 kernel invocations per config in the GPU trace.
    benchmark_repeats = 5
    # We run nsys once per configuration; the averaging is over the 5 kernel
    # entries seen in that single trace.

    results: List[dict] = []

    # For "prefill" mode, we want only prefill traffic, so set decode batch to 0.
    # For "decode" mode, we want only decode traffic, so set prefill batch to 0.
    # We still use positive sequence lengths so that the benchmark script's
    # internal cu_seqlen calculations remain valid even when batch == 0.
    DECODE_DEFAULT_CTX = 1024
    DECODE_ZERO_BATCH = 0

    PREFILL_DEFAULT_SEQLEN = 1024
    PREFILL_ZERO_BATCH = 0

    # ---------- Prefill ----------
    for seqlen_prefill in prefill_seqlens:
        for batch_prefill in prefill_batches:
            mode = "prefill"
            all_times: List[float] = []
            _, _, trace_path = profile_config(
                mode=mode,
                prefill_seqlen=seqlen_prefill,
                prefill_batch=batch_prefill,
                decode_ctx_len=DECODE_DEFAULT_CTX,
                decode_batch=DECODE_ZERO_BATCH,
                repeats=benchmark_repeats,
                dtype=DTYPE,
                page_size=PAGE_SIZE,
                contiguous_blocks=args.contiguous_blocks,
                flush_cache=args.flush_cache,
            )
            times = parse_kernel_times_from_trace(trace_path)
            # PyTorch / nsys may include extra warmup kernels; keep only the last 5.
            tail = times[-5:] if len(times) >= 5 else times
            all_times.extend(tail)

            avg_time = average(all_times)
            results.append(
                {
                    "mode": mode,
                    "dtype": DTYPE,
                    "prefill_seqlen": seqlen_prefill,
                    "prefill_batch": batch_prefill,
                    "decode_ctx_len": DECODE_DEFAULT_CTX,
                    "decode_batch": DECODE_ZERO_BATCH,
                    "avg_duration_ns": avg_time,
                    "num_samples": len(all_times),
                }
            )

    # ---------- Decode ----------
    for ctx_len in decode_ctx_lens:
        for batch_decode in decode_batches:
            mode = "decode"
            all_times = []
            _, _, trace_path = profile_config(
                mode=mode,
                prefill_seqlen=PREFILL_DEFAULT_SEQLEN,
                prefill_batch=PREFILL_ZERO_BATCH,
                decode_ctx_len=ctx_len,
                decode_batch=batch_decode,
                repeats=benchmark_repeats,
                dtype=DTYPE,
                page_size=PAGE_SIZE,
                contiguous_blocks=args.contiguous_blocks,
                flush_cache=args.flush_cache,
            )
            times = parse_kernel_times_from_trace(trace_path)
            tail = times[-5:] if len(times) >= 5 else times
            all_times.extend(tail)

            avg_time = average(all_times)
            results.append(
                {
                    "mode": mode,
                    "dtype": DTYPE,
                    "prefill_seqlen": PREFILL_DEFAULT_SEQLEN,
                    "prefill_batch": PREFILL_ZERO_BATCH,
                    "decode_ctx_len": ctx_len,
                    "decode_batch": batch_decode,
                    "avg_duration_ns": avg_time,
                    "num_samples": len(all_times),
                }
            )

    # ---------- Mix (all combinations of prefill + decode) ----------
    for seqlen_prefill in prefill_seqlens:
        for batch_prefill in prefill_batches:
            for ctx_len in decode_ctx_lens:
                for batch_decode in decode_batches:
                    mode = "mix"
                    all_times = []
                    _, _, trace_path = profile_config(
                        mode=mode,
                        prefill_seqlen=seqlen_prefill,
                        prefill_batch=batch_prefill,
                        decode_ctx_len=ctx_len,
                        decode_batch=batch_decode,
                        repeats=benchmark_repeats,
                        dtype=DTYPE,
                        page_size=PAGE_SIZE,
                        contiguous_blocks=args.contiguous_blocks,
                        flush_cache=args.flush_cache,
                    )
                    times = parse_kernel_times_from_trace(trace_path)
                    tail = times[-5:] if len(times) >= 5 else times
                    all_times.extend(tail)

                    avg_time = average(all_times)
                    results.append(
                        {
                            "mode": mode,
                            "dtype": DTYPE,
                            "prefill_seqlen": seqlen_prefill,
                            "prefill_batch": batch_prefill,
                            "decode_ctx_len": ctx_len,
                            "decode_batch": batch_decode,
                            "avg_duration_ns": avg_time,
                            "num_samples": len(all_times),
                        }
                    )

    # Write CSV
    # PAGE_SIZE should always be set (defaults to 16 in benchmark script if not passed)
    actual_page_size = PAGE_SIZE if PAGE_SIZE is not None else 16
    page_suffix = f"_pgsz{actual_page_size}"
    csv_out = OUT_BASE / f"fa3_hopper_sweep_results_{args.contiguous_blocks}_{args.flush_cache}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "dtype",
        "headdim",
        "nheads_q",
        "nheads_kv",
        "prefill_seqlen",
        "prefill_batch",
        "decode_ctx_len",
        "decode_batch",
        "avg_duration_ns",
        "num_samples",
    ]

    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Attach geometry info to every row for clarity
            row = {
                **row,
                "headdim": HEAD_DIM,
                "nheads_q": NHEADS_Q,
                "nheads_kv": NHEADS_KV,
            }
            writer.writerow(row)

    print(f"Wrote results to {csv_out}")


if __name__ == "__main__":
    main()


