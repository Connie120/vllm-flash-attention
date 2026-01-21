"""
Benchmark script for forward-only attention on Hopper GPU (FA3) with variable-length sequences.
This script measures inference performance using flash_attn_varlen_func with fa_version=3.
Unlike benchmark_hopper_forward.py, this script supports variable sequence lengths within each batch,
with lengths specified in an input file.
"""
import argparse
import math
import sys
import os
from pathlib import Path

# Ensure unbuffered output so print statements are immediately visible
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import torch
import torch.utils.benchmark as benchmark

# Remove local flash-attention directory from path to avoid circular import
# We want to import from vLLM installation, not the local source
parent_dir = str(Path(__file__).parent.parent)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

# Try to import from vLLM installation (where the extension is built)
vllm_venv_paths = [
    os.path.expanduser("~/vllm-12-0-venv/lib/python3.12/site-packages/vllm_flash_attn-2.7.2.post1+cu128-py3.12-linux-x86_64.egg/vllm_flash_attn"),
    # os.path.expanduser("~/vllm-12-0-venv/lib/python3.12/site-packages/vllm/"),
    # os.path.expanduser("~/gpu-app-collection/bin/12.9/release/vllm/vllm-env/lib/python3.12/site-packages/vllm/"),
]

fa3_imported = False
for vllm_path in vllm_venv_paths:
    expanded_path = os.path.expanduser(vllm_path) if vllm_path.startswith("~") else vllm_path
    if os.path.exists(expanded_path):
        # Insert at the beginning to prioritize vLLM installation
        if expanded_path not in sys.path:
            sys.path.insert(0, expanded_path)
        try:
            # Remove any cached modules to force fresh import
            if 'vllm_flash_attn' in sys.modules:
                del sys.modules['vllm_flash_attn']
            if 'vllm_flash_attn.flash_attn_interface' in sys.modules:
                del sys.modules['vllm_flash_attn.flash_attn_interface']
            
            from vllm_flash_attn.flash_attn_interface import (
                flash_attn_varlen_func,
                get_scheduler_metadata,
                FA3_AVAILABLE,
                FA3_UNAVAILABLE_REASON,
                is_fa_version_supported,
                fa_version_unsupported_reason,
            )
            # Import reshape_and_cache_flash to simulate vLLM's cache write
            reshape_and_cache_flash_available = False
            reshape_and_cache_flash_func = None
            try:
                from vllm import _custom_ops as ops
                if hasattr(ops, 'reshape_and_cache_flash'):
                    reshape_and_cache_flash_func = ops.reshape_and_cache_flash
                    reshape_and_cache_flash_available = True
                    print(f"Found reshape_and_cache_flash in vllm._custom_ops")
            except ImportError:
                pass
            
            if not reshape_and_cache_flash_available:
                try:
                    from vllm.attention.utils.fa_utils import reshape_and_cache_flash
                    reshape_and_cache_flash_func = reshape_and_cache_flash
                    reshape_and_cache_flash_available = True
                    print(f"Found reshape_and_cache_flash in vllm.attention.utils.fa_utils")
                except ImportError:
                    print("Warning: reshape_and_cache_flash not available - cannot simulate vLLM cache write")
            fa3_imported = True
            print(f"Successfully imported FA3 from: {expanded_path}")
            break
        except ImportError as e:
            print(f"Failed to import from {expanded_path}: {e}")
            continue

# If not found in vLLM paths, raise error
if not fa3_imported:
    raise ImportError(
        "FA3 CUDA extension (_vllm_fa3_C) could not be imported.\n"
        "Tried paths:\n" + "\n".join(f"  {p}" for p in vllm_venv_paths) + "\n"
        "Please ensure:\n"
        "  1. The vLLM environment is activated\n"
        "  2. The extension is built in one of the above paths\n"
        "  3. The extension module _vllm_fa3_C.so exists"
    )

# Verify FA3 is available and supported
if not FA3_AVAILABLE:
    raise ImportError(
        f"FA3 CUDA extension is not available: {FA3_UNAVAILABLE_REASON}"
    )
if not is_fa_version_supported(3):
    reason = fa_version_unsupported_reason(3)
    raise RuntimeError(
        f"FA3 is not supported on this device. Reason: {reason}"
    )


def parse_lengths_file(file_path):
    """
    Parse sequence lengths from an input file.
    
    Expected format (one of):
    1. One length per line (simple format):
       1024
       512
       2048
       ...
    
    2. CSV format with header (optional):
       seqlen_q,seqlen_k
       1024,1024
       512,512
       2048,2048
       ...
    
    3. Space-separated format:
       1024 1024
       512 512
       2048 2048
       ...
    
    Returns:
        tuple: (seqlen_q_list, seqlen_k_list) where both are lists of integers
               If only one column is provided, seqlen_k defaults to seqlen_q
    """
    seqlen_q_list = []
    seqlen_k_list = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip empty lines and comments (lines starting with #)
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        raise ValueError(f"Input file {file_path} is empty or contains only comments/whitespace")
    
    # Check if first line is a header
    first_line = lines[0]
    has_header = False
    if ',' in first_line:
        # CSV format
        if first_line.lower().startswith('seqlen') or first_line.lower().startswith('length'):
            has_header = True
            lines = lines[1:]
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 1:
                seqlen_q = int(parts[0])
                seqlen_k = seqlen_q
            elif len(parts) >= 2:
                seqlen_q = int(parts[0])
                seqlen_k = int(parts[1])
            else:
                raise ValueError(f"Invalid line format: {line}")
            seqlen_q_list.append(seqlen_q)
            seqlen_k_list.append(seqlen_k)
    elif ' ' in first_line or '\t' in first_line:
        # Space/tab-separated format
        for line in lines:
            parts = line.split()
            if len(parts) == 1:
                seqlen_q = int(parts[0])
                seqlen_k = seqlen_q
            elif len(parts) >= 2:
                seqlen_q = int(parts[0])
                seqlen_k = int(parts[1])
            else:
                raise ValueError(f"Invalid line format: {line}")
            seqlen_q_list.append(seqlen_q)
            seqlen_k_list.append(seqlen_k)
    else:
        # Simple format: one length per line
        for line in lines:
            seqlen_q = int(line.strip())
            seqlen_k = seqlen_q
            seqlen_q_list.append(seqlen_q)
            seqlen_k_list.append(seqlen_k)
    
    if not seqlen_q_list:
        raise ValueError(f"No valid sequence lengths found in {file_path}")
    
    return seqlen_q_list, seqlen_k_list


def thrash_l2_cache(device='cuda'):
    """Thrash the GPU L2 cache by allocating and accessing large amounts of memory.
    
    This function allocates tensors large enough to fill the L2 cache multiple times
    and performs read/write operations to evict existing cache lines. This ensures
    that subsequent operations start with a cold cache.
    
    Args:
        device: The CUDA device to use (default: 'cuda')
    
    Note:
        - Hopper GPUs (H100) have ~50MB L2 cache
        - We allocate 2-3x the cache size to ensure complete eviction
        - Uses write operations which trigger write-back cache eviction
    """
    if not torch.cuda.is_available():
        return
    
    # L2 cache sizes: A100 ~40MB, H100 ~50MB, we use 150MB to be safe (3x)
    # Allocate in chunks to avoid single large allocation issues
    l2_cache_size_bytes = 150 * 1024 * 1024  # 150MB
    chunk_size_bytes = 10 * 1024 * 1024  # 10MB chunks
    num_chunks = (l2_cache_size_bytes + chunk_size_bytes - 1) // chunk_size_bytes
    
    # Allocate and write to multiple tensors to thrash the cache
    # Using different access patterns helps ensure cache eviction
    flush_tensors = []
    for i in range(num_chunks):
        # Allocate chunk as int8 to maximize memory footprint
        tensor = torch.empty(chunk_size_bytes, dtype=torch.int8, device=device)
        # Write to tensor to ensure it's accessed and evicts cache lines
        tensor.zero_()
        flush_tensors.append(tensor)
    
    # Perform additional operations to ensure cache eviction
    # Read and write to different parts of memory
    for tensor in flush_tensors:
        # Read operation
        _ = tensor.sum()
        # Write operation with different pattern
        tensor.fill_(1)
    
    # Synchronize to ensure all operations complete
    torch.cuda.synchronize()
    
    # Clean up (tensors will be garbage collected)
    del flush_tensors


def benchmark_forward(fn, *inputs, repeats=10, desc="", verbose=True, flush_cache=False, warmup=True, **kwinputs):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function.
    
    Args:
        fn: Function to benchmark
        *inputs: Positional arguments for fn
        repeats: Number of measurement iterations
        desc: Description string
        verbose: Whether to print timing information
        flush_cache: If True, thrash L2 cache before each run to ensure cold cache
        warmup: If True, perform warmup runs before timing (default: True)
        **kwinputs: Keyword arguments for fn
    """
    if verbose:
        print(desc, "- Forward pass")
    
    # For single runs, use simple timing to avoid PyTorch Timer's automatic warmup
    if repeats == 1:
        # Do one warmup run (if enabled)
        if warmup:
            if flush_cache and torch.cuda.is_available():
                thrash_l2_cache(device='cuda')
            fn(*inputs, **kwinputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Single timed run
        if flush_cache and torch.cuda.is_available():
            thrash_l2_cache(device='cuda')
        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_event.record()
        else:
            import time as time_module
            start_time = time_module.perf_counter()
        
        fn(*inputs, **kwinputs)
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        else:
            elapsed_time = time_module.perf_counter() - start_time
        
        # Create a mock Measurement object for compatibility
        class MockMeasurement:
            def __init__(self, mean_time):
                self.mean = mean_time
                self.median = mean_time
                self.min = mean_time
                self.max = mean_time
        
        m = MockMeasurement(elapsed_time)
        t = None  # No Timer object for single runs
        if verbose:
            warmup_info = " (with warmup)" if warmup else " (no warmup)"
            print(f"Single run{warmup_info}: {elapsed_time*1000:.3f} ms")
        return t, m
    
    # For multiple repeats, use PyTorch Timer (which does automatic warmup)
    num_warmup = 0
    for _ in range(num_warmup):
        if flush_cache and torch.cuda.is_available():
            thrash_l2_cache(device='cuda')
        fn(*inputs, **kwinputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Create a wrapper function that thrashes L2 cache before each call
    if flush_cache:
        def wrapped_fn(*args, **kwargs):
            if torch.cuda.is_available():
                thrash_l2_cache(device='cuda')
            return fn(*args, **kwargs)
        timer_fn = wrapped_fn
    else:
        timer_fn = fn
    
    t = benchmark.Timer(
        stmt="timer_fn(*inputs, **kwinputs)",
        globals={"timer_fn": timer_fn, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        cache_info = " (with L2 cache thrashing)" if flush_cache else ""
        print(f"Ran {repeats} measurement iteration(s){cache_info} (with {num_warmup} explicit warmup + Timer's automatic warmup)")
        print(m)
    return t, m


def flops(batch, seqlen_q, seqlen_k, headdim, nheads, causal, mode="fwd"):
    """Calculate FLOPS for attention operation.
    
    Args:
        batch: Batch size
        seqlen_q: Query sequence length (can be a list for variable lengths)
        seqlen_k: Key sequence length (can be a list for variable lengths)
        headdim: Head dimension
        nheads: Number of attention heads
        causal: Whether causal masking is applied (reduces FLOPS by 2x)
        mode: "fwd" for forward pass only
    
    Returns:
        Total FLOPS for the operation
    """
    assert mode == "fwd", "This benchmark only measures forward pass"
    
    # Handle variable-length sequences
    if isinstance(seqlen_q, list):
        total_flops = 0
        for sq, sk in zip(seqlen_q, seqlen_k):
            # Base FLOPS: QK^T + Softmax + Attention * V
            # Total: ~4 * seqlen_q * seqlen_k * nheads * headdim
            # For causal, we only compute half the matrix
            f = 4 * sq * sk * nheads * headdim // (2 if causal else 1)
            total_flops += f
        return total_flops
    else:
        # Constant length
        f = 4 * batch * seqlen_q * seqlen_k * nheads * headdim // (2 if causal else 1)
        return f


def efficiency(flop, time):
    """Convert FLOPS and time to TFLOPs/s (Tera-FLOPS per second).
    
    Args:
        flop: Total FLOPS
        time: Time in seconds
    
    Returns:
        Throughput in TFLOPs/s
    """
    return (flop / time / 10**12) if not math.isnan(time) and time > 0 else 0.0


def time_forward(func, *args, flush_cache=False, warmup=True, **kwargs):
    """Benchmark forward pass only.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for func
        flush_cache: If True, thrash L2 cache before each run to ensure cold cache
        warmup: If True, perform warmup runs before timing (default: True)
        **kwargs: Keyword arguments for func (repeats, verbose, etc.)
    
    Returns:
        Mean forward time in seconds
    """
    time_f = benchmark_forward(func, *args, flush_cache=flush_cache, warmup=warmup, **kwargs)
    return time_f[1].mean


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark FA3 attention on Hopper GPU with variable-length sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input file format examples:

1. Simple format (one length per line, seqlen_k = seqlen_q):
   1024
   512
   2048

2. CSV format (seqlen_q,seqlen_k):
   seqlen_q,seqlen_k
   1024,1024
   512,512
   2048,2048

3. Space-separated format:
   1024 1024
   512 512
   2048 2048

Lines starting with # are treated as comments and ignored.
        """
    )
    parser.add_argument('--lengths-file', type=str, required=True,
                        help='Input file containing sequence lengths (one per line or CSV format)')
    parser.add_argument('--headdim', type=int, default=128, help='Head dimension')
    parser.add_argument('--dim', type=int, default=2048, help='Total model dimension')
    parser.add_argument('--nheads-q', type=int, default=None,
                        help='Number of query heads (if not set, uses dim // headdim)')
    parser.add_argument('--nheads-kv', type=int, default=None,
                        help='Number of KV heads for GQA (if not set, equals nheads-q; '
                             'must divide nheads-q)')
    parser.add_argument('--repeats', type=int, default=1, help='Number of benchmark iterations')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], help='Data type')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--page-size', type=int, default=16, help='Page size for paged KV cache')
    parser.add_argument('--simulate-reshape-cache', action='store_true', 
                        help='Simulate vLLM by calling reshape_and_cache_flash before attention kernel')
    parser.add_argument('--contiguous-blocks', action='store_true',
                        help='Use contiguous (sequential) block allocation instead of scattered blocks')
    parser.add_argument('--flush-cache', action='store_true',
                        help='Thrash L2 cache before each benchmark run to ensure cold cache (avoids cache effects)')
    parser.add_argument('--prefill-sm-percentage', type=float, default=0.0,
                        help='Percentage of SMs dedicated to prefill (0.0-0.9). Default: 0.0')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip warmup runs before timing (default: warmup is enabled)')
    parser.add_argument('--tile-scheduler-debug', action='store_true',
                        help='Enable printf debug output in tile scheduler (default: disabled)')
    parser.add_argument('--decode-mode', action='store_true',
                        help='Treat all sequences as decode (seqlen_q=1, seqlen_k from file)')
    
    args = parser.parse_args()
    
    # Parse sequence lengths from file
    print(f"Reading sequence lengths from: {args.lengths_file}")
    seqlen_q_list, seqlen_k_list = parse_lengths_file(args.lengths_file)
    batch_size = len(seqlen_q_list)
    
    if args.decode_mode:
        # For decode mode, seqlen_q is always 1, seqlen_k comes from file
        seqlen_q_list = [1] * batch_size
        print(f"Decode mode: Setting all seqlen_q to 1, using seqlen_k from file")
    
    print(f"Loaded {batch_size} sequences")
    print(f"  seqlen_q range: [{min(seqlen_q_list)}, {max(seqlen_q_list)}]")
    print(f"  seqlen_k range: [{min(seqlen_k_list)}, {max(seqlen_k_list)}]")
    print(f"  seqlen_q mean: {sum(seqlen_q_list) / len(seqlen_q_list):.1f}")
    print(f"  seqlen_k mean: {sum(seqlen_k_list) / len(seqlen_k_list):.1f}")
    
    # Parse dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16}
    dtype = dtype_map[args.dtype]
    
    # Configuration
    repeats = args.repeats
    device = args.device
    fa_version = 3  # FA3 for Hopper (compute capability 9.0)
    headdim = args.headdim
    dim = args.dim
    dropout_p = 0.0

    # Derive number of Q and KV heads
    if args.nheads_q is None:
        nheads_q = dim // headdim
    else:
        nheads_q = args.nheads_q
    
    # Only check dim consistency if both dim and nheads_q are explicitly set
    if args.dim != 2048 or args.nheads_q is not None:
        expected_dim = nheads_q * headdim
        if dim != expected_dim:
            print(f"WARNING: dim ({dim}) != nheads-q ({nheads_q}) * headdim ({headdim}) = {expected_dim}")
            print(f"  Using nheads-q={nheads_q} and headdim={headdim} (ignoring dim={dim})")

    if args.nheads_kv is None:
        nheads_kv = nheads_q
    else:
        nheads_kv = args.nheads_kv
    assert nheads_q % nheads_kv == 0, \
        f"nheads-q ({nheads_q}) must be divisible by nheads-kv ({nheads_kv}) for GQA"
    
    # Set causal=True (vLLM uses causal=True for both prefill and decode)
    causal = True
    
    print("=" * 80)
    print("Hopper GPU (FA3) Forward-Only Attention Benchmark (Variable-Length)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"FA Version: {fa_version}")
    print(f"Repeats: {repeats}")
    print(f"Headdim: {headdim}, Total dim: {dim}")
    print(f"nheads-q: {nheads_q}, nheads-kv: {nheads_kv}")
    print(f"Batch size: {batch_size}")
    print(f"Page size: {args.page_size if args.page_size is not None else 'None (no paging)'}")
    print(f"Block allocation: {'Contiguous (sequential)' if args.contiguous_blocks else 'Scattered (non-sequential)'}")
    print(f"Prefill SM percentage: {args.prefill_sm_percentage:.1%}")
    print(f"Decode mode: {args.decode_mode}")
    print("=" * 80)
    
    # Calculate max_seqlen_q and max_seqlen_k
    max_seqlen_q = max(seqlen_q_list)
    max_seqlen_k = max(seqlen_k_list)
    
    # Print kernel selection info
    qhead_per_khead = nheads_q // nheads_kv if nheads_kv > 0 else 1
    print(f"\nKernel Selection Info (to match first kernel: kBlockM=64, kHeadDimV=128):")
    print(f"  - headdim_v must equal headdim: {headdim} (✓)")
    print(f"  - use_one_mma_wg requires: max_seqlen_q * (nheads_q / nheads_kv) <= 64")
    print(f"    NOTE: For varlen batches, params.seqlen_q = max_seqlen_q (not per-sequence length!)")
    print(f"  - max_seqlen_q={max_seqlen_q} * {qhead_per_khead} = {max_seqlen_q * qhead_per_khead} {'<= 64 ✓' if max_seqlen_q * qhead_per_khead <= 64 else '> 64 ✗'}")
    
    if max_seqlen_q * qhead_per_khead <= 64:
        print(f"  ✓ Will use first kernel (kBlockM=64, kHeadDimV=128)")
    else:
        print(f"  ✗ Will use different kernel (kBlockM=128) because max_seqlen_q * qhead_per_khead > 64")
        print(f"    To fix: reduce max_seqlen_q to <= {64 // qhead_per_khead}")
    
    # Ensure V headdim matches Q/K headdim
    headdim_v = headdim
    print(f"\nDEBUG: Setting headdim_v = {headdim_v} (must equal headdim={headdim} for use_one_mma_wg)")
    
    # Check if use_one_mma_wg will be enabled
    use_one_mma_wg_enabled = (headdim == 128 or headdim == 64) and (max_seqlen_q * qhead_per_khead <= 64)
    
    if not use_one_mma_wg_enabled:
        print(f"WARNING: use_one_mma_wg will NOT be enabled.")
        print(f"  Condition: max_seqlen_q * (nheads_q / nheads_kv) <= 64")
        print(f"  Current: max_seqlen_q={max_seqlen_q} * {qhead_per_khead} = {max_seqlen_q * qhead_per_khead} > 64")
        print(f"  To enable: reduce max_seqlen_q to <= {64 // qhead_per_khead} or increase nheads_kv")
    
    # Calculate total tokens
    total_q_tokens = sum(seqlen_q_list)
    total_k_tokens = sum(seqlen_k_list)
    
    # Always use paged KV format
    page_size = args.page_size
    
    # Create Q tensor (vLLM format: [num_tokens, num_heads, head_size])
    # Generate random data on CPU then move to GPU (faster than GPU random generation)
    q = torch.randn(total_q_tokens, nheads_q, headdim, dtype=dtype).to(device)
    
    # Create KV cache and block tables for variable-length sequences
    # Calculate number of blocks needed for each sequence
    num_blocks_per_seq = [math.ceil(sk / page_size) for sk in seqlen_k_list]
    max_num_blocks_per_seq = max(num_blocks_per_seq) if num_blocks_per_seq else 0
    
    # For decode mode, we need an extra block for the new token
    if args.decode_mode:
        max_num_blocks_per_seq += 1
        num_blocks_per_seq = [nb + 1 for nb in num_blocks_per_seq]
    
    # Calculate total blocks needed
    num_blocks_total = sum(num_blocks_per_seq)
    
    # vLLM format: [2, num_blocks, block_size, num_kv_heads, head_size]
    kv_cache = torch.randn(2, num_blocks_total, page_size, nheads_kv, headdim, dtype=dtype).to(device)
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache = kv_cache[1]  # [num_blocks, block_size, num_kv_heads, head_size]
    
    # Create block_table: (batch_size, max_num_blocks_per_seq)
    block_table = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device)
    
    # Allocate blocks for each sequence
    block_idx = 0
    for b in range(batch_size):
        num_blocks_needed = num_blocks_per_seq[b]
        
        if args.contiguous_blocks:
            # Sequential (contiguous) block allocation
            block_table[b, :num_blocks_needed] = torch.arange(
                block_idx, block_idx + num_blocks_needed, dtype=torch.int32, device=device
            )
            block_idx += num_blocks_needed
        else:
            # Scattered block allocation (default)
            # Generate scattered block indices
            scatter_pattern = []
            if num_blocks_needed == 1:
                scatter_pattern = [block_idx]
            elif num_blocks_needed == 2:
                scatter_pattern = [block_idx, min(block_idx + 5, num_blocks_total - 1)]
            elif num_blocks_needed == 3:
                scatter_pattern = [block_idx, min(block_idx + 8, num_blocks_total - 1), 
                                  min(block_idx + 3, num_blocks_total - 1)]
            else:
                # For 4+ blocks, use scattered pattern
                for i in range(num_blocks_needed):
                    if i == 0:
                        offset = 0
                    elif i % 4 == 1:
                        offset = min(i * 10, num_blocks_total - 1 - block_idx)
                    elif i % 4 == 2:
                        offset = min((i - 1) * 5, num_blocks_total - 1 - block_idx)
                    elif i % 4 == 3:
                        offset = min((i - 2) * 15, num_blocks_total - 1 - block_idx)
                    else:
                        offset = min(i * 7, num_blocks_total - 1 - block_idx)
                    scatter_pattern.append(block_idx + offset)
            
            # Clamp indices to valid range
            scatter_pattern = [max(0, min(idx, num_blocks_total - 1)) for idx in scatter_pattern]
            scatter_pattern = scatter_pattern[:num_blocks_needed]
            
            # Ensure we have enough unique indices
            while len(scatter_pattern) < num_blocks_needed:
                scatter_pattern.append(min(block_idx + len(scatter_pattern), num_blocks_total - 1))
            
            block_table[b, :num_blocks_needed] = torch.tensor(scatter_pattern, dtype=torch.int32, device=device)
            block_idx += num_blocks_needed
    
    # Create new K/V tokens for prefill (if not decode mode)
    if args.decode_mode:
        k_new = None
        v_new = None
    else:
        # For prefill, we have new K/V tokens to cache
        k_new = torch.randn(total_q_tokens, nheads_kv, headdim, dtype=dtype).to(device)
        v_new = torch.randn(total_q_tokens, nheads_kv, headdim_v, dtype=dtype).to(device)
    
    # Create cumulative sequence lengths for Q (cu_seqlens_q)
    # cu_seqlens_q[0] = 0, cu_seqlens_q[i+1] = sum of first i+1 sequence lengths
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, sq in enumerate(seqlen_q_list):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + sq
    
    # Create seqused_k (actual sequence length for each batch item)
    seqused_k = torch.tensor(seqlen_k_list, dtype=torch.int32, device=device)
    
    print(f"\nBatch configuration:")
    print(f"  Total sequences: {batch_size}")
    print(f"  Total Q tokens: {total_q_tokens}")
    print(f"  Total K tokens: {total_k_tokens}")
    print(f"  max_seqlen_q: {max_seqlen_q}")
    print(f"  max_seqlen_k: {max_seqlen_k}")
    print(f"Q shape: {q.shape} (vLLM format: [num_tokens, num_heads, head_size])")
    print(f"KV cache shape: {kv_cache.shape} (vLLM format: [2, num_blocks, block_size, num_kv_heads, head_size])")
    print(f"  key_cache shape: {key_cache.shape}, value_cache shape: {value_cache.shape}")
    if k_new is not None:
        print(f"New K/V tokens shape: {k_new.shape} (vLLM format: [num_tokens, num_kv_heads, head_size])")
    else:
        print(f"New K/V tokens: None (decode mode - tokens already in cache)")
    print(f"nheads-q: {nheads_q}, nheads-kv: {nheads_kv}, "
          f"q_per_kv: {nheads_q // nheads_kv}, "
          f"causal={causal}")
    print(f"Paged KV: ENABLED (page_size={args.page_size})")
    
    # Verify V cache headdim matches expected value
    actual_v_headdim = value_cache.shape[-1]
    if actual_v_headdim != headdim_v:
        print(f"ERROR: V cache headdim mismatch! Expected {headdim_v}, got {actual_v_headdim}")
        print(f"  This will cause kHeadDimV != kHeadDim, preventing use_one_mma_wg")
    else:
        print(f"✓ V cache headdim matches: {actual_v_headdim}")
    
    # Additional checks for kernel selection
    print(f"\nDEBUG Kernel Selection Conditions:")
    print(f"  - headdim: {headdim}, headdim_v: {actual_v_headdim} (must be equal: {headdim == actual_v_headdim})")
    print(f"  - max_seqlen_q: {max_seqlen_q}")
    print(f"  - qhead_per_khead: {qhead_per_khead}")
    print(f"  - use_one_mma_wg condition: max_seqlen_q * qhead_per_khead <= 64 (uses max_seqlen_q for varlen!)")
    print(f"    Current: {max_seqlen_q} * {qhead_per_khead} = {max_seqlen_q * qhead_per_khead} {'<= 64 ✓' if max_seqlen_q * qhead_per_khead <= 64 else '> 64 ✗'}")
    if headdim == 128 or headdim == 64:
        print(f"  - headdim check: {headdim} is 128 or 64 ✓")
    else:
        print(f"  - headdim check: {headdim} is NOT 128 or 64 ✗")
    
    if (headdim == 128 or headdim == 64) and (headdim == actual_v_headdim) and (max_seqlen_q * qhead_per_khead <= 64):
        print(f"\n✓ All conditions met for first kernel (kBlockM=64, kHeadDimV=128)")
    else:
        print(f"\n✗ Conditions NOT met - will use different kernel")
        if headdim != actual_v_headdim:
            print(f"  - Fix: Ensure V cache has headdim={headdim}, not {actual_v_headdim}")
    
    # Calculate FLOPS
    total_flops = flops(batch_size, seqlen_q_list, seqlen_k_list, headdim, nheads_q, causal, mode="fwd")
    
    # Create output tensor
    out = torch.empty(total_q_tokens, nheads_q, headdim, device=device, dtype=dtype)
    
    # Create scheduler_metadata with page_size if specified
    scheduler_metadata = None
    if args.page_size is not None:
        print(f"Using page_size={args.page_size} for scheduler metadata")
        sys.stdout.flush()
        sys.stderr.flush()
        scheduler_metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=nheads_q,
            num_heads_kv=nheads_kv,
            headdim=headdim,
            headdim_v=headdim_v,
            cache_seqlens=seqused_k,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=args.page_size,
            causal=causal,
            window_size=(-1, -1),
            num_splits=1,
            prefill_sm_percentage=args.prefill_sm_percentage,
            num_prefill_batches=batch_size if not args.decode_mode else 0,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        print("get_scheduler_metadata completed")
        sys.stdout.flush()
    
    # Final verification before kernel call
    print(f"\n=== FINAL VERIFICATION BEFORE KERNEL CALL ===")
    print(f"Q shape: {q.shape}")
    print(f"key_cache shape: {key_cache.shape}")
    print(f"value_cache shape: {value_cache.shape}")
    if k_new is not None:
        print(f"New K shape: {k_new.shape}, New V shape: {v_new.shape}")
    else:
        print(f"New K/V: None (decode mode)")
    print(f"Output shape: {out.shape}")
    print(f"Expected V cache headdim: {headdim_v}")
    if value_cache.shape[-1] != headdim_v:
        print(f"ERROR: V cache has wrong headdim! Expected {headdim_v}, got {value_cache.shape[-1]}")
        raise ValueError(f"V cache headdim mismatch: expected {headdim_v}, got {value_cache.shape[-1]}")
    print(f"✓ V cache headdim is correct: {value_cache.shape[-1]}")
    print(f"cu_seqlens_q: {cu_seqlens_q.cpu().numpy()}")
    print(f"seqused_k: {seqused_k.cpu().numpy()}")
    
    # Simulate vLLM's reshape_and_cache_flash if requested
    if args.simulate_reshape_cache and reshape_and_cache_flash_available and not args.decode_mode:
        print(f"\n=== SIMULATING vLLM: Calling reshape_and_cache_flash before attention ===")
        
        # Compute slot_mapping for new tokens
        slot_mapping_list = []
        token_idx = 0
        for seq_idx in range(batch_size):
            seqlen = seqlen_q_list[seq_idx]
            for pos in range(seqlen):
                block_idx = pos // page_size
                block_offset = pos % page_size
                if block_idx < num_blocks_per_seq[seq_idx]:
                    block_number = block_table[seq_idx, block_idx].item()
                    slot = block_number * page_size + block_offset
                    slot_mapping_list.append(slot)
                else:
                    slot_mapping_list.append(-1)  # PAD_SLOT_ID
            token_idx += seqlen
        
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64, device=device)
        print(f"Computed slot_mapping for {len(slot_mapping_list)} new tokens")
        
        # Create dummy scale tensors
        kv_cache_dtype = "auto"
        k_scale = torch.ones(1, dtype=torch.float32, device=device)
        v_scale = torch.ones(1, dtype=torch.float32, device=device)
        
        # Call reshape_and_cache_flash
        print(f"Calling reshape_and_cache_flash...")
        try:
            if reshape_and_cache_flash_func is not None:
                reshape_and_cache_flash_func(
                    k_new,
                    v_new,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                )
                print(f"✓ reshape_and_cache_flash completed")
                torch.cuda.synchronize()
            else:
                print("Error: reshape_and_cache_flash_func is None")
        except Exception as e:
            print(f"Error calling reshape_and_cache_flash: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without reshape_and_cache_flash...")
        print("=" * 80)
    
    # Benchmark
    try:
        print(f"Running benchmark with {repeats} repeats...")
        print(f"\n=== PARAMETERS PASSED TO flash_attn_varlen_func ===")
        print(f"q shape: {q.shape}, dtype: {q.dtype}")
        print(f"k shape: {key_cache.shape}, dtype: {key_cache.dtype}")
        print(f"v shape: {value_cache.shape}, dtype: {value_cache.dtype}")
        print(f"out shape: {out.shape}, dtype: {out.dtype}")
        print(f"max_seqlen_q: {max_seqlen_q}")
        print(f"max_seqlen_k: {max_seqlen_k}")
        print(f"cu_seqlens_q: {cu_seqlens_q.cpu().numpy()}")
        print(f"seqused_k: {seqused_k.cpu().numpy()}")
        print(f"dropout_p: {dropout_p}")
        print(f"causal: {causal}")
        print(f"block_table shape: {block_table.shape}")
        print(f"fa_version: {fa_version}")
        print("=" * 80)
        
        func_kwargs = {
            'q': q,
            'k': key_cache,
            'v': value_cache,
            'out': out,
            'max_seqlen_q': max_seqlen_q,
            'cu_seqlens_q': cu_seqlens_q,
            'max_seqlen_k': max_seqlen_k,
            'seqused_k': seqused_k,
            'block_table': block_table,
            'dropout_p': dropout_p,
            'causal': causal,
            'scheduler_metadata': scheduler_metadata,
            'fa_version': fa_version,
            'num_splits': 1,
            'prefill_sm_percentage': args.prefill_sm_percentage,
            'num_prefill_batches': batch_size if not args.decode_mode else 0,
            'tile_scheduler_debug': args.tile_scheduler_debug,
            'repeats': repeats,
            'verbose': True
        }
        
        func_kwargs['cu_seqlens_k'] = None  # Not needed when using block_table
        
        f_time = time_forward(
            flash_attn_varlen_func,
            flush_cache=args.flush_cache,
            warmup=not args.no_warmup,
            **func_kwargs
        )
        
        # vLLM flattens output to [num_tokens, num_heads * head_size]
        out_flat = out.view(total_q_tokens, nheads_q * headdim)
        
        speed = efficiency(total_flops, f_time)
        print(f"\nTime: {f_time*1000:.3f} ms")
        print(f"Total FLOPS: {total_flops/1e12:.2f} TFLOPS")
        print(f"Throughput: {speed:.2f} TFLOPs/s")
    except Exception as e:
        print(f"Error benchmarking: {e}")
        import traceback
        traceback.print_exc()
        f_time = float('nan')
        speed = 0.0
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Batch':<20} {'Time (ms)':<15} {'TFLOPs':<15} {'TFLOPs/s':<15}")
    print("-" * 80)
    if not math.isnan(f_time):
        print(f"{'Variable-Length':<20} {f_time*1000:<15.3f} {total_flops/1e12:<15.2f} {speed:<15.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

