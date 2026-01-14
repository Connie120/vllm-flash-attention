"""
Benchmark script for forward-only attention on Hopper GPU (FA3).
This script measures inference performance using flash_attn_varlen_func with fa_version=3.
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
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        headdim: Head dimension
        nheads: Number of attention heads
        causal: Whether causal masking is applied (reduces FLOPS by 2x)
        mode: "fwd" for forward pass only
    
    Returns:
        Total FLOPS for the operation
    """
    assert mode == "fwd", "This benchmark only measures forward pass"
    # Base FLOPS: QK^T (batch * seqlen_q * seqlen_k * nheads * headdim)
    #            + Softmax (batch * seqlen_q * seqlen_k * nheads)
    #            + Attention * V (batch * seqlen_q * seqlen_k * nheads * headdim)
    # Total: ~4 * batch * seqlen_q * seqlen_k * nheads * headdim
    # For causal, we only compute half the matrix
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
    parser = argparse.ArgumentParser(description='Benchmark FA3 attention on Hopper GPU')
    parser.add_argument('--headdim', type=int, default=128, help='Head dimension')
    parser.add_argument('--dim', type=int, default=2048, help='Total model dimension')
    parser.add_argument('--nheads-q', type=int, default=None,
                        help='Number of query heads (if not set, uses dim // headdim)')
    parser.add_argument('--nheads-kv', type=int, default=None,
                        help='Number of KV heads for GQA (if not set, equals nheads-q; '
                             'must divide nheads-q)')
    parser.add_argument('--batch-prefill', type=int, default=0, help='Batch size for prefill')
    parser.add_argument('--seqlen-prefill', type=int, default=1024, help='Sequence length for prefill')
    parser.add_argument('--batch-decode', type=int, default=0, help='Batch size for decode')
    parser.add_argument('--seqlen-q-decode', type=int, default=1, help='Query sequence length for decode')
    parser.add_argument('--seqlen-k-decode', type=int, default=1024, help='Key sequence length (context) for decode')
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
                        help='Percentage of SMs dedicated to prefill (0.0-0.9). Default: 0.5 (50%% prefill, 50%% decode)')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip warmup runs before timing (default: warmup is enabled)')
    parser.add_argument('--tile-scheduler-debug', action='store_true',
                        help='Enable printf debug output in tile scheduler (default: disabled)')
    
    args = parser.parse_args()
    
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
    
    # Prefill: causal=True, Decode: causal=False (decode typically uses causal=False for attention)
    # However, vLLM may use causal=True for decode in some cases
    causal_prefill = True
    causal_decode = False  # For decode, typically causal=False (but vLLM may override this)
    
    print("=" * 80)
    print("Hopper GPU (FA3) Forward-Only Attention Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"FA Version: {fa_version}")
    print(f"Repeats: {repeats}")
    print(f"Headdim: {headdim}, Total dim: {dim}")
    print(f"nheads-q: {nheads_q}, nheads-kv: {nheads_kv}")
    print(f"Page size: {args.page_size if args.page_size is not None else 'None (no paging)'}")
    print(f"Block allocation: {'Contiguous (sequential)' if args.contiguous_blocks else 'Scattered (non-sequential)'}")
    print(f"Prefill SM percentage: {args.prefill_sm_percentage:.1%} ({args.prefill_sm_percentage*100:.0f}%% prefill, {(1-args.prefill_sm_percentage)*100:.0f}%% decode)")
    print("=" * 80)
    
    # ========== Combined Prefill + Decode in single batch ==========
    print("\n### Combined Prefill + Decode Batch ###")
    batch_prefill = args.batch_prefill
    seqlen_prefill = args.seqlen_prefill
    batch_decode = args.batch_decode
    seqlen_q_decode = args.seqlen_q_decode
    seqlen_k_decode = args.seqlen_k_decode
    
    # Calculate max_seqlen_q_combined from actual sequence lengths
    # Handle different cases: pure decode, pure prefill, or mixed
    if batch_prefill > 0 and batch_decode > 0:
        # Mixed batch: take max of both
        max_seqlen_q_combined_estimate = max(seqlen_prefill, seqlen_q_decode)
        max_seqlen_k_combined_estimate = max(seqlen_prefill, seqlen_k_decode)
    elif batch_prefill > 0:
        # Pure prefill batch
        max_seqlen_q_combined_estimate = seqlen_prefill
        max_seqlen_k_combined_estimate = seqlen_prefill
    elif batch_decode > 0:
        # Pure decode batch
        max_seqlen_q_combined_estimate = seqlen_q_decode
        max_seqlen_k_combined_estimate = seqlen_k_decode
    else:
        # No batches (shouldn't happen, but handle gracefully)
        max_seqlen_q_combined_estimate = 0
        max_seqlen_k_combined_estimate = 0
    
    # Print kernel selection info
    qhead_per_khead = nheads_q // nheads_kv if nheads_kv > 0 else 1
    print(f"\nKernel Selection Info (to match first kernel: kBlockM=64, kHeadDimV=128):")
    print(f"  - headdim_v must equal headdim: {headdim} (✓)")
    print(f"  - use_one_mma_wg requires: max_seqlen_q * (nheads_q / nheads_kv) <= 64")
    print(f"    NOTE: For varlen batches, params.seqlen_q = max_seqlen_q (not per-sequence length!)")
    
    if batch_prefill > 0 and batch_decode > 0:
        # Mixed batch
        print(f"  - Mixed batch (prefill + decode):")
        print(f"    Prefill: max_seqlen_q={seqlen_prefill} * {qhead_per_khead} = {seqlen_prefill * qhead_per_khead} {'<= 64 ✓' if seqlen_prefill * qhead_per_khead <= 64 else '> 64 ✗'}")
        print(f"    Decode: max_seqlen_q={seqlen_q_decode} * {qhead_per_khead} = {seqlen_q_decode * qhead_per_khead} {'<= 64 ✓' if seqlen_q_decode * qhead_per_khead <= 64 else '> 64 ✗'}")
        print(f"    Combined max: {max_seqlen_q_combined_estimate} * {qhead_per_khead} = {max_seqlen_q_combined_estimate * qhead_per_khead} {'<= 64 ✓' if max_seqlen_q_combined_estimate * qhead_per_khead <= 64 else '> 64 ✗'}")
    elif batch_prefill > 0:
        # Pure prefill batch
        print(f"  - Pure prefill batch:")
        print(f"    max_seqlen_q={seqlen_prefill} * {qhead_per_khead} = {seqlen_prefill * qhead_per_khead} {'<= 64 ✓' if seqlen_prefill * qhead_per_khead <= 64 else '> 64 ✗'}")
    elif batch_decode > 0:
        # Pure decode batch
        print(f"  - Pure decode batch:")
        print(f"    max_seqlen_q={seqlen_q_decode} * {qhead_per_khead} = {seqlen_q_decode * qhead_per_khead} {'<= 64 ✓' if seqlen_q_decode * qhead_per_khead <= 64 else '> 64 ✗'}")
        print(f"    (Each decode sequence has seqlen_q={seqlen_q_decode}, so max_seqlen_q={seqlen_q_decode})")
    
    if max_seqlen_q_combined_estimate * qhead_per_khead <= 64:
        print(f"  ✓ Will use first kernel (kBlockM=64, kHeadDimV=128)")
    else:
        print(f"  ✗ Will use different kernel (kBlockM=128) because max_seqlen_q * qhead_per_khead > 64")
        print(f"    To fix: reduce max_seqlen_q to <= {64 // qhead_per_khead}")
    
    # Create QKV for prefill
    # To match first kernel (kBlockM=64, kHeadDimV=128), we need:
    # 1. headdim_v == headdim == 128 (not 256) - CRITICAL: kHeadDimV must equal kHeadDim
    # 2. use_one_mma_wg enabled: seqlen_q * (nheads_q / nheads_kv) <= 64
    # 3. Arch >= 90 (Hopper)
    # 4. kHeadDim == 128 or 64
    headdim_v = headdim  # Ensure V headdim matches Q/K headdim (128, not 256)
    print(f"\nDEBUG: Setting headdim_v = {headdim_v} (must equal headdim={headdim} for use_one_mma_wg)")
    
    # Check if use_one_mma_wg will be enabled
    # IMPORTANT: For varlen batches, params.seqlen_q = max_seqlen_q, not per-sequence length!
    qhead_per_khead = nheads_q // nheads_kv
    # For varlen, use max_seqlen_q (which is what params.seqlen_q will be)
    use_one_mma_wg_enabled = (headdim == 128 or headdim == 64) and (max_seqlen_q_combined_estimate * qhead_per_khead <= 64)
    
    if not use_one_mma_wg_enabled and (batch_prefill > 0 or batch_decode > 0):
        print(f"WARNING: use_one_mma_wg will NOT be enabled.")
        print(f"  Condition: max_seqlen_q * (nheads_q / nheads_kv) <= 64")
        print(f"  Current: max_seqlen_q={max_seqlen_q_combined_estimate} * {qhead_per_khead} = {max_seqlen_q_combined_estimate * qhead_per_khead} > 64")
        if batch_decode > 0 and batch_prefill == 0:
            print(f"  Note: For pure decode batch, max_seqlen_q = seqlen_q_decode = {seqlen_q_decode}")
        print(f"  To enable: reduce max_seqlen_q to <= {64 // qhead_per_khead} or increase nheads_kv")
    
    # Create QKV tensors (handle empty batches)
    total_q_prefill = batch_prefill * seqlen_prefill if batch_prefill > 0 else 0
    total_q_decode = batch_decode * seqlen_q_decode if batch_decode > 0 else 0
    
    # Always use paged KV format for both prefill and decode
    page_size = args.page_size
    
    # Create Q tensors (vLLM format: [num_tokens, num_heads, head_size])
    # Generate random data on CPU then move to GPU (faster than GPU random generation)
    if batch_prefill > 0:
        q_prefill = torch.randn(total_q_prefill, nheads_q, headdim, dtype=dtype).to(device)
    else:
        q_prefill = torch.empty(0, nheads_q, headdim, device=device, dtype=dtype)
    
    if batch_decode > 0:
        q_decode = torch.randn(total_q_decode, nheads_q, headdim, dtype=dtype).to(device)
    else:
        q_decode = torch.empty(0, nheads_q, headdim, device=device, dtype=dtype)
    
    # vLLM format: KV cache is [2, num_blocks, block_size, num_kv_heads, head_size]
    # For prefill: new K/V tokens are [num_tokens, num_kv_heads, head_size] (will be cached)
    # For decode: new K/V tokens are None (already in cache)
    num_blocks_total_prefill = 0  # Initialize for use in decode block_table calculation
    num_blocks_allocated_prefill = 0  # Initialize for scattered allocation
    if batch_prefill > 0:
        max_num_blocks_per_seq_prefill = math.ceil(seqlen_prefill / page_size)
        # Allocate exactly the number of blocks needed
        num_blocks_total_prefill = max_num_blocks_per_seq_prefill * batch_prefill
        num_blocks_allocated_prefill = num_blocks_total_prefill
        # vLLM format: [2, num_blocks, block_size, num_kv_heads, head_size]
        # Generate random data on CPU then move to GPU (faster than GPU random generation)
        kv_cache_prefill = torch.randn(2, num_blocks_allocated_prefill, page_size, nheads_kv, headdim, dtype=dtype).to(device)
        # New K/V tokens for prefill: [num_tokens, num_kv_heads, head_size]
        k_prefill_new = torch.randn(total_q_prefill, nheads_kv, headdim, dtype=dtype).to(device)
        v_prefill_new = torch.randn(total_q_prefill, nheads_kv, headdim_v, dtype=dtype).to(device)
        # Create block_table for prefill: (batch_prefill, max_num_blocks_per_seq_prefill)
        # Simulate scattered memory access by using non-sequential block indices
        # This makes the benchmark more realistic since in real vLLM scenarios, blocks
        # might not be allocated contiguously due to memory fragmentation.
        
        block_table_prefill = torch.zeros(batch_prefill, max_num_blocks_per_seq_prefill, dtype=torch.int32, device=device)
        
        if args.contiguous_blocks:
            # Sequential (contiguous) block allocation
            block_idx = 0
            for b in range(batch_prefill):
                num_blocks_needed = math.ceil(seqlen_prefill / page_size)
                block_table_prefill[b, :num_blocks_needed] = torch.arange(
                    block_idx, block_idx + num_blocks_needed, dtype=torch.int32, device=device
                )
                block_idx += num_blocks_needed
        else:
            # Scattered block allocation (default)
            # Track used block indices to avoid collisions
            used_blocks = set()
            max_valid_idx = num_blocks_allocated_prefill - 1
            
            for b in range(batch_prefill):
                num_blocks_needed = math.ceil(seqlen_prefill / page_size)
                # Generate scattered block indices instead of sequential
                # Pattern: scatter blocks with varying offsets relative to sequence start
                # Each sequence gets a starting block index, then scatters within its range
                seq_start_idx = b * max_num_blocks_per_seq_prefill
                scatter_pattern = []
                
                if num_blocks_needed == 1:
                    scatter_pattern = [seq_start_idx]
                elif num_blocks_needed == 2:
                    # Example: [start, start+5] but wrap if needed
                    scatter_pattern = [seq_start_idx, min(seq_start_idx + 5, max_valid_idx)]
                elif num_blocks_needed == 3:
                    scatter_pattern = [seq_start_idx, min(seq_start_idx + 8, max_valid_idx), min(seq_start_idx + 3, max_valid_idx)]
                else:
                    # For 4+ blocks, use pattern similar to user's example: [0, 10, 5, 15, ...]
                    # But make offsets relative to sequence start and wrap within bounds
                    for i in range(num_blocks_needed):
                        if i == 0:
                            offset = 0
                        elif i % 4 == 1:
                            offset = min(i * 10, max_valid_idx - seq_start_idx)  # Large jump
                        elif i % 4 == 2:
                            offset = min((i - 1) * 5, max_valid_idx - seq_start_idx)  # Medium jump
                        elif i % 4 == 3:
                            offset = min((i - 2) * 15, max_valid_idx - seq_start_idx)  # Very large jump
                        else:
                            offset = min(i * 7, max_valid_idx - seq_start_idx)  # Default medium jump
                        scatter_pattern.append(seq_start_idx + offset)
                
                # Ensure all indices are within bounds and find unused blocks
                scattered_indices = []
                for idx in scatter_pattern:
                    # Clamp to valid range
                    idx = max(0, min(idx, max_valid_idx))
                    # Try to find an unused block if this one is taken
                    attempts = 0
                    original_idx = idx
                    while idx in used_blocks and attempts < 100:
                        idx = (idx + 1) % num_blocks_allocated_prefill
                        attempts += 1
                    # If we couldn't find an unused block, use sequential fallback for this sequence
                    if idx in used_blocks:
                        # Fall back to sequential allocation for this sequence
                        scattered_indices = []
                        for i in range(num_blocks_needed):
                            candidate = seq_start_idx + i
                            if candidate <= max_valid_idx:
                                scattered_indices.append(candidate)
                            else:
                                # Wrap around if needed
                                scattered_indices.append(candidate % num_blocks_allocated_prefill)
                        break
                    scattered_indices.append(idx)
                    used_blocks.add(idx)
                
                # Ensure we have exactly num_blocks_needed unique indices
                scattered_indices = list(dict.fromkeys(scattered_indices))  # Remove duplicates while preserving order
                while len(scattered_indices) < num_blocks_needed:
                    # Find next available block
                    for candidate in range(num_blocks_allocated_prefill):
                        if candidate not in used_blocks:
                            scattered_indices.append(candidate)
                            used_blocks.add(candidate)
                            break
                    else:
                        # Fall back to sequential if we run out of space
                        scattered_indices = list(range(seq_start_idx, min(seq_start_idx + num_blocks_needed, num_blocks_allocated_prefill)))
                        if len(scattered_indices) < num_blocks_needed:
                            # Fill remaining with wrap-around
                            for i in range(len(scattered_indices), num_blocks_needed):
                                scattered_indices.append(i % num_blocks_allocated_prefill)
                        break
                
                scattered_indices = scattered_indices[:num_blocks_needed]
                scattered_indices_tensor = torch.tensor(scattered_indices, dtype=torch.int32, device=device)
                block_table_prefill[b, :num_blocks_needed] = scattered_indices_tensor
    else:
        kv_cache_prefill = None
        k_prefill_new = None
        v_prefill_new = None
        block_table_prefill = None
        max_num_blocks_per_seq_prefill = 0
    
    # Create KV cache for decode: [2, num_blocks, block_size, num_kv_heads, head_size]
    if batch_decode > 0:
        # For decode, we need blocks for existing tokens (seqlen_k_decode) PLUS one more block
        # for the new token being written. So we need ceil(seqlen_k_decode / page_size) blocks
        # for existing tokens, plus 1 more for the new token.
        max_num_blocks_per_seq_decode = math.ceil(seqlen_k_decode / page_size)
        # When simulating reshape_and_cache_flash, we need an extra block for the new token
        # So allocate max_num_blocks_per_seq_decode + 1 blocks per sequence
        max_num_blocks_per_seq_decode_with_new_token = max_num_blocks_per_seq_decode + 1
        # Calculate number of blocks needed (including space for new token)
        num_blocks_total_decode = max_num_blocks_per_seq_decode_with_new_token * batch_decode
        # For decode, new K/V tokens are None (already in cache)
        k_decode_new = None
        v_decode_new = None
        # Create block_table for decode: (batch_decode, max_num_blocks_per_seq_decode)
        # Simulate scattered memory access by using non-sequential block indices
        # This makes the benchmark more realistic since in real vLLM scenarios, blocks
        # might not be allocated contiguously due to memory fragmentation.
        # Allocate exactly the number of blocks needed
        num_blocks_allocated_decode = num_blocks_total_decode
        # KV cache with exact allocation
        # vLLM format: [2, num_blocks, block_size, num_kv_heads, head_size]
        # Generate random data on CPU then move to GPU (faster than GPU random generation)
        kv_cache_decode = torch.randn(2, num_blocks_allocated_decode, page_size, nheads_kv, headdim, dtype=dtype).to(device)
        
        # Block indices are relative to decode cache [0, num_blocks_allocated_decode - 1]
        # Will be offset by num_blocks_allocated_prefill when combining with prefill cache
        # Use max_num_blocks_per_seq_decode_with_new_token to accommodate new token
        block_table_decode = torch.zeros(batch_decode, max_num_blocks_per_seq_decode_with_new_token, dtype=torch.int32, device=device)
        
        if args.contiguous_blocks:
            # Sequential (contiguous) block allocation
            block_idx = 0
            for b in range(batch_decode):
                num_blocks_needed = math.ceil(seqlen_k_decode / page_size)
                num_blocks_to_allocate = num_blocks_needed + 1  # Extra block for new token
                block_table_decode[b, :num_blocks_to_allocate] = torch.arange(
                    block_idx, block_idx + num_blocks_to_allocate, dtype=torch.int32, device=device
                )
                block_idx += num_blocks_to_allocate
        else:
            # Scattered block allocation (default)
            # Track used block indices to avoid collisions within decode cache
            used_blocks_decode = set()
            max_valid_idx_decode = num_blocks_allocated_decode - 1
            
            for b in range(batch_decode):
                # num_blocks_needed for existing tokens
                num_blocks_needed = math.ceil(seqlen_k_decode / page_size)
                # But we allocate one more block for the new token being written
                num_blocks_to_allocate = num_blocks_needed + 1
                # Generate scattered block indices instead of sequential
                # Pattern: scatter blocks with varying offsets relative to sequence start
                seq_start_idx = b * max_num_blocks_per_seq_decode
                scatter_pattern = []
                
                if num_blocks_to_allocate == 1:
                    scatter_pattern = [seq_start_idx]
                elif num_blocks_to_allocate == 2:
                    scatter_pattern = [seq_start_idx, min(seq_start_idx + 5, max_valid_idx_decode)]
                elif num_blocks_to_allocate == 3:
                    scatter_pattern = [seq_start_idx, min(seq_start_idx + 8, max_valid_idx_decode), min(seq_start_idx + 3, max_valid_idx_decode)]
                else:
                    # For 4+ blocks, use pattern similar to user's example: [0, 10, 5, 15, ...]
                    # But make offsets relative to sequence start and wrap within bounds
                    for i in range(num_blocks_to_allocate):
                        if i == 0:
                            offset = 0
                        elif i % 4 == 1:
                            offset = min(i * 10, max_valid_idx_decode - seq_start_idx)  # Large jump
                        elif i % 4 == 2:
                            offset = min((i - 1) * 5, max_valid_idx_decode - seq_start_idx)  # Medium jump
                        elif i % 4 == 3:
                            offset = min((i - 2) * 15, max_valid_idx_decode - seq_start_idx)  # Very large jump
                        else:
                            offset = min(i * 7, max_valid_idx_decode - seq_start_idx)  # Default medium jump
                        scatter_pattern.append(seq_start_idx + offset)
                
                # Ensure all indices are within bounds and find unused blocks
                scattered_indices = []
                for idx in scatter_pattern:
                    # Clamp to valid range
                    idx = max(0, min(idx, max_valid_idx_decode))
                    # Try to find an unused block if this one is taken
                    attempts = 0
                    while idx in used_blocks_decode and attempts < 100:
                        idx = (idx + 1) % num_blocks_allocated_decode
                        attempts += 1
                    # If we couldn't find an unused block, use sequential fallback for this sequence
                    if idx in used_blocks_decode:
                        # Fall back to sequential allocation for this sequence
                        scattered_indices = []
                        for i in range(num_blocks_to_allocate):
                            candidate = seq_start_idx + i
                            if candidate <= max_valid_idx_decode:
                                scattered_indices.append(candidate)
                            else:
                                # Wrap around if needed
                                scattered_indices.append(candidate % num_blocks_allocated_decode)
                        break
                    scattered_indices.append(idx)
                    used_blocks_decode.add(idx)
                
                # Ensure we have exactly num_blocks_to_allocate unique indices
                scattered_indices = list(dict.fromkeys(scattered_indices))  # Remove duplicates while preserving order
                while len(scattered_indices) < num_blocks_to_allocate:
                    # Find next available block
                    for candidate in range(num_blocks_allocated_decode):
                        if candidate not in used_blocks_decode:
                            scattered_indices.append(candidate)
                            used_blocks_decode.add(candidate)
                            break
                    else:
                        # Fall back to sequential if we run out of space
                        scattered_indices = list(range(seq_start_idx, min(seq_start_idx + num_blocks_to_allocate, num_blocks_allocated_decode)))
                        if len(scattered_indices) < num_blocks_to_allocate:
                            # Fill remaining with wrap-around
                            for i in range(len(scattered_indices), num_blocks_to_allocate):
                                scattered_indices.append(i % num_blocks_allocated_decode)
                        break
                
                scattered_indices = scattered_indices[:num_blocks_to_allocate]
                scattered_indices_tensor = torch.tensor(scattered_indices, dtype=torch.int32, device=device)
                block_table_decode[b, :num_blocks_to_allocate] = scattered_indices_tensor
    else:
        kv_cache_decode = None
        k_decode_new = None
        v_decode_new = None
        block_table_decode = None
        max_num_blocks_per_seq_decode = 0
    
    # Concatenate Q tensors (vLLM format: [num_tokens, num_heads, head_size])
    q_combined = torch.cat([q_prefill, q_decode], dim=0) if (batch_prefill > 0 or batch_decode > 0) else torch.empty(0, nheads_q, headdim, device=device, dtype=dtype)
    
    # Combine KV caches: [2, num_blocks_total, block_size, num_kv_heads, head_size]
    if batch_prefill > 0 and batch_decode > 0:
        # Mixed batch: combine prefill and decode KV caches
        num_blocks_total = num_blocks_total_prefill + num_blocks_total_decode
        kv_cache_combined = torch.cat([kv_cache_prefill, kv_cache_decode], dim=1)
        # New K/V tokens: only prefill has new tokens
        k_new_combined = k_prefill_new
        v_new_combined = v_prefill_new
    elif batch_prefill > 0:
        # Pure prefill with KV cache
        kv_cache_combined = kv_cache_prefill
        k_new_combined = k_prefill_new
        v_new_combined = v_prefill_new
    elif batch_decode > 0:
        # Pure decode with KV cache
        kv_cache_combined = kv_cache_decode
        k_new_combined = None
        v_new_combined = None
    else:
        # Empty batch (shouldn't happen, but handle gracefully)
        kv_cache_combined = torch.empty(2, 0, page_size, nheads_kv, headdim, device=device, dtype=dtype)
        k_new_combined = None
        v_new_combined = None
    
    # vLLM splits KV cache: key_cache, value_cache = kv_cache.unbind(0)
    key_cache = kv_cache_combined[0]  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache = kv_cache_combined[1]  # [num_blocks, block_size, num_kv_heads, head_size]
    
    # Create combined cumulative sequence lengths for Q
    # For paged KV, we don't use cu_seqlens_k, but we still need cu_seqlens_q
    if batch_prefill > 0 and batch_decode > 0:
        # Mixed batch: combine prefill and decode
        cu_seqlens_q_prefill = torch.arange(0, (batch_prefill + 1) * seqlen_prefill, 
                                           step=seqlen_prefill, dtype=torch.int32, device=device)
        cu_seqlens_q_decode = torch.arange(0, (batch_decode + 1) * seqlen_q_decode,
                                          step=seqlen_q_decode, dtype=torch.int32, device=device)
        cu_seqlens_q_decode = cu_seqlens_q_decode + total_q_prefill
        cu_seqlens_q_combined = torch.cat([cu_seqlens_q_prefill, cu_seqlens_q_decode[1:]], dim=0)
    elif batch_prefill > 0:
        # Pure prefill batch
        cu_seqlens_q_combined = torch.arange(0, (batch_prefill + 1) * seqlen_prefill, 
                                           step=seqlen_prefill, dtype=torch.int32, device=device)
    elif batch_decode > 0:
        # Pure decode batch
        cu_seqlens_q_combined = torch.arange(0, (batch_decode + 1) * seqlen_q_decode,
                                          step=seqlen_q_decode, dtype=torch.int32, device=device)
    else:
        # Empty batch (shouldn't happen, but handle gracefully)
        cu_seqlens_q_combined = torch.tensor([0], dtype=torch.int32, device=device)
    
    # Calculate total batch size
    total_batch = batch_prefill + batch_decode
    
    # Create seqused_k and block_table for paged KV (sequence lengths for each batch item)
    # Note: These must be int32 (not args.dtype) as they are metadata tensors, not data tensors
    # Always use paged KV format for all cases (prefill, decode, and mixed)
    if batch_prefill > 0 and batch_decode > 0:
        # Mixed batch: combine prefill and decode
        seqused_k = torch.cat([
            torch.full((batch_prefill,), seqlen_prefill, dtype=torch.int32, device=device),
            torch.full((batch_decode,), seqlen_k_decode, dtype=torch.int32, device=device)
        ])
        # Create combined block_table with max blocks across both prefill and decode
        max_num_blocks_per_seq = max(max_num_blocks_per_seq_prefill, max_num_blocks_per_seq_decode_with_new_token)
        block_table_combined = torch.zeros(total_batch, max_num_blocks_per_seq, dtype=torch.int32, device=device)
        # Copy prefill block_table (indices are already correct: [0, num_blocks_allocated_prefill - 1])
        block_table_combined[:batch_prefill, :max_num_blocks_per_seq_prefill] = block_table_prefill
        # Copy decode block_table and offset indices to account for prefill cache
        # Decode block_table has indices [0, num_blocks_allocated_decode - 1] relative to decode cache
        # After concatenation, decode cache starts at num_blocks_allocated_prefill
        if max_num_blocks_per_seq_decode_with_new_token > 0:
            block_table_decode_offset = block_table_decode + num_blocks_allocated_prefill
            block_table_combined[batch_prefill:, :max_num_blocks_per_seq_decode_with_new_token] = block_table_decode_offset
    elif batch_prefill > 0:
        # Pure prefill batch with paged KV
        seqused_k = torch.full((total_batch,), seqlen_prefill, dtype=torch.int32, device=device)
        block_table_combined = block_table_prefill
    elif batch_decode > 0:
        # Pure decode batch with paged KV
        seqused_k = torch.full((total_batch,), seqlen_k_decode, dtype=torch.int32, device=device)
        block_table_combined = block_table_decode
    else:
        # Empty batch (shouldn't happen, but handle gracefully)
        seqused_k = torch.empty(0, dtype=torch.int32, device=device)
        block_table_combined = None
    
    # Calculate max_seqlen_q and max_seqlen_k from cu_seqlens (actual maximum sequence lengths)
    # max_seqlen_q = max difference between consecutive cu_seqlens_q values
    if len(cu_seqlens_q_combined) > 1:
        seqlen_q_actual = cu_seqlens_q_combined[1:] - cu_seqlens_q_combined[:-1]
        max_seqlen_q_combined = seqlen_q_actual.max().item()
    else:
        max_seqlen_q_combined = max_seqlen_q_combined_estimate
    
    # For paged KV, max_seqlen_k is determined from seqused_k
    if total_batch > 0:
        max_seqlen_k_combined = seqused_k.max().item()
    else:
        max_seqlen_k_combined = max_seqlen_k_combined_estimate
    
    print(f"\nCalculated max_seqlen from cu_seqlens:")
    print(f"  max_seqlen_q_combined: {max_seqlen_q_combined} (from actual sequence lengths)")
    print(f"  max_seqlen_k_combined: {max_seqlen_k_combined} (from actual sequence lengths)")
    
    # Set causal=True for both decode and prefill
    # Note: For decode with seqlen_q=1, Flash Attention may force causal=False internally
    # (see mha_fwd_kvcache.cpp line 333: if (seqlen_q == 1 && !alibi_slopes) { is_causal = false; })
    # But we'll set it to True here to match vLLM behavior for mixed batches
    # (see vllm/v1/worker/gpu_model_runner.py line 1055: causal=True)
    causal_combined = True
    print(f"Combined batch: {total_batch} sequences total")
    print(f"  - Prefill: {batch_prefill} sequences, {seqlen_prefill} tokens each")
    print(f"  - Decode: {batch_decode} sequences, {seqlen_q_decode} query tokens, {seqlen_k_decode} context tokens each")
    print(f"Q shape: {q_combined.shape} (vLLM format: [num_tokens, num_heads, head_size])")
    print(f"KV cache shape: {kv_cache_combined.shape} (vLLM format: [2, num_blocks, block_size, num_kv_heads, head_size])")
    print(f"  key_cache shape: {key_cache.shape}, value_cache shape: {value_cache.shape}")
    if k_new_combined is not None:
        print(f"New K/V tokens shape: {k_new_combined.shape} (vLLM format: [num_tokens, num_kv_heads, head_size])")
    else:
        print(f"New K/V tokens: None (decode - tokens already in cache)")
    print(f"nheads-q: {nheads_q}, nheads-kv: {nheads_kv}, "
          f"q_per_kv: {nheads_q // nheads_kv}, "
          f"causal={causal_combined} (vLLM uses causal=True for both prefill and decode)")
    print(f"Paged KV: ENABLED for all batches (page_size={args.page_size})")
    print(f"  This will preserve is_causal=true for decode with headdim=128 → kBlockN=128")
    
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
    print(f"  - seqlen_q_decode (per sequence): {seqlen_q_decode}, max_seqlen_q_combined: {max_seqlen_q_combined}")
    print(f"  - qhead_per_khead: {qhead_per_khead}")
    print(f"  - use_one_mma_wg condition: max_seqlen_q * qhead_per_khead <= 64 (uses max_seqlen_q for varlen!)")
    print(f"    Decode: {max_seqlen_q_combined} * {qhead_per_khead} = {max_seqlen_q_combined * qhead_per_khead} {'<= 64 ✓' if max_seqlen_q_combined * qhead_per_khead <= 64 else '> 64 ✗'}")
    if headdim == 128 or headdim == 64:
        print(f"  - headdim check: {headdim} is 128 or 64 ✓")
    else:
        print(f"  - headdim check: {headdim} is NOT 128 or 64 ✗")
    
    if (headdim == 128 or headdim == 64) and (headdim == actual_v_headdim) and (max_seqlen_q_combined * qhead_per_khead <= 64):
        print(f"\n✓ All conditions met for first kernel (kBlockM=64, kHeadDimV=128)")
    else:
        print(f"\n✗ Conditions NOT met - will use different kernel")
        if headdim != actual_v_headdim:
            print(f"  - Fix: Ensure V cache has headdim={headdim}, not {actual_v_headdim}")
    
    # Calculate combined FLOPS (for summary even if benchmark fails)
    # FLOPs scale with number of query heads; for GQA we still use nheads-q here
    flops_prefill = flops(batch_prefill, seqlen_prefill, seqlen_prefill, headdim, nheads_q, causal_combined, mode="fwd")
    flops_decode = flops(batch_decode, seqlen_q_decode, seqlen_k_decode, headdim, nheads_q, causal_combined, mode="fwd")
    total_flops = flops_prefill + flops_decode
    
    # Create output tensor
    # flash_attn_varlen_func expects [num_tokens, num_heads, headdim] format
    # vLLM flattens it to [num_tokens, num_heads * head_size] after the call
    num_tokens_total = q_combined.shape[0]
    out_combined = torch.empty(num_tokens_total, nheads_q, headdim, device=device, dtype=dtype)
    
    # Create cache_seqlens (seqused_k) for scheduler metadata
    # This represents the actual sequence length for each batch item
    cache_seqlens = seqused_k
    
    # Create scheduler_metadata with page_size if specified
    scheduler_metadata = None
    if args.page_size is not None:
        print(f"Using page_size={args.page_size} for scheduler metadata")
        sys.stdout.flush()  # Ensure output is visible
        sys.stderr.flush()
        scheduler_metadata = get_scheduler_metadata(
            batch_size=total_batch,
            max_seqlen_q=max_seqlen_q_combined,
            max_seqlen_k=max_seqlen_k_combined,
            num_heads_q=nheads_q,
            num_heads_kv=nheads_kv,
            headdim=headdim,
            headdim_v=headdim_v,  # Explicitly pass headdim_v to ensure it matches
            cache_seqlens=cache_seqlens,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_q_combined,
            page_size=args.page_size,
            causal=causal_combined,
            window_size=(-1, -1),
            num_splits=1,  # Always set to 1 to match flash_attn_varlen_func
            prefill_sm_percentage=args.prefill_sm_percentage,
            num_prefill_batches=batch_prefill,
        )
        sys.stdout.flush()  # Flush after get_scheduler_metadata to show its output
        sys.stderr.flush()
        print("get_scheduler_metadata completed")
        sys.stdout.flush()
    
    # Final verification before kernel call
    print(f"\n=== FINAL VERIFICATION BEFORE KERNEL CALL ===")
    print(f"Q shape: {q_combined.shape} (vLLM format: [num_tokens, num_heads, head_size])")
    print(f"key_cache shape: {key_cache.shape} (vLLM format: [num_blocks, block_size, num_kv_heads, head_size])")
    print(f"value_cache shape: {value_cache.shape} (vLLM format: [num_blocks, block_size, num_kv_heads, head_size])")
    if k_new_combined is not None:
        print(f"New K shape: {k_new_combined.shape}, New V shape: {v_new_combined.shape}")
    else:
        print(f"New K/V: None (decode - tokens already in cache)")
    print(f"Output shape: {out_combined.shape} (will be flattened to [num_tokens, num_heads * head_size] like vLLM)")
    print(f"Expected V cache headdim: {headdim_v}")
    if value_cache.shape[-1] != headdim_v:
        print(f"ERROR: V cache has wrong headdim! Expected {headdim_v}, got {value_cache.shape[-1]}")
        print(f"This will cause kHeadDimV={value_cache.shape[-1]} instead of {headdim_v}")
        print(f"Kernel selection is based on v_cache.size(-1), so this must match!")
        raise ValueError(f"V cache headdim mismatch: expected {headdim_v}, got {value_cache.shape[-1]}")
    print(f"✓ V cache headdim is correct: {value_cache.shape[-1]}")
    
    # CRITICAL: Check max_seqlen_q for use_one_mma_wg
    print(f"\nCRITICAL: params.seqlen_q = max_seqlen_q (NOT per-sequence length!)")
    print(f"  max_seqlen_q_combined: {max_seqlen_q_combined}")
    print(f"  qhead_per_khead: {qhead_per_khead}")
    print(f"  Condition for use_one_mma_wg: max_seqlen_q * qhead_per_khead <= 64")
    print(f"  Current: {max_seqlen_q_combined} * {qhead_per_khead} = {max_seqlen_q_combined * qhead_per_khead}")
    if max_seqlen_q_combined * qhead_per_khead > 64:
        print(f"  ✗ use_one_mma_wg will be DISABLED → will use kBlockM=128 instead of 64")
        print(f"  To fix: Set max_seqlen_q <= {64 // qhead_per_khead}")
        print(f"    (Even though each decode sequence has seqlen_q=1, params.seqlen_q uses max_seqlen_q!)")
    else:
        print(f"  ✓ use_one_mma_wg will be ENABLED → will use kBlockM=64")
    
    # Print headdim info that affects kernel selection
    print(f"\nKernel Selection Parameters:")
    print(f"  headdim (Q/K): {headdim}")
    print(f"  headdim_v (V): {headdim_v}")
    print(f"  Note: Flash Attention kernel selection depends on headdim_v")
    print(f"  Supported headdims: [32, 64, 96, 128, 160, 192, 224, 256]")
    if headdim not in [32, 64, 96, 128, 160, 192, 224, 256]:
        print(f"  ⚠ WARNING: headdim={headdim} is NOT in supported list - may select unexpected kernel!")
    if headdim_v not in [32, 64, 96, 128, 160, 192, 224, 256]:
        print(f"  ⚠ WARNING: headdim_v={headdim_v} is NOT in supported list - may select unexpected kernel!")
    print(f"  Page size (KV cache block size): {page_size}")
    print(f"  Flash Attention tile sizes (kBlockN/kBlockM) are determined by kernel selection")
    print(f"    - kBlockN is typically 128 for decode")
    print(f"    - kBlockM is 64 if use_one_mma_wg, else 128")
    print("=" * 80)
    
    # Simulate vLLM's reshape_and_cache_flash if requested
    if args.simulate_reshape_cache and reshape_and_cache_flash_available:
        print(f"\n=== SIMULATING vLLM: Calling reshape_and_cache_flash before attention ===")
        
        # For decode, compute slot_mapping for new tokens
        # New tokens go to positions starting at seqlen_k_decode (the next position after cached tokens)
        num_new_tokens = total_batch  # For decode: 1 token per sequence
        slot_mapping_list = []
        
        for seq_idx in range(total_batch):
            # Current sequence length (before adding new token)
            current_seq_len = seqused_k[seq_idx].item()  # e.g., 1024 for decode
            # New token position (where it will be written in cache)
            new_token_pos = current_seq_len  # e.g., 1024
            
            # Compute slot_mapping: slot = block_number * block_size + offset_in_block
            block_idx = new_token_pos // page_size  # e.g., 1024 // 16 = 64
            block_offset = new_token_pos % page_size  # e.g., 1024 % 16 = 0
            
            # Get block number from block_table
            # Now that we allocate an extra block, block_idx should be within bounds
            max_blocks = block_table_combined.shape[1]
            
            if block_idx < max_blocks:
                # Block already allocated in block_table (including the extra block for new token)
                block_number = block_table_combined[seq_idx, block_idx].item()
                slot = block_number * page_size + block_offset
                slot_mapping_list.append(slot)
            else:
                # This shouldn't happen now that we allocate extra blocks, but handle gracefully
                print(f"Warning: block_idx {block_idx} >= max_blocks {max_blocks} for seq {seq_idx}")
                print(f"  current_seq_len={current_seq_len}, new_token_pos={new_token_pos}, page_size={page_size}")
                slot_mapping_list.append(-1)  # PAD_SLOT_ID
        
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64, device=device)
        print(f"Computed slot_mapping for {num_new_tokens} new tokens")
        print(f"  slot_mapping shape: {slot_mapping.shape}, dtype: {slot_mapping.dtype}")
        print(f"  slot_mapping values: {slot_mapping.cpu().numpy()}")
        
        # Create dummy K/V tensors for new tokens (or use actual ones if available)
        # Shape: [num_new_tokens, num_kv_heads, headdim]
        if k_new_combined is not None and v_new_combined is not None:
            # Use actual new K/V tokens (for prefill case)
            k_new_for_cache = k_new_combined
            v_new_for_cache = v_new_combined
            print(f"Using actual new K/V tokens: shape {k_new_for_cache.shape}")
        else:
            # Create dummy K/V for decode (simulating new tokens being written to cache)
            # Generate random data on CPU then move to GPU (faster than GPU random generation)
            k_new_for_cache = torch.randn(num_new_tokens, nheads_kv, headdim, dtype=dtype).to(device)
            v_new_for_cache = torch.randn(num_new_tokens, nheads_kv, headdim_v, dtype=dtype).to(device)
            print(f"Created dummy K/V tensors for cache write: shape {k_new_for_cache.shape}")
        
        # Create dummy scale tensors (if FP8, otherwise not used)
        kv_cache_dtype = "auto"  # or "fp8" if using FP8
        k_scale = torch.ones(1, dtype=torch.float32, device=device)
        v_scale = torch.ones(1, dtype=torch.float32, device=device)
        
        # Call reshape_and_cache_flash to simulate vLLM's cache write
        print(f"Calling reshape_and_cache_flash...")
        try:
            if reshape_and_cache_flash_func is not None:
                reshape_and_cache_flash_func(
                    k_new_for_cache,
                    v_new_for_cache,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    kv_cache_dtype,
                    k_scale,
                    v_scale,
                )
                print(f"✓ reshape_and_cache_flash completed")
                torch.cuda.synchronize()  # Ensure cache write completes before attention
            else:
                print("Error: reshape_and_cache_flash_func is None")
        except Exception as e:
            print(f"Error calling reshape_and_cache_flash: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without reshape_and_cache_flash...")
        print("=" * 80)
    
    # Benchmark combined batch
    try:
        print(f"Running benchmark with {repeats} repeats...")
        # Prepare function arguments - vLLM format with KV cache
        # flash_attn_varlen_func takes k and v directly; when using block_table, these are the cache
        # For prefill with new tokens, vLLM writes them to cache first using reshape_and_cache_flash
        # For benchmarking, we'll pass the cache directly as k and v (matching vLLM's usage)
        # Print all parameters being passed to flash_attn_varlen_func for comparison
        print(f"\n=== PARAMETERS PASSED TO flash_attn_varlen_func ===")
        print(f"q shape: {q_combined.shape}, dtype: {q_combined.dtype}")
        print(f"k shape: {key_cache.shape}, dtype: {key_cache.dtype}")
        print(f"v shape: {value_cache.shape}, dtype: {value_cache.dtype}")
        print(f"out shape: {out_combined.shape}, dtype: {out_combined.dtype}")
        print(f"max_seqlen_q: {max_seqlen_q_combined}")
        print(f"max_seqlen_k: {max_seqlen_k_combined}")
        print(f"cu_seqlens_q: {cu_seqlens_q_combined.cpu().numpy()}")
        print(f"seqused_k: {seqused_k.cpu().numpy()}")
        print(f"dropout_p: {dropout_p}")
        print(f"causal: {causal_combined}")
        print(f"scheduler_metadata: {scheduler_metadata}")
        if scheduler_metadata is not None:
            print(f"  scheduler_metadata shape: {scheduler_metadata.shape}, dtype: {scheduler_metadata.dtype}")
            print(f"  scheduler_metadata values: {scheduler_metadata.cpu().numpy()}")
        print(f"block_table shape: {block_table_combined.shape if block_table_combined is not None else None}")
        if block_table_combined is not None:
            print(f"  block_table first row: {block_table_combined[0].cpu().numpy()}")
        print(f"fa_version: {fa_version}")
        print(f"num_splits: 1 (always set to 1)")
        print(f"q_descale: None (not set in script)")
        print(f"k_descale: None (not set in script)")
        print(f"v_descale: None (not set in script)")
        print(f"s_aux: None (not set in script)")
        print("=" * 80)
        
        func_kwargs = {
            'q': q_combined,
            'k': key_cache,  # vLLM format: [num_blocks, block_size, num_kv_heads, head_size]
            'v': value_cache,  # vLLM format: [num_blocks, block_size, num_kv_heads, head_size]
            'out': out_combined,
            'max_seqlen_q': max_seqlen_q_combined,
            'cu_seqlens_q': cu_seqlens_q_combined,
            'max_seqlen_k': max_seqlen_k_combined,
            'seqused_k': seqused_k,
            'block_table': block_table_combined,
            'dropout_p': dropout_p,
            'causal': causal_combined,
            'scheduler_metadata': scheduler_metadata,
            'fa_version': fa_version,
            'num_splits': 1,  # Always set to 1
            'prefill_sm_percentage': args.prefill_sm_percentage,
            'num_prefill_batches': batch_prefill,
            'tile_scheduler_debug': args.tile_scheduler_debug,
            'repeats': repeats,
            'verbose': True  # Set to True to see Timer output showing all repeats
        }
        
        # Always use block_table and seqused_k for paged KV (all cases)
        func_kwargs['block_table'] = block_table_combined
        func_kwargs['seqused_k'] = seqused_k
        # Note: cu_seqlens_k is not needed when using block_table
        func_kwargs['cu_seqlens_k'] = None
        
        print(f"\nUsing vLLM format with paged KV cache:")
        print(f"  block_table shape={block_table_combined.shape}, seqused_k shape={seqused_k.shape}")
        print(f"  k (key_cache) shape={key_cache.shape}, v (value_cache) shape={value_cache.shape}")
        print(f"  Note: For prefill, new tokens would be written to cache first (like vLLM's reshape_and_cache_flash)")
        print(f"  Paged KV enabled for all batches - is_causal will be preserved for decode with headdim=128")
        
        f_combined = time_forward(
            flash_attn_varlen_func,
            flush_cache=args.flush_cache,
            warmup=not args.no_warmup,
            **func_kwargs
        )
        
        # vLLM flattens output to [num_tokens, num_heads * head_size]
        # The function returns [num_tokens, num_heads, headdim], so we flatten it here
        out_combined_flat = out_combined.view(num_tokens_total, nheads_q * headdim)
        
        speed_combined = efficiency(total_flops, f_combined)
        print(f"\nTime: {f_combined*1000:.3f} ms")
        print(f"Total FLOPS: {total_flops/1e12:.2f} TFLOPS")
        print(f"  - Prefill: {flops_prefill/1e12:.2f} TFLOPS")
        print(f"  - Decode: {flops_decode/1e12:.2f} TFLOPS")
        print(f"Throughput: {speed_combined:.2f} TFLOPs/s")
    except Exception as e:
        print(f"Error benchmarking combined batch: {e}")
        import traceback
        traceback.print_exc()
        f_combined = float('nan')
        speed_combined = 0.0
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Batch':<20} {'Time (ms)':<15} {'TFLOPs':<15} {'TFLOPs/s':<15}")
    print("-" * 80)
    if not math.isnan(f_combined):
        print(f"{'Combined (Prefill+Decode)':<20} {f_combined*1000:<15.3f} {total_flops/1e12:<15.2f} {speed_combined:<15.2f}")
    print("=" * 80)


if __name__ == '__main__':
    main()

