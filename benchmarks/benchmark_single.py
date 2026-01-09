"""
Single benchmark run for specific prefill/decode scenario.
"""
import torch
from flash_attn.utils.benchmark import benchmark_forward
from vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
import math

def flops(batch, seqlen_q, seqlen_k, headdim, nheads, causal):
    """Calculate FLOPS for attention."""
    # QK^T: batch * seqlen_q * seqlen_k * nheads * headdim
    # Softmax: batch * seqlen_q * seqlen_k * nheads
    # Attention * V: batch * seqlen_q * seqlen_k * nheads * headdim
    # Total: ~4 * batch * seqlen_q * seqlen_k * nheads * headdim
    # For causal, divide by 2
    f = 4 * batch * seqlen_q * seqlen_k * nheads * headdim // (2 if causal else 1)
    return f

def efficiency(flop, time):
    """Convert to TFLOPs/s."""
    return (flop / time / 10**12) if not math.isnan(time) and time > 0 else 0.0

device = 'cuda'
dtype = torch.bfloat16
fa_version = 3
headdim = 128
nheads = 16  # Assuming 2048 total dim / 128 headdim
repeats = 1

print("=" * 80)
print("Single Benchmark: Prefill + Decode")
print("=" * 80)
print(f"Device: {device}, Dtype: {dtype}, FA Version: {fa_version}")
print(f"Headdim: {headdim}, Nheads: {nheads}")
print("=" * 80)

# ========== Prefill: 16 sequences, each 1024 tokens ==========
print("\n### Prefill Phase ###")
batch_prefill = 16
seqlen_prefill = 1024
total_q_prefill = batch_prefill * seqlen_prefill

# Create QKV for prefill
q_prefill = torch.randn(total_q_prefill, nheads, headdim, device=device, dtype=dtype)
k_prefill = torch.randn(total_q_prefill, nheads, headdim, device=device, dtype=dtype)
v_prefill = torch.randn(total_q_prefill, nheads, headdim, device=device, dtype=dtype)

# Cumulative sequence lengths for prefill
cu_seqlens_q_prefill = torch.arange(0, (batch_prefill + 1) * seqlen_prefill, 
                                   step=seqlen_prefill, dtype=torch.int32, device=device)
cu_seqlens_k_prefill = cu_seqlens_q_prefill.clone()

print(f"Prefill: {batch_prefill} sequences, {seqlen_prefill} tokens each")
print(f"Q shape: {q_prefill.shape}, K shape: {k_prefill.shape}, V shape: {v_prefill.shape}")

# Benchmark prefill
time_f_prefill = benchmark_forward(
    flash_attn_varlen_func,
    q_prefill, k_prefill, v_prefill,
    max_seqlen_q=seqlen_prefill,
    cu_seqlens_q=cu_seqlens_q_prefill,
    max_seqlen_k=seqlen_prefill,
    cu_seqlens_k=cu_seqlens_k_prefill,
    causal=True,
    fa_version=fa_version,
    repeats=repeats,
    verbose=False
)

time_prefill = time_f_prefill[1].mean
flops_prefill = flops(batch_prefill, seqlen_prefill, seqlen_prefill, headdim, nheads, causal=True)
tflops_prefill = efficiency(flops_prefill, time_prefill)

print(f"Time: {time_prefill*1000:.3f} ms")
print(f"FLOPS: {flops_prefill/1e12:.2f} TFLOPS")
print(f"Throughput: {tflops_prefill:.2f} TFLOPs/s")

# ========== Decode: 512 sequences, 1 query token, 1024 context ==========
print("\n### Decode Phase ###")
batch_decode = 512
seqlen_q_decode = 1  # 1 new token per sequence
seqlen_k_decode = 1024  # 1024 context tokens
total_q_decode = batch_decode * seqlen_q_decode
total_k_decode = batch_decode * seqlen_k_decode

# Create Q, K, V for decode
q_decode = torch.randn(total_q_decode, nheads, headdim, device=device, dtype=dtype)
k_decode = torch.randn(total_k_decode, nheads, headdim, device=device, dtype=dtype)
v_decode = torch.randn(total_k_decode, nheads, headdim, device=device, dtype=dtype)

# Cumulative sequence lengths for decode
cu_seqlens_q_decode = torch.arange(0, (batch_decode + 1) * seqlen_q_decode,
                                  step=seqlen_q_decode, dtype=torch.int32, device=device)
cu_seqlens_k_decode = torch.arange(0, (batch_decode + 1) * seqlen_k_decode,
                                   step=seqlen_k_decode, dtype=torch.int32, device=device)

print(f"Decode: {batch_decode} sequences, {seqlen_q_decode} query token, {seqlen_k_decode} context tokens each")
print(f"Q shape: {q_decode.shape}, K shape: {k_decode.shape}, V shape: {v_decode.shape}")

# Benchmark decode
time_f_decode = benchmark_forward(
    flash_attn_varlen_func,
    q_decode, k_decode, v_decode,
    max_seqlen_q=seqlen_q_decode,
    cu_seqlens_q=cu_seqlens_q_decode,
    max_seqlen_k=seqlen_k_decode,
    cu_seqlens_k=cu_seqlens_k_decode,
    causal=True,
    fa_version=fa_version,
    repeats=repeats,
    verbose=False
)

time_decode = time_f_decode[1].mean
flops_decode = flops(batch_decode, seqlen_q_decode, seqlen_k_decode, headdim, nheads, causal=True)
tflops_decode = efficiency(flops_decode, time_decode)

print(f"Time: {time_decode*1000:.3f} ms")
print(f"FLOPS: {flops_decode/1e12:.2f} TFLOPS")
print(f"Throughput: {tflops_decode:.2f} TFLOPs/s")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"{'Phase':<15} {'Time (ms)':<15} {'TFLOPs':<15} {'TFLOPs/s':<15}")
print("-" * 80)
print(f"{'Prefill':<15} {time_prefill*1000:<15.3f} {flops_prefill/1e12:<15.2f} {tflops_prefill:<15.2f}")
print(f"{'Decode':<15} {time_decode*1000:<15.3f} {flops_decode/1e12:<15.2f} {tflops_decode:<15.2f}")
print("=" * 80)

