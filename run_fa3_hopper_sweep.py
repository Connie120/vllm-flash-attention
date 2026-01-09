"""
Run FA3 (FlashAttention v3) Hopper sweep with nsys profiling.

This script is specific to FlashAttention FA3 on Hopper (H100/H200) and
profiles prefill, decode, and mixed workloads using benchmark_hopper_forward.py.
"""

from pathlib import Path

from run_hopper_sweep import main as _legacy_main  # type: ignore


def main() -> None:
    """Entry point for the FlashAttention-specific Hopper sweep."""
    _legacy_main()


if __name__ == "__main__":
    main()




