#!/usr/bin/env python3
"""
Pre-load FA3 CUDA extension before nvbit starts.
This helps avoid hangs when nvbit intercepts CUDA initialization.

Usage:
    python preload_fa3.py
    # Then in another terminal or script, run nvbit with the benchmark
"""

import sys
import os
from pathlib import Path

# Add paths similar to benchmark script
parent_dir = str(Path(__file__).parent.parent)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

# Try to import FA3 to pre-load the CUDA extension
print("Pre-loading FA3 CUDA extension...")
try:
    from vllm_flash_attn.flash_attn_interface import (
        FA3_AVAILABLE,
        FA3_UNAVAILABLE_REASON,
    )
    if FA3_AVAILABLE:
        print("✓ FA3 CUDA extension loaded successfully")
        print("You can now run nvbit with the benchmark script")
    else:
        print(f"✗ FA3 not available: {FA3_UNAVAILABLE_REASON}")
        sys.exit(1)
except ImportError as e:
    print(f"✗ Failed to import FA3: {e}")
    sys.exit(1)




