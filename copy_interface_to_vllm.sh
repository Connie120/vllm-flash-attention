#!/bin/bash
# Simple script to copy flash_attn_interface.py to vLLM installation after rebuild

SOURCE_FILE="/home/kang222/vllm-flash-attention/vllm_flash_attn/flash_attn_interface.py"
TARGET_PATTERN="$HOME/vllm-12-0-venv/lib/python3.12/site-packages/vllm_flash_attn-*/vllm_flash_attn/flash_attn_interface.py"

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file not found: $SOURCE_FILE"
    exit 1
fi

# Find and copy to all matching vLLM installations
for target in $TARGET_PATTERN; do
    if [ -f "$target" ]; then
        target_dir=$(dirname "$target")
        echo "Copying $SOURCE_FILE to $target_dir/"
        cp "$SOURCE_FILE" "$target_dir/"
        echo "Successfully copied to $target_dir"
    fi
done

