#!/bin/bash
# Quick rebuild script for FA3 extension only
# Usage: ./rebuild_fa3.sh
# This only rebuilds changed files (incremental build)

set -e

# Find the build directory
BUILD_DIR=$(find build -name "CMakeCache.txt" -type f 2>/dev/null | head -1 | xargs dirname)

if [ -z "$BUILD_DIR" ]; then
    echo "Build directory not found. Run 'pip install -e .' first to configure CMake."
    exit 1
fi

echo "Building _vllm_fa3_C target in $BUILD_DIR..."
echo "This will only rebuild changed files (incremental build)"

# Store the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$BUILD_DIR"

# Check if using ninja (faster and better at incremental builds)
if command -v ninja &> /dev/null && [ -f build.ninja ]; then
    echo "Using Ninja build system (better incremental builds)"
    # Ninja will automatically do incremental builds
    # Show what will be built (dry-run)
    echo "Checking what needs rebuilding..."
    ninja -n _vllm_fa3_C 2>&1 | head -5 || true
    echo ""
    echo "Building (ninja will only rebuild changed files)..."
    ninja _vllm_fa3_C -j$(nproc)
else
    # Fall back to CMake build
    echo "Using CMake build system"
    echo "Building (CMake will only rebuild changed files)..."
    cmake --build . --target _vllm_fa3_C -j$(nproc)
fi

echo ""
echo "Build complete! Looking for built library..."

# Find the built library file (search from project root)
BUILT_LIB=$(find "$PROJECT_ROOT/build" -name "_vllm_fa3_C*.so" -type f 2>/dev/null | head -1)

if [ -z "$BUILT_LIB" ]; then
    echo "Error: Could not find built library file."
    echo "Searched in: $PROJECT_ROOT/build/"
    exit 1
fi

echo "Found library: $BUILT_LIB"

# Find the installation directory in venv
INSTALL_DIR="$HOME/vllm-12-0-venv/lib/python3.12/site-packages/vllm_flash_attn-2.7.2.post1+cu128-py3.12-linux-x86_64.egg/vllm_flash_attn"

# Check if install directory exists
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Warning: Install directory not found: $INSTALL_DIR"
    echo "Trying to find vllm_flash_attn in site-packages..."
    INSTALL_DIR=$(python3 -c "import vllm_flash_attn; import os; print(os.path.dirname(vllm_flash_attn.__file__))" 2>/dev/null || echo "")
    if [ -z "$INSTALL_DIR" ] || [ ! -d "$INSTALL_DIR" ]; then
        echo "Error: Could not find vllm_flash_attn installation directory."
        echo "Please ensure the package is installed in your venv."
        exit 1
    fi
fi

echo "Installing to: $INSTALL_DIR"

# Get the library filename
LIB_FILENAME=$(basename "$BUILT_LIB")

# Copy the library
cp -v "$BUILT_LIB" "$INSTALL_DIR/$LIB_FILENAME"

if [ $? -eq 0 ]; then
    echo "✓ Successfully installed $LIB_FILENAME"
else
    echo "✗ Failed to copy library. You may need to run with sudo or check permissions."
    exit 1
fi

# Also copy the Python interface file if it exists locally
PYTHON_INTERFACE="$PROJECT_ROOT/vllm_flash_attn/flash_attn_interface.py"
if [ -f "$PYTHON_INTERFACE" ]; then
    echo "Copying Python interface file..."
    cp -v "$PYTHON_INTERFACE" "$INSTALL_DIR/flash_attn_interface.py"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully updated Python interface"
    else
        echo "⚠ Warning: Failed to copy Python interface file"
    fi
else
    echo "⚠ Warning: Python interface file not found at $PYTHON_INTERFACE"
fi

echo ""
echo "✓ Installation complete!"

