#!/bin/bash
# Quick rebuild script for files that depend on tile_scheduler.hpp
# Usage: ./rebuild_tile_scheduler.sh
# This rebuilds only files that include tile_scheduler.hpp and relinks the library

set -e

# Find the build directory
BUILD_DIR=$(find build -name "CMakeCache.txt" -type f 2>/dev/null | head -1 | xargs dirname)

if [ -z "$BUILD_DIR" ]; then
    echo "Build directory not found. Run 'pip install -e .' first to configure CMake."
    exit 1
fi

# Store the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Rebuilding files that depend on tile_scheduler.hpp..."
echo "Build directory: $BUILD_DIR"

HEADER_FILE="$PROJECT_ROOT/hopper/tile_scheduler.hpp"
if [ ! -f "$HEADER_FILE" ]; then
    echo "Error: tile_scheduler.hpp not found at $HEADER_FILE"
    exit 1
fi

cd "$BUILD_DIR"

# Check if header is newer than object files - if so, we need to rebuild
# If object files are newer, nothing needs rebuilding
HEADER_TIME=$(stat -c %Y "$HEADER_FILE" 2>/dev/null || stat -f %m "$HEADER_FILE" 2>/dev/null)

# Count how many object files are older than the header (need rebuilding)
echo "Checking which object files need rebuilding..."
NEEDS_REBUILD=0
if command -v ninja &> /dev/null && [ -f build.ninja ]; then
    # Use ninja to check what needs rebuilding (dry-run)
    NINJA_OUTPUT=$(ninja -n _vllm_fa3_C 2>&1)
    if echo "$NINJA_OUTPUT" | grep -q "no work to do"; then
        echo "✓ No files need rebuilding - all object files are up to date!"
        NEEDS_REBUILD=0
    else
        # Count how many files will be rebuilt
        REBUILD_COUNT=$(echo "$NINJA_OUTPUT" | grep -c "\[.*\]" || echo "0")
        if [ "$REBUILD_COUNT" -gt 0 ]; then
            echo "Found $REBUILD_COUNT file(s) that need rebuilding"
            echo "Sample of files to rebuild:"
            echo "$NINJA_OUTPUT" | grep "\[.*\]" | head -5
            NEEDS_REBUILD=1
        else
            echo "✓ No files need rebuilding"
            NEEDS_REBUILD=0
        fi
    fi
else
    # Fallback: check timestamps manually
    echo "Checking object file timestamps..."
    OLD_OBJECTS=$(find CMakeFiles/_vllm_fa3_C.dir/hopper -name "*.o" -newer "$HEADER_FILE" 2>/dev/null | wc -l)
    if [ "$OLD_OBJECTS" -eq 0 ]; then
        echo "⚠ Warning: Could not determine if rebuild is needed. Proceeding anyway..."
        NEEDS_REBUILD=1
    else
        echo "Found object files that are older than header - rebuild needed"
        NEEDS_REBUILD=1
    fi
fi

if [ "$NEEDS_REBUILD" -eq 0 ]; then
    echo ""
    echo "Nothing to rebuild! Library is up to date."
    echo "If you made changes to tile_scheduler.hpp, they may not be reflected."
    echo "You can force a rebuild by touching the header: touch hopper/tile_scheduler.hpp"
    exit 0
fi

echo ""
echo "Rebuilding (CMake/Ninja will rebuild files with outdated object files)..."
if command -v ninja &> /dev/null && [ -f build.ninja ]; then
    echo "Using Ninja build system"
    ninja _vllm_fa3_C -j$(nproc)
else
    echo "Using CMake build system"
    cmake --build . --target _vllm_fa3_C -j$(nproc)
fi

echo ""
echo "Build complete! Looking for built library..."

# Find the built library file
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

echo ""
echo "✓ Installation complete!"
echo ""
echo "Note: If many files were rebuilt, this is expected because tile_scheduler.hpp"
echo "      is included by flash_fwd_launch_template.h, which is included by all"
echo "      instantiation files. Any change to the header requires rebuilding all"
echo "      dependent files (this is correct C++ behavior)."

