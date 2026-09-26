#!/bin/bash

set -e

echo "========== Python environment =========="

echo "Python executable: $(which python)"
echo "Python version:    $(python --version)"

echo "========== Building Mitsuba =========="

rm -rf build
mkdir -p build
cd build

# Configure Mitsuba against the Python interpreter
# from the currently active Conda environment.
cmake -GNinja .. \
    -DPython_EXECUTABLE="$(which python)"

ninja -j4

# Make this locally built Mitsuba + Python bindings available
# in the current shell.
source setpath.sh

echo "========== Testing Python bindings =========="

python -c "import mitsuba as mi; import drjit as dr; print('Mitsuba:', mi.__file__); print('Variants:', mi.variants())"
