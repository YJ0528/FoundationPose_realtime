#!/bin/bash
DIR=$(pwd)

# Find Python executable
PYTHON_EXEC=$(which python3)
if [ -z "$PYTHON_EXEC" ]; then
    PYTHON_EXEC=$(which python)
fi

if [ -z "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found"
    exit 1
fi

echo "Using Python: $PYTHON_EXEC"

# Get Python paths correctly
PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIBRARY_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

echo "Python include: $PYTHON_INCLUDE_DIR"
echo "Python library dir: $PYTHON_LIBRARY_DIR"

# Build mycpp (skip if it fails, focus on CUDA extension)
cd $DIR/mycpp/ && mkdir -p build && cd build
echo "Attempting CMake build..."
cmake .. \
    -DPYTHON_EXECUTABLE=$PYTHON_EXEC \
    -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
    -DPYTHON_LIBRARY_DIR=$PYTHON_LIBRARY_DIR

if [ $? -eq 0 ]; then
    echo "CMake succeeded, building..."
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Make failed, but continuing with CUDA extension..."
    fi
else
    echo "CMake failed, but continuing with CUDA extension..."
fi

# Set environment for CUDA build
export TORCH_CUDA_ARCH_LIST="8.6"  # Adjust based on your GPU
export FORCE_CUDA=1

# Build bundlesdf/mycuda with fixed setup
cd $DIR/bundlesdf/mycuda && rm -rf build *egg*

# Create a patched setup.py that works with newer PyTorch
cat > setup_fixed.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

ext_modules = [
    CUDAExtension(
        name='common',
        sources=[
            'bindings.cpp',
            'common.cu',
        ],
        include_dirs=[
            '/usr/include/eigen3',
            '/usr/local/include/eigen3',
        ],
        extra_compile_args={
            'cxx': ['-std=c++17', '-O3'],
            'nvcc': [
                '-std=c++17',
                '-O3',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '-Xcompiler', '-fPIC',
                # Add flags to suppress warnings
                '-diag-suppress=20012',  # Suppress __host__ __device__ warnings
                '-diag-suppress=2361',   # Suppress narrowing conversion warnings
            ]
        },
        define_macros=[
            ('TORCH_EXTENSION_NAME', 'common'),
            ('TORCH_API_INCLUDE_EXTENSION_H', None),
        ]
    )
]

setup(
    name='common',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
EOF

echo "Attempting to build CUDA extension with fixed setup..."
python setup_fixed.py develop

# If that fails, try with the original but with environment variables
if [ $? -ne 0 ]; then
    echo "Fixed setup failed, trying with bypass flags..."
    export TORCH_CUDA_VERSION_CHECK=0
    pip install -e . -v
fi

# Clean up
rm -f setup_fixed.py

cd ${DIR}
echo "Build script completed."
