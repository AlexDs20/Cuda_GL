#!/bin/bash

rm -r generated build/bin/linux/OpenGL_Cuda
./premake/premake5 gmake2

pushd generated
    bear -- make -j$(nproc) config=debug
    cp compile_commands.json ..
popd
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia LD_LIBRARY_PATH=./build/bin/linux/GLAD/debug/:./build/bin/linux/GLFW/debug/:./build/bin/linux/GLM/debug/ ./build/bin/linux/OpenGL_Cuda/debug/OpenGL_Cuda
