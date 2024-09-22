#!/bin/bash

CONFIG=debug

if [ -d build/bin/linux/OpenGL_Cuda ]; then
    rm -r build/bin/linux/OpenGL_Cuda
fi
if [ -d generated ]; then
    rm -r generated
fi
./premake/premake5 gmake2

pushd generated
    make -j$(nproc) config=$CONFIG
popd

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=./build/bin/linux/GLAD/$CONFIG/:./build/bin/linux/GLFW/$CONFIG/:./build/bin/linux/GLM/$CONFIG/
./build/bin/linux/OpenGL_Cuda/$CONFIG/OpenGL_Cuda
