#!/bin/bash

rm -r generated build/bin/linux/OpenGL_Cuda
./premake/premake5 gmake2

pushd generated
    bear -- make -j$(nproc) config=debug
    cp compile_commands.json ..
popd
LD_LIBRARY_PATH=./build/bin/linux/GLAD/debug/:./build/bin/linux/GLFW/debug/ ./build/bin/linux/OpenGL_Cuda/debug/OpenGL_Cuda
