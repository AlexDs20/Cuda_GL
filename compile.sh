#!/bin/bash

rm -r generated build/bin/linux/OpenGL_Cuda
./premake/premake5 gmake2

pushd generated
    bear -- make config=debug
    cp compile_commands.json ..
popd
./build/bin/linux/OpenGL_Cuda/debug/OpenGL_Cuda
