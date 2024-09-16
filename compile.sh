#!/bin/bash

rm -r generated build/bin/linux/OpenGL_Cuda
./premake/premake5 gmake2

pushd generated
    make config=debug
    bear -- make config=debug
popd
./build/bin/linux/OpenGL_Cuda/debug/OpenGL_Cuda
