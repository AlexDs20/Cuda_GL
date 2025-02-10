#pragma once
#include <cuda_runtime.h>

void kernel_update_texture_wrapper(dim3 gridDim, dim3 blockDim, unsigned char* d_textureData, int width, int height, int i);
void kernel_update_vertices_wrapper(dim3 gridDim, dim3 blockDim, float* vertex, int width, int height, float t);
void normalize(dim3 gridDim, dim3 blockDim, unsigned char* d_textureData, float* d_dataSource, unsigned int width, unsigned int height);
