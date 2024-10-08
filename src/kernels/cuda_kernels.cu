#include <cuda_runtime.h>

__global__ void
kernel_update_texture(unsigned char* d_textureData, int width, int height, int i) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x<width) & (y<height)) {
        int idx = y * width + x;
        d_textureData[idx * 4 + 0] = (255 + x*i)%255;
        d_textureData[idx * 4 + 1] = (255 + y*i)%255;
        d_textureData[idx * 4 + 2] = (255 + (x+y)*i)%255;
        d_textureData[idx * 4 + 3] = 255;
    }
}

void kernel_update_texture_wrapper(dim3 gridDim, dim3 blockDim, unsigned char* d_textureData, int width, int height, int i) {
    kernel_update_texture<<<gridDim, blockDim>>>(d_textureData, width, height, i++);
}

__global__ void
kernel_update_vertices(float* vertex, int width, int height, float t){
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;

    if ((x<width) & (y<height)) {
        int idx = y * width + x;
        vertex[idx * 2 + 0] = (x-0.5) + 0.2*cosf(t);
        vertex[idx * 2 + 1] = (y-0.5) + 0.2*sinf(t);
    }

}

void kernel_update_vertices_wrapper(dim3 gridDim, dim3 blockDim, float* vertex, int width, int height, float t) {
    kernel_update_vertices<<<gridDim, blockDim>>>(vertex, width, height, t);
}
