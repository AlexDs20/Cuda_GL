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


template< typename R, typename T >
__device__ R Clamp( T value, T min, T max )
{
    if ( value < min )
    {
        return (R)min;
    }
    else if ( value > max )
    {
        return (R)max;
    }
    else
    {
        return (R)value;
    }
}

__global__ void
normalize(unsigned char* d_textureData, float* d_dataSource, unsigned int width, unsigned int height) {
    const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i >= width) || (j >= height)) {
        return;
    }

    const unsigned int idx = j * width + i;
    d_dataSource[idx] = (float)(20*(i * j) % 255) / 7;
    d_textureData[idx] = (unsigned char)Clamp<unsigned char>(d_dataSource[idx], 0.0f, 255.0f);
}

void normalize(dim3 gridDim, dim3 blockDim, unsigned char* d_textureData, float* d_dataSource, unsigned int width, unsigned int height) {
    normalize<<<gridDim, blockDim>>>(d_textureData, d_dataSource, width, height);
}
