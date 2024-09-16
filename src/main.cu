#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>

#include "renderer/render.h"

#define errCheck(call)                                                      \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);              \
        fprintf(stderr, "code: %d, reason: %s\n",                           \
                error, cudaGetErrorString(error));                          \
    }                                                                       \
}

void first_method();
void second_method();

int main() {
    if (0) {
        first_method();
    } else {
        second_method();
    }
}

void print_gpu_prop() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("Number of devices: %d\n", nDevices);

    for (int device=0; device<nDevices; device++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, device);
        cudaDeviceSynchronize();

        printf("Device name: %s\n", deviceProp.name);
        printf("  Compute capability:                   %d.%d\n",
                    deviceProp.major, deviceProp.minor);
        printf("  Concurrent kernels:                   %s\n",
                    deviceProp.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n",
                    deviceProp.deviceOverlap ? "yes" : "no");
        printf("  Clock Rate (kHz):                     %d\n",
                    deviceProp.clockRate);
        printf("  Memory Clock Rate (MHz):              %d\n",
                    deviceProp.memoryClockRate/1000);
        printf("  Memory Bus Width (bits):              %d\n",
                    deviceProp.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s):         %.1f\n",
                    2.0*deviceProp.memoryClockRate*((float)deviceProp.memoryBusWidth/8)/1.0e6);
        printf("  ---\n");

        printf("  Total global memory (Gbytes)          %.1f\n",
                    (float)(deviceProp.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Total constant memory (Kbytes)        %.1f\n",
                    (float)(deviceProp.totalConstMem)/1024.0);
        printf("  Shared memory per block (Kbytes)      %.1f\n",
                    (float)(deviceProp.sharedMemPerBlock)/1024.0);
        printf("  Shared memory per MP (Kbytes)         %.1f\n",
                    (float)(deviceProp.sharedMemPerMultiprocessor)/1024.0);
        printf("  ---\n");

        printf("  Warp-size:                            %d\n",
                    deviceProp.warpSize);
        printf("  Max threads per block:                %d\n",
                    deviceProp.maxThreadsPerBlock);
        printf("  Max threads dim:                      (%d,%d,%d)\n",
                    deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max grid size:                        (%d,%d,%d)\n",
                    deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  ---\n");

        printf("  Number of multiprocessors:            %d\n",
                    deviceProp.multiProcessorCount);
        printf("  Max blocks per multiprocessors:       %d\n",
                    deviceProp.maxBlocksPerMultiProcessor);
        printf("  Max threads per multiprocessors:      %d\n",
                    deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max regs per multiprocessors:         %d\n",
                    deviceProp.regsPerMultiprocessor);

    }
    cudaDeviceReset();
}

__global__ void empty_kernel() {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    printf("(%d,%d): \n", x, y);
}

void first_method() {
    // --------------------------
    // OpenGL setup
    Render::setup_opengl(3, 3);
    GLFWwindow* window = Render::create_window(1024, 768, "Cuda_OpenGL_Interop");
    Render::setup_glad();

    GLuint shaderProgram;
    Render::create_shader_program(&shaderProgram);

    GLuint quad_vao;
    Render::create_quad(&quad_vao);

    int width = 2;
    int height = 2;
    float texture_array[] = {
        0.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
    };
    GLuint texture;
    Render::create_texture_2D(texture_array, width, height, &texture);

    // --------------------------
    // Setup CUDA
    print_gpu_prop();
    int device = 0;
    cudaSetDevice(device);

    // Register texture with CUDA
    struct cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    // map the opengl texture to cuda
    cudaArray* cuArray;

    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
    dim3 blockDim(1,1,1);
    dim3 gridDim(width,height,1);
    empty_kernel <<< blockDim, gridDim >>> ();
    errCheck(cudaDeviceSynchronize());

    // unmap the openGL texture from cuda
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.3, 0.5, 0.7, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        Render::draw_quad(shaderProgram, quad_vao, texture);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    // Unregister the openGL resource from CUDA
    cudaGraphicsUnregisterResource(cudaResource);
    cudaDeviceReset();
    // Destroy opengl texture
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}

__global__ void
updateTexture(unsigned char* d_textureData, int width, int height, int i) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x<width && y < height) {
        int idx = y * width + x;
        d_textureData[idx * 4 + 0] = (255 + x*i)%255;
        d_textureData[idx * 4 + 1] = (255 + y*i)%255;
        d_textureData[idx * 4 + 2] = (255 + (x+y)*i)%255;
        d_textureData[idx * 4 + 3] = 255;
    }

}

void second_method(){
    // --------------------------
    // OpenGL setup
    Render::setup_opengl(3, 3);
    GLFWwindow* window = Render::create_window(1024, 768, "Cuda_OpenGL_Interop");
    Render::setup_glad();

    GLuint shaderProgram;
    Render::create_shader_program(&shaderProgram);

    GLuint quad_vao;
    Render::create_quad(&quad_vao);

    int width = 5;
    int height = 5;

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr); // Initialize texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Register the texture with CUDA
    cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // Update texture
    cudaArray* cuArray;
    unsigned char* d_textureData;
    cudaMalloc(&d_textureData, width*height*sizeof(unsigned char)*4);
    dim3 blockDim(16, 16, 1);
    dim3 gridDim( (width+blockDim.x-1)/blockDim.x, (height+blockDim.y-1)/blockDim.y, 1 );

    glfwSwapInterval(1);
    int i = 0;
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.3, 0.5, 0.7, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        cudaGraphicsMapResources(1, &cudaResource, 0);
        cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);            // Get a cudaArray to actually be able to access texture data
        updateTexture<<<gridDim, blockDim>>>(d_textureData, width, height, i++);
        cudaMemcpyToArray(cuArray, 0, 0, d_textureData, width*height*sizeof(unsigned char)*4, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        cudaDeviceSynchronize();

        Render::draw_quad(shaderProgram, quad_vao, texture);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_textureData);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
}
