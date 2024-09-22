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

void print_gpu_prop();
void update_texture();
void update_vbo();

int main() {
    print_gpu_prop();
    update_texture();
    update_vbo();
}

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

void update_texture(){
    // --------------------------
    // OpenGL setup
    Render::setup_opengl(3, 3);
    GLFWwindow* window = Render::create_window(1024, 768, "Cuda_OpenGL_Interop");
    Render::setup_glad();

    // Check which GPU is used
    const GLubyte* vendor = glGetString(GL_VENDOR); // Returns the vendor
    const GLubyte* renderer = glGetString(GL_RENDERER);
    printf("GPU used by OpenGL: \n\t Vendor: %s, Renderer:%s\n", vendor, renderer);

    GLuint shaderProgram;
    Render::create_shader_program(&shaderProgram, "src/shaders/quad_texture.vert.glsl", "src/shaders/quad_texture.frag.glsl");

    GLuint quad_vao;
    Render::create_quad(&quad_vao);

    int width = 2;
    int height = 2;

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr); // Initialize texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register the texture with CUDA
    struct cudaGraphicsResource* cudaResource;
    errCheck(cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    // Create associated data to be able to use the texture in CUDA
    cudaArray* cuArray;
    unsigned char* d_textureData;
    errCheck(cudaMalloc(&d_textureData, width*height*sizeof(unsigned char)*4));
    errCheck(cudaDeviceSynchronize());
    dim3 blockDim(16, 16, 1);
    dim3 gridDim( (width+blockDim.x-1)/blockDim.x, (height+blockDim.y-1)/blockDim.y, 1 );

    glfwSwapInterval(1);
    int i = 0;
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.3, 0.5, 0.7, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        // Update Texture -> Map, get pointer to data
        errCheck(cudaGraphicsMapResources(1, &cudaResource, 0));
        errCheck(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0));            // Get a cudaArray to actually be able to access texture data
        kernel_update_texture<<<gridDim, blockDim>>>(d_textureData, width, height, i++);
        errCheck(cudaMemcpy2DToArray(cuArray, 0, 0, d_textureData, width*sizeof(unsigned char)*4, width*sizeof(unsigned char)*4, height, cudaMemcpyDefault));
        errCheck(cudaGraphicsUnmapResources(1, &cudaResource, 0));
        errCheck(cudaDeviceSynchronize());

        Render::draw_quad(shaderProgram, quad_vao, texture);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    errCheck(cudaFree(d_textureData));
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
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

void update_vbo() {
    // --------------------------
    // OpenGL setup
    Render::setup_opengl(3, 3);
    GLFWwindow* window = Render::create_window(1024, 768, "Cuda_OpenGL_Interop");
    Render::setup_glad();

    // Check which GPU is used
    const GLubyte* vendor = glGetString(GL_VENDOR); // Returns the vendor
    const GLubyte* renderer = glGetString(GL_RENDERER);
    printf("GPU used by OpenGL: %s, %s\n", vendor, renderer);

    GLuint shaderProgram;
    Render::create_shader_program(&shaderProgram, "src/shaders/quad.vert.glsl", "src/shaders/quad.frag.glsl");

    // Create VAO
    const float quadVertices[] = {          // NOT used as we write over using cuda
      // X     Y
       -0.5f,-0.5f,
        0.5f,-0.5f,
       -0.5f, 0.5f,
        0.5f, 0.5f,
    };

    GLuint VAO;
    GLuint VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    GLuint EBO;
    unsigned int indices[6] = {
        0, 1, 2,
        1, 2, 3
    };
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,  6* sizeof(unsigned int), (void*)indices, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Register the VAO with CUDA
    struct cudaGraphicsResource* cudaResource;
    errCheck(cudaGraphicsGLRegisterBuffer(&cudaResource, VAO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Setup cuda variables
    float* d_vboData;        // Will contain the device address where the data are
    size_t d_size;           // Will get size of the data
    int width = 2;
    int height = 2;
    dim3 blockDim(16, 16, 1);
    dim3 gridDim( (width+blockDim.x-1)/blockDim.x, (height+blockDim.y-1)/blockDim.y, 1 );
    
    glfwSwapInterval(1);
    float t = 0;
    float dt = 1.0f/60.0f;

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.3, 0.5, 0.7, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        // Map VAO resource to cuda
        errCheck(cudaGraphicsMapResources(1, &cudaResource, 0));
        errCheck(cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &d_size, cudaResource));
        kernel_update_vertices<<<gridDim, blockDim>>>(d_vboData, width, height, t);
        errCheck(cudaGraphicsUnmapResources(1, &cudaResource, 0));
        errCheck(cudaDeviceSynchronize());

        // Draw shape
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        t += dt;
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

void print_gpu_prop() {
    int nDevices;
    errCheck(cudaGetDeviceCount(&nDevices));
    nDevices = 1;
    printf("Number of devices: %d\n", nDevices);
    for (int device=0; device<nDevices; device++) {
        cudaDeviceProp deviceProp;
        errCheck(cudaGetDeviceProperties(&deviceProp, device));
        errCheck(cudaDeviceSynchronize());
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
    errCheck(cudaDeviceReset());
}
