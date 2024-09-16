#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// #include <stdio.h>
#include "renderer/render.h"

void first_method();

int main() {
    if (1) {
        first_method();
    } else {
    }
}

__global__ void make_red(cudaArray* array, int n) {
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
    // Register texture with CUDA
    struct cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    // map the opengl texture to cuda
    cudaArray* cuArray;
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
    // dim3 blockDim(1,1,1);
    // dim3 gridDim(4,1,1);
    // make_red <<< blockDim, gridDim >>> (cuArray, 4);

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

    // Unregister the openGL resource from CUDA
    cudaGraphicsUnregisterResource(cudaResource);
    // Destroy opengl texture
    glDeleteTextures(1, &texture);
    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
}
