#include <glad/glad.h>
#include <GLFW/glfw3.h>
// #include <cuda_runtime.h>
// #include <gl_interop.h>

#include<stdio.h>
#include "renderer/render.h"

int main() {
    Render::setup_opengl(3, 3);
    GLFWwindow* window = Render::create_window(1024, 768, "Cuda_OpenGL_Interop");
    Render::setup_glad();

    // GLuint shaderProgram;
    // Render::create_shader_program(&shaderProgram);

    // GLuint quad_vao;
    // Render::create_quad(&quad_vao);

    // int width = 1024;
    // int height = 768;
    // float* n = nullptr;
    // GLuint texture;
    // Render::create_texture_2D(n, width, height, &texture);

    glfwSwapInterval(1);
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.3, 0.5, 0.7, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    // free(n);
}

void first_method() {
    printf("HERE");
    // // Initialize
    // glad_init();
    // cudaFree(0);

    // // Create Resources
    // // opengl texture
    // GLuint texture;
    // glGenTextures(1, &texture);
    // glBindTexture(GL_TEXTURE_2D, texture);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    // glBindTexture(GL_TEXTURE_2D, 0);
    // // Register texture with CUDA
    // cudaGraphicsResource* cudaResource;
    // cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadWrite);

    // // Cuda Operations
    // // map the opengl texture to cuda
    // cudaArray* cuArray;
    // cudaGraphicsMapResources(1, &cudaResource, 0);
    // cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
    // // Launch CUDA kernel
    // // kenel<<<gridSize,blockSize>>>(cuArray);
    // // unmap the openGL texture from cuda
    // cudaGraphicsUnmapResources(1, &cudaResource, 0);

    // // OpenGL operations
    // // Use the texture in OpenGL Rendering
    // glBindTexture(GL_TEXTURE_2D, texture);
    // // Render using OpenGL
    // // glDraw();

    // // Cleanup
    // // Unregister the openGL resource from CUDA
    // cudaGraphicsUnregisterResource(cudaResource);
    // // Destroy opengl texture
    // glDeleteTextures(1, &texture);
}
