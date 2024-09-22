#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>

#include "renderer/render.h"

std::string read_file(const char* filepath) {
    std::string file_content;

    try
    {
      // Open file
      std::ifstream input_stream(filepath, std::ios::in | std::ios::binary);

      // ensure ifstream can throw exceptions
      input_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);

      // read file's buffer contents into streams
      std::stringstream string_stream;
      string_stream << input_stream.rdbuf();

      // close file handlers
      input_stream.close();

      file_content = string_stream.str();
    }
    catch(std::ifstream::failure& e)
    {
      std::cerr << "ERROR: " << filepath << " :: " << e.what() << std::endl;
      return "";
    }

    return file_content;
}


namespace Render {

int setup_opengl(const int major, const int minor) {
    if (!glfwInit()) {
        printf("Failed to initialize GLFW!\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __MAC
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    return 0;
}

int setup_glad(){
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("Failed to initialize GLAD\n");
        return 1;
    }
    return 0;
}

void framebuffer_size_callback(GLFWwindow* , int width, int height) {
    glViewport(0, 0, width, height);
}

GLFWwindow* create_window(const unsigned int width, const unsigned int height, const char* title){
    GLFWwindow* window = glfwCreateWindow(width, height, title, NULL, NULL);

    if (window == NULL) {
        printf("Failed to create GLFW window\n");
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    return window;
}

void create_shader_program(GLuint* shaderProgram, const char* vertex_shader, const char* fragment_shader) {
    std::string  vertex_shader_string = read_file(vertex_shader);
    const char* vertexShaderSource = vertex_shader_string.c_str();

    std::string fragment_shader_string = read_file(fragment_shader);
    const char *fragmentShaderSource = fragment_shader_string.c_str();

    GLuint vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n %s\n", infoLog);
    }

    GLuint fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n %s\n", infoLog);
    }

    *shaderProgram = glCreateProgram();
    glAttachShader(*shaderProgram, vertexShader);
    glAttachShader(*shaderProgram, fragmentShader);
    glLinkProgram(*shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void create_quad(GLuint* VAO) {
    const float quadVertices[] = {
      // X     Y
       -0.5f,-0.5f,
        0.5f,-0.5f,
        0.5f, 0.5f,

        0.5f, 0.5f,
       -0.5f, 0.5f,
       -0.5f,-0.5f,

      // textCoord
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,

        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
    };

    GLuint VBO;
    glGenVertexArrays(1, VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(*VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)(12*sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void create_texture_2D(float* data, unsigned int width, unsigned int height, GLuint* texture) {
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, (void*)data);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_FLOAT, (void*)data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void create_pbo(float* data, size_t size, GLuint* pbo) {
    glGenBuffers(1, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, (void*) data, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void draw_quad(GLuint shaderProgram, GLuint VAO, GLuint texture) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        glUseProgram(shaderProgram);
        glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
}

}
