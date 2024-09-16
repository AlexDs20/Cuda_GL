#pragma once
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace Render {

int setup_opengl(const int major, const int minor);
int setup_glad();
GLFWwindow* create_window(const unsigned int width, const unsigned int height, const char* title);

void create_shader_program(GLuint* shaderProgram);
void create_quad(GLuint* VAO);
void create_texture_2D(float* data, unsigned int width, unsigned int height, GLuint* texture);
void draw_quad(GLuint shaderProgram, GLuint VAO, GLuint texture);
}
