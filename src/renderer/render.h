#pragma once
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace Render {

int setup_opengl(const int major, const int minor);
int setup_glad();
GLFWwindow* create_window(const unsigned int width, const unsigned int height, const char* title);

glm::mat4 compute_model_matrix(glm::vec3 position, glm::vec3 scale);
void update_and_draw(const float* field, float* renderField, GLuint texture, int N, int M, glm::mat4 matrix, GLuint quad_vao, GLuint shaderProgram);

void set_uniform_m4f(GLuint shaderProgram, const char* name, const glm::mat4& value);
void set_uniform_v3f(GLuint shaderProgram, const char* name, const glm::vec3& value);

void create_shader_program(GLuint* shaderProgram);
void create_quad(GLuint* VAO);
void create_texture_2D(float* data, unsigned int width, unsigned int height, GLuint* texture);
void create_texture_byte_2D(int* data, unsigned int width, unsigned int height, GLuint* texture);
void update_texture_2D(float* data, unsigned int width, unsigned int height, GLuint* texture);
void update_texture_byte_2D(int* data, unsigned int width, unsigned int height, GLuint* texture);
void draw_rect(GLuint shaderProgram, GLuint VAO, GLuint texture);
}
