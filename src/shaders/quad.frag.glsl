#version 330 core
in vec2 texCoord;

uniform sampler2D texture1;

out vec4 FragColor;

void main(){
    vec3 colour = texture(texture1, texCoord).rgb;
    FragColor = vec4(colour, 1.0);
};
