project "GLAD"
    kind "SharedLib"
    language "C"

    includedirs { "glad/include" }

    files { "glad/src/glad.c" }
