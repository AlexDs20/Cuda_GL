workspace "OpenGL_Cuda"
    location "generated"
    language "C++"
    architecture "x86_64"

    configurations { "release", "debug" }

    filter { "configurations:debug" }
        defines { "_DEBUG" }
        symbols "On"
        runtime "Debug"

    filter { "configurations:release" }
        defines { "_RELEASE" }
        optimize "On"
        runtime "Release"

    filter { "system:Windows" }
        defines { "_WIN32" }

    filter { "system:Unix" }
        toolset "clang"
        defines { "_LINUX" }

    filter { "system:Mac" }
        defines { "_MAC" }

    filter {}

    targetdir ("build/bin/%{cfg.system}/%{prj.name}/%{cfg.longname}")
    objdir ("build/obj/%{cfg.system}/%{prj.name}/%{cfg.longname}")

include "deps/glad.lua"
include "deps/glfw.lua"
include "deps/glm.lua"

project "OpenGL_Cuda"
    kind "WindowedApp"
    toolset "clang"

    includedirs
    {
        "./src/",
        "./deps/glad/include",
        "./deps/glfw/include",
        "./deps/glm/",
        "/usr/local/cuda-12.6/include"
    }

    files
    {
        "src/**.cpp"
    }

    libdirs
    {
        "/usr/local/cuda/lib64"
    }

    links { "GLAD", "GLFW", "GLM", "cuda", "cudart" }
