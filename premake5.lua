require "premake/premake5-cuda"

workspace "OpenGL_Cuda"
    -- location "generated"
    language "C++"
    architecture "x86_64"

    configurations { "release", "debug" }

    filter { "configurations:debug" }
        defines { "_DEBUG" }
        symbols "On"
        optimize "Off"
        runtime "Debug"

    filter { "configurations:release" }
        defines { "_RELEASE" }
        symbols "Off"
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


project "KERNELS"
    kind "SharedLib"
    toolset "nvcc"

    cudaPath "/opt/cuda/"

    includedirs {
        "./src/kernels/",
    }

    files {
        "src/kernels/*.cu",
    }

    rules { "cu" }

    cudaCompilerOptions { "--gpu-architecture=sm_50" }

    links { "cuda", "cudart" }



project "OpenGL_Cuda"
    kind "WindowedApp"
    toolset "clang"

    includedirs {
        "./src/",
        "./deps/glad/include",
        "./deps/glfw/include",
        "./deps/glm/",
        "/opt/cuda/include"
        -- "/usr/local/cuda-12.6/include"
    }

    files {
        "src/**.cpp"
    }

    libdirs {
        "/opt/cuda/lib64"
        -- "/usr/local/cuda/lib64"
    }

    links { "GLAD", "GLFW", "GLM", "cuda", "cudart", "KERNELS" }
