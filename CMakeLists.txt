cmake_minimum_required(VERSION 3.18)
project(WebGpuServer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Znajdź CUDA
find_package(CUDAToolkit REQUIRED)

# 2. Ręczne i bezpieczne wskazanie ścieżek
# Używamy cudzysłowów, aby obsłużyć spacje w ścieżkach Windowsa
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")

# Szukamy nagłówków rdzenia Node.js w standardowej lokalizacji node-gyp na Windows.
# Gwiazdka (*) pozwoli CMake'owi przeszukać foldery z różnymi wersjami Node.
file(GLOB NODE_GYP_INCLUDES "$ENV{LOCALAPPDATA}/node-gyp/Cache/*/include/node")

foreach(NODE_DIR ${NODE_GYP_INCLUDES})
    include_directories(SYSTEM "${NODE_DIR}")
endforeach()

include_directories(SYSTEM "${CUDAToolkit_INCLUDE_DIRS}")

# 3. Definicje, które uciszają błędy Clangd
add_definitions(-DNAPI_CPP_EXCEPTIONS)

# 4. Cel dla edytora
add_executable(MagisterkaDummy
        src/cuda/addon.cpp
        src/cuda/matrix_kernels.cu
        src/cuda/cuda_worker.h
        src/cuda/matrix_kernels.h
)

target_link_libraries(MagisterkaDummy CUDA::cudart)