cmake_minimum_required(VERSION 3.24)

project(kvvmm_driver_console LANGUAGES CXX)

option(KVVMM_BUILD_TESTS "Build CPU-only unit tests for the radix cache" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(kvvmm_core
    src/cuda_driver.cpp
    src/driver_kernels.cpp
    src/radix_cache.cpp
    src/vmm_allocator.cpp
)

target_include_directories(kvvmm_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(kvvmm_core
    PUBLIC
        CUDA::cuda_driver
)

if(MSVC)
    target_compile_options(kvvmm_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(kvvmm_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(kvvmm-demo
    src/main.cpp
)

target_link_libraries(kvvmm-demo PRIVATE kvvmm_core)

if(KVVMM_BUILD_TESTS)
    enable_testing()

    add_executable(kvvmm-radix-tests
        tests/radix_cache_tests.cpp
        src/radix_cache.cpp
    )

    target_include_directories(kvvmm-radix-tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    if(MSVC)
        target_compile_options(kvvmm-radix-tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(kvvmm-radix-tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME radix_cache COMMAND kvvmm-radix-tests)
endif()
