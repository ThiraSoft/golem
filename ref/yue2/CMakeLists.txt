cmake_minimum_required(VERSION 3.14)
project(yue2_reference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED YUE2_ROOT)
    set(YUE2_ROOT "/mnt/data/dev/musicbench/yue2")
endif()

find_package(Threads REQUIRED)

add_compile_definitions(GGML_MAX_NAME=128)

add_executable(dump_vae dump_vae.cpp)

target_include_directories(dump_vae PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${YUE2_ROOT}/src
    ${YUE2_ROOT}
)
target_include_directories(dump_vae SYSTEM PRIVATE
    ${YUE2_ROOT}/ggml/include
)

target_link_directories(dump_vae PRIVATE
    ${YUE2_ROOT}/build
    ${YUE2_ROOT}/build/ggml/src
)

target_link_libraries(dump_vae PRIVATE
    ggml
    ggml-base
    ggml-cpu
    ggml-vulkan
    Threads::Threads
)

set_target_properties(dump_vae PROPERTIES
    BUILD_RPATH "${YUE2_ROOT}/build"
    INSTALL_RPATH "${YUE2_ROOT}/build"
)
