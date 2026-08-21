cmake_minimum_required(VERSION 3.14)
project(gemma_reference CXX)
set(CMAKE_CXX_STANDARD 17)

# Point at a built llama.cpp tree: -DLLAMA_DIR=/path/to/llama.cpp
if(NOT DEFINED LLAMA_DIR)
    message(FATAL_ERROR "set -DLLAMA_DIR to a built llama.cpp checkout")
endif()

add_executable(dump_quants dump_quants.cpp)
target_include_directories(dump_quants PRIVATE
    ${LLAMA_DIR}/ggml/include)
target_link_directories(dump_quants PRIVATE ${LLAMA_DIR}/build/bin)
target_link_libraries(dump_quants PRIVATE ggml ggml-base ggml-cpu)
set_target_properties(dump_quants PROPERTIES
    BUILD_RPATH ${LLAMA_DIR}/build/bin)

add_executable(dump_layers dump_layers.cpp)
target_include_directories(dump_layers PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include)
target_link_directories(dump_layers PRIVATE ${LLAMA_DIR}/build/bin)
target_link_libraries(dump_layers PRIVATE llama ggml ggml-base ggml-cpu)
set_target_properties(dump_layers PROPERTIES
    BUILD_RPATH ${LLAMA_DIR}/build/bin)

add_executable(dump_tokens dump_tokens.cpp)
target_include_directories(dump_tokens PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include)
target_link_directories(dump_tokens PRIVATE ${LLAMA_DIR}/build/bin)
target_link_libraries(dump_tokens PRIVATE llama ggml ggml-base ggml-cpu)
set_target_properties(dump_tokens PROPERTIES
    BUILD_RPATH ${LLAMA_DIR}/build/bin)
