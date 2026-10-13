# Builds llama.cpp from tools/llama.cpp/src (fetched by build.sh) plus the
# scribe-dump tool, CPU only.
cmake_minimum_required(VERSION 3.14)
project(scribe_llama_tools CXX C)

add_subdirectory(src llama.cpp EXCLUDE_FROM_ALL)

add_executable(scribe-dump scribe_dump.cpp)
set_target_properties(scribe-dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
find_package(Threads REQUIRED)
target_link_libraries(scribe-dump PRIVATE llama-common llama ggml Threads::Threads)
target_compile_features(scribe-dump PRIVATE cxx_std_17)
