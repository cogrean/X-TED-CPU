cmake_minimum_required(VERSION 3.23)
project(XTED_CPU)

set(CMAKE_CXX_STANDARD 17)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(XTED_CPU cpp/bindings.cpp cpp/X-TED_C++.cpp)
target_link_libraries(XTED_CPU PRIVATE Threads::Threads)

pybind11_add_module(TEST_XTED_REF
    cpp/bindings_ref.cpp
    cpp/TEST_XTED_CPU_IMPLEMENTATION/TED.cpp
    cpp/TEST_XTED_CPU_IMPLEMENTATION/dataset_preprocess.cpp
    cpp/TEST_XTED_CPU_IMPLEMENTATION/parallel_computing.cpp)
target_include_directories(TEST_XTED_REF PRIVATE cpp/TEST_XTED_CPU_IMPLEMENTATION)
target_link_libraries(TEST_XTED_REF PRIVATE Threads::Threads)

install(TARGETS XTED_CPU TEST_XTED_REF DESTINATION xted_cpu)
