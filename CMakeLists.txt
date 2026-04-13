cmake_minimum_required(VERSION 3.23)
project(XTED_CPU)

set(CMAKE_CXX_STANDARD 20)
set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(XTED_CPU cpp/bindings.cpp cpp/X-TED_C++.cpp)
target_link_libraries(XTED_CPU PRIVATE Threads::Threads)

install(TARGETS XTED_CPU DESTINATION xted)
