#include "X-TED_C++.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(XTED_CPU, m) {
    m.def("compute_tree_edit_distance", &xted::XTED_CPU, py::call_guard<py::gil_scoped_release>());
    m.def("compute_tree_edit_distance_uniform", &xted::XTED_CPU_uniform, py::call_guard<py::gil_scoped_release>());
}