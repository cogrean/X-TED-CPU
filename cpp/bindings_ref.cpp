#include <chrono>
#include <functional>
#include "TEST_XTED_CPU_IMPLEMENTATION/TED_C++.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Convert flat parent-index array to adjacency list.
static std::vector<std::vector<int>> parent_to_adj(const std::vector<int> &parent)
{
    int m = (int)parent.size();
    std::vector<std::vector<int>> adj(m);
    for (int i = 0; i < m; i++)
    {
        if (parent[i] >= 0)
            adj[parent[i]].push_back(i);
    }
    return adj;
}

// standard_ted mutates its label vectors (clears them); ref_compute takes them
// by value so pybind11 copies them before handing control to standard_ted.
static int ref_compute(
    std::vector<std::string> labels1, std::vector<int> parent1,
    std::vector<std::string> labels2, std::vector<int> parent2,
    int num_threads)
{
    std::vector<std::vector<int>> adj1 = parent_to_adj(parent1);
    std::vector<std::vector<int>> adj2 = parent_to_adj(parent2);
    return standard_ted(labels1, adj1, labels2, adj2, num_threads, 0);
}

PYBIND11_MODULE(TEST_XTED_REF, m) {
    m.def("compute", &ref_compute,
          py::call_guard<py::gil_scoped_release>(),
          "Reference Zhang-Shasha TED (rename cost: 0 same label, 1 different).");
}
