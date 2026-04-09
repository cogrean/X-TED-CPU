#pragma once

#include <string>
#include <vector>

namespace xted {
// Generic multi-core CPU implementation with a manual cost matrix
// parent1/parent2: flat parent-index arrays where parent[i] is the parent of node i (-1 for root)
int XTED_CPU(const std::vector<std::string>& label1, const std::vector<int>& parent1, const std::vector<std::string>& label2, const std::vector<int>& parent2, const std::vector<std::vector<int>>& cost_matrix, int num_threads);
// Uniform-cost variant: rename costs 0 for matching labels, 1 otherwise. No cost matrix required.
int XTED_CPU_uniform(const std::vector<std::string>& label1, const std::vector<int>& parent1, const std::vector<std::string>& label2, const std::vector<int>& parent2, int num_threads);
}
