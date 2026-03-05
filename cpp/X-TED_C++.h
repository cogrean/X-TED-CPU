#include <string>
#include <vector>

namespace xted {
// Generic multi-core CPU implementation with a manual cost matrix
int XTED_CPU(std::vector<std::string> label1, std::vector<std::vector<int>> parent1, std::vector<std::string> label2, std::vector<std::vector<int>> parent2, std::vector<std::vector<int>> cost_matrix, int num_threads);
// Uniform-cost variant: rename costs 0 for matching labels, 1 otherwise. No cost matrix required.
int XTED_CPU_uniform(std::vector<std::string> label1, std::vector<std::vector<int>> parent1, std::vector<std::string> label2, std::vector<std::vector<int>> parent2, int num_threads);
}
