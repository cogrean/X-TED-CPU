#include "ref_wrapper.h"

// Pull the reference implementation into this translation unit.
#include "TED_C++.h"
#include "TED.cpp"
#include "dataset_preprocess.cpp"
#include "parallel_computing.cpp"

int ref_ted(const std::vector<std::string>& l1,
            const std::vector<std::vector<int>>& a1,
            const std::vector<std::string>& l2,
            const std::vector<std::vector<int>>& a2,
            int threads)
{
    // standard_ted takes non-const refs and clears the label vectors,
    // so we must pass copies.
    auto labels1 = l1;
    auto adj1    = a1;
    auto labels2 = l2;
    auto adj2    = a2;
    return standard_ted(labels1, adj1, labels2, adj2, threads, /*parallel_version=*/1);
}
