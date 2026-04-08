#ifndef REF_WRAPPER_H
#define REF_WRAPPER_H

#include <string>
#include <vector>

// Thin wrapper around the reference implementation's standard_ted().
// Makes copies of inputs because standard_ted() mutates its arguments.
int ref_ted(const std::vector<std::string>& l1,
            const std::vector<std::vector<int>>& a1,
            const std::vector<std::string>& l2,
            const std::vector<std::vector<int>>& a2,
            int threads);

#endif // REF_WRAPPER_H
