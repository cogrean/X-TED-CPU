// ============================================================
// bench_compare.cpp — Head-to-head benchmark
//   X-TED_CPU  vs  TEST_XTED_CPU_IMPLEMENTATION
//
// Both implementations share the same Zhang-Shasha algorithm with
// depth-levelled worklist parallelism, but differ in:
//   - namespace isolation (xted:: vs global)
//   - cost-matrix ownership (explicit vs internal label comparison)
//   - branchless min3 (X-TED) vs branched min3 (reference)
//   - keyroot-depth array sizing (K/L vs m/n)
//   - forest-distance matrix strategy (identical between both)
//
// Sections:
//   A. Correctness Agreement   — both impls must return the same TED,
//                                cross-checked against known ground truth
//   B. Head-to-Head Timing     — median wall-clock time over N_RUNS
//                                for 1, 2, and 4 threads
//   C. Thread-Scaling Summary  — per-impl speedup table derived from B
//
// Build:  make bench
// Run:    ./bench_compare
// ============================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Pull both implementations into this translation unit.
// X-TED lives in namespace xted; ref_ted() is a plain C++ function
// declared in ref_wrapper.h (definition compiled separately).
#include "X-TED_C++.cpp"
#include "ref_wrapper.h"

using std::cout;
using std::ifstream;
using std::istringstream;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

// ============================================================
// 0.  Infrastructure
// ============================================================

static const int N_RUNS = 7;  // odd → unambiguous median

// --- Output suppressor ----------------------------------------
// Both implementations print internal timing lines to std::cout.
// RAII guard: redirect cout to a sink while a call is in progress.
struct SuppressStdout {
    std::streambuf*   orig;
    std::ostringstream sink;
    SuppressStdout()  { orig = cout.rdbuf(sink.rdbuf()); }
    ~SuppressStdout() { cout.rdbuf(orig); }
};

static double vec_median(vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double vec_min(const vector<double>& v) {
    return *std::min_element(v.begin(), v.end());
}

// ============================================================
// 1.  Dataset helpers (shared with test_xted.cpp)
// ============================================================

static vector<vector<int>> build_cost(const vector<string>& l1,
                                      const vector<string>& l2) {
    int m = (int)l1.size(), n = (int)l2.size();
    vector<vector<int>> c(m, vector<int>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            c[i][j] = (l1[i] == l2[j]) ? 0 : 1;
    return c;
}

static vector<string> parse_labels(const string& s) {
    vector<string> v;
    istringstream ss(s);
    string tok;
    while (ss >> tok) v.push_back(tok);
    return v;
}

static vector<vector<int>> parse_adj(const string& line) {
    vector<vector<int>> adj;
    vector<int> cur;
    string num;
    int depth = 0;
    for (char c : line) {
        if (c == '[') {
            ++depth;
            if (depth == 2) { cur.clear(); num.clear(); }
        } else if (c == ']') {
            if (depth == 2) {
                if (!num.empty()) { cur.push_back(std::stoi(num)); num.clear(); }
                adj.push_back(cur);
            }
            --depth;
        } else if ((c == ',' || c == ' ') && depth == 2) {
            if (!num.empty()) { cur.push_back(std::stoi(num)); num.clear(); }
        } else if (std::isdigit((unsigned char)c) && depth == 2) {
            num += c;
        }
    }
    return adj;
}

static pair<vector<string>, vector<vector<int>>>
load_tree(const string& nodes_path, const string& adj_path, int line_idx) {
    auto get_line = [](const string& path, int idx) {
        ifstream f(path);
        if (!f.is_open()) throw runtime_error("Cannot open: " + path);
        string line;
        for (int i = 0; i <= idx; ++i)
            if (!std::getline(f, line))
                throw runtime_error("Not enough lines in: " + path);
        return line;
    };
    return { parse_labels(get_line(nodes_path, line_idx)),
             parse_adj(get_line(adj_path,  line_idx)) };
}

// Dataset roots
static const string BASE =
    "/Users/colinogrean/Desktop/pyX-TED/X-TED_CPU/Sampled_Dataset/";
static const string BASE_XL =
    "/Users/colinogrean/Desktop/pyX-TED/X-TED_CPU/Sampled_Dataset_Extra_Large_Trees/";

// ============================================================
// 2.  Timed call wrappers
// ============================================================

// Returns {TED value, elapsed ms} — suppresses impl stdout.
static pair<int, double>
timed_xted(const vector<string>& l1, const vector<vector<int>>& a1,
           const vector<string>& l2, const vector<vector<int>>& a2,
           const vector<vector<int>>& cost, int threads)
{
    SuppressStdout s;
    auto t0  = std::chrono::steady_clock::now();
    int  ted = xted::XTED_CPU(l1, a1, l2, a2, cost, threads);
    auto t1  = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return {ted, ms};
}

static pair<int, double>
timed_ref(const vector<string>& l1, const vector<vector<int>>& a1,
          const vector<string>& l2, const vector<vector<int>>& a2,
          int threads)
{
    SuppressStdout s;
    auto t0  = std::chrono::steady_clock::now();
    int  ted = ref_ted(l1, a1, l2, a2, threads);
    auto t1  = std::chrono::steady_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return {ted, ms};
}

// Collect N_RUNS samples and return {median_ms, min_ms, last_ted}.
struct RunStats { double med_ms, min_ms; int ted; };

static RunStats
collect_xted(const vector<string>& l1, const vector<vector<int>>& a1,
             const vector<string>& l2, const vector<vector<int>>& a2,
             const vector<vector<int>>& cost, int threads)
{
    vector<double> times;
    int ted = -1;
    for (int r = 0; r < N_RUNS; ++r) {
        auto [t, ms] = timed_xted(l1, a1, l2, a2, cost, threads);
        ted = t; times.push_back(ms);
    }
    return { vec_median(times), vec_min(times), ted };
}

static RunStats
collect_ref(const vector<string>& l1, const vector<vector<int>>& a1,
            const vector<string>& l2, const vector<vector<int>>& a2,
            int threads)
{
    vector<double> times;
    int ted = -1;
    for (int r = 0; r < N_RUNS; ++r) {
        auto [t, ms] = timed_ref(l1, a1, l2, a2, threads);
        ted = t; times.push_back(ms);
    }
    return { vec_median(times), vec_min(times), ted };
}

// ============================================================
// Section A: Correctness Agreement
// ============================================================

struct CorrectCase {
    string name;
    string nf, af;
    int ia, ib;
    int expected; // -1 = unknown (only test agreement)
};

void section_a_correctness() {
    cout << "\n=== Section A: Correctness Agreement ===\n";
    cout << "  Both implementations run at 1 thread.\n"
         << "  Known expected values come from the verified test suite.\n\n";

    // Columns header
    cout << "  " << std::left  << std::setw(35) << "Case"
         << std::right
         << std::setw(10) << "Expected"
         << std::setw(8)  << "X-TED"
         << std::setw(8)  << "Ref"
         << "  Status\n";
    cout << "  " << string(35+10+8+8+9, '-') << "\n";

    int s_agree = 0, s_disagree = 0, s_skipped = 0;

    vector<CorrectCase> cases = {
        // ---- Verified ground-truth pairs ----
        { "swissport_100 pair(0,1)",
          BASE+"1_Swissport/swissport_nodes_100.txt",
          BASE+"1_Swissport/swissport_nodes_adj_100.txt",     0, 1,   44 },
        { "python_100 pair(0,1)",
          BASE+"2_Python/python_nodes_100.txt",
          BASE+"2_Python/python_nodes_adj_100.txt",           0, 1,  100 },
        { "dblp_100 pair(0,1)",
          BASE+"4_DBLP/dblp_nodes_100.txt",
          BASE+"4_DBLP/dblp_nodes_adj_100.txt",               0, 1,   47 },
        { "swissport_500 pair(0,1)",
          BASE+"1_Swissport/swissport_nodes_500.txt",
          BASE+"1_Swissport/swissport_nodes_adj_500.txt",     0, 1,  424 },
        { "python_500 pair(0,1)",
          BASE+"2_Python/python_nodes_500.txt",
          BASE+"2_Python/python_nodes_adj_500.txt",           0, 1,  546 },
        { "dblp_500 pair(0,1)",
          BASE+"4_DBLP/dblp_nodes_500.txt",
          BASE+"4_DBLP/dblp_nodes_adj_500.txt",               0, 1,  224 },
        { "swissport_1000 pair(0,1)",
          BASE+"1_Swissport/swissport_nodes_1000.txt",
          BASE+"1_Swissport/swissport_nodes_adj_1000.txt",    0, 1,  767 },
        { "python_1000 pair(0,1)",
          BASE+"2_Python/python_nodes_1000.txt",
          BASE+"2_Python/python_nodes_adj_1000.txt",          0, 1, 1103 },
        // ---- Self-distance: both must return 0 ----
        { "swissport_100 self(0)",
          BASE+"1_Swissport/swissport_nodes_100.txt",
          BASE+"1_Swissport/swissport_nodes_adj_100.txt",     0, 0,    0 },
        { "python_100 self(0)",
          BASE+"2_Python/python_nodes_100.txt",
          BASE+"2_Python/python_nodes_adj_100.txt",           0, 0,    0 },
        { "dblp_500 self(0)",
          BASE+"4_DBLP/dblp_nodes_500.txt",
          BASE+"4_DBLP/dblp_nodes_adj_500.txt",               0, 0,    0 },
        // ---- Additional pairs without known ground truth ----
        { "swissport_100 pair(1,2)",
          BASE+"1_Swissport/swissport_nodes_100.txt",
          BASE+"1_Swissport/swissport_nodes_adj_100.txt",     1, 2,   -1 },
        { "python_100 pair(1,2)",
          BASE+"2_Python/python_nodes_100.txt",
          BASE+"2_Python/python_nodes_adj_100.txt",           1, 2,   -1 },
        { "dblp_100 pair(1,2)",
          BASE+"4_DBLP/dblp_nodes_100.txt",
          BASE+"4_DBLP/dblp_nodes_adj_100.txt",               1, 2,   -1 },
        { "swissport_200 pair(0,1)",
          BASE+"1_Swissport/swissport_nodes_200.txt",
          BASE+"1_Swissport/swissport_nodes_adj_200.txt",     0, 1,   -1 },
        { "python_200 pair(0,1)",
          BASE+"2_Python/python_nodes_200.txt",
          BASE+"2_Python/python_nodes_adj_200.txt",           0, 1,   -1 },
        // ---- Bolzano small trees ----
        { "bolzano pair(0,1)",
          BASE+"3_Bolzano/bolzano_nodes.txt",
          BASE+"3_Bolzano/bolzano_nodes_adj.txt",             0, 1,    1 },
        { "bolzano pair(5,6)",
          BASE+"3_Bolzano/bolzano_nodes.txt",
          BASE+"3_Bolzano/bolzano_nodes_adj.txt",             5, 6,    2 },
    };

    for (auto& c : cases) {
        try {
            auto [l1, a1] = load_tree(c.nf, c.af, c.ia);
            auto [l2, a2] = load_tree(c.nf, c.af, c.ib);
            auto cost = build_cost(l1, l2);

            auto [xv, _x] = timed_xted(l1, a1, l2, a2, cost, 1);
            auto [rv, _r] = timed_ref(l1, a1, l2, a2, 1);

            bool agree = (xv == rv);
            bool vs_expected = (c.expected < 0) || (xv == c.expected && rv == c.expected);
            bool ok = agree && vs_expected;

            cout << "  " << std::left  << std::setw(35) << c.name;
            if (c.expected >= 0)
                cout << std::right << std::setw(10) << c.expected;
            else
                cout << std::right << std::setw(10) << "(n/a)";
            cout << std::setw(8) << xv
                 << std::setw(8) << rv
                 << "  " << (ok ? "[PASS]" : (agree ? "[WRONG]" : "[MISMATCH]"))
                 << "\n";

            ok ? ++s_agree : ++s_disagree;
        } catch (const std::exception& e) {
            cout << "  " << std::left << std::setw(35) << c.name
                 << "  [SKIP] " << e.what() << "\n";
            ++s_skipped;
        }
    }

    cout << "\n  Agreement: " << s_agree
         << "  Failures: " << s_disagree
         << "  Skipped: " << s_skipped << "\n";
}

// ============================================================
// Section B: Head-to-Head Timing
// ============================================================

struct BenchCase {
    string name;
    string nf, af;
    int ia, ib;
};

// Result for one (case × thread-count) combination.
struct TimingRow {
    string name;
    int    threads;
    int    ted_xted, ted_ref;
    bool   agree;
    double xted_med, xted_min;
    double ref_med,  ref_min;
    double speedup; // ref_med / xted_med  (>1 means X-TED is faster)
};

static vector<TimingRow> g_timing_results; // populated by section B, read by section C

void section_b_timing() {
    cout << "\n=== Section B: Head-to-Head Timing (" << N_RUNS
         << " runs, median) ===\n\n";

    // Column header
    cout << std::left
         << "  " << std::setw(22) << "Dataset"
         << " Th"
         << std::right
         << std::setw(12) << "X-TED med"
         << std::setw(11) << "Ref med"
         << std::setw(9)  << "Speedup"
         << std::setw(8)  << "TED"
         << "  OK?\n";
    cout << "  " << string(22+2+12+11+9+8+6, '-') << "\n";

    vector<BenchCase> cases = {
        // ---- Small (100 nodes) ----
        { "swissport_100",
          BASE+"1_Swissport/swissport_nodes_100.txt",
          BASE+"1_Swissport/swissport_nodes_adj_100.txt", 0, 1 },
        { "python_100",
          BASE+"2_Python/python_nodes_100.txt",
          BASE+"2_Python/python_nodes_adj_100.txt",       0, 1 },
        { "dblp_100",
          BASE+"4_DBLP/dblp_nodes_100.txt",
          BASE+"4_DBLP/dblp_nodes_adj_100.txt",           0, 1 },
        // ---- Medium (500 nodes) ----
        { "swissport_500",
          BASE+"1_Swissport/swissport_nodes_500.txt",
          BASE+"1_Swissport/swissport_nodes_adj_500.txt", 0, 1 },
        { "python_500",
          BASE+"2_Python/python_nodes_500.txt",
          BASE+"2_Python/python_nodes_adj_500.txt",       0, 1 },
        { "dblp_500",
          BASE+"4_DBLP/dblp_nodes_500.txt",
          BASE+"4_DBLP/dblp_nodes_adj_500.txt",           0, 1 },
        // ---- Large (1000 nodes) ----
        { "swissport_1000",
          BASE+"1_Swissport/swissport_nodes_1000.txt",
          BASE+"1_Swissport/swissport_nodes_adj_1000.txt", 0, 1 },
        { "python_1000",
          BASE+"2_Python/python_nodes_1000.txt",
          BASE+"2_Python/python_nodes_adj_1000.txt",       0, 1 },
        // ---- Extra-large (2000 nodes) ----
        { "swissport_XL_2000",
          BASE_XL+"swissport/2000_nodes/swissport_nodes_2000.txt",
          BASE_XL+"swissport/2000_nodes/swissport_nodes_adj_2000.txt", 0, 1 },
        { "python_XL_2000",
          BASE_XL+"python/2000_nodes/python_nodes_2000.txt",
          BASE_XL+"python/2000_nodes/python_nodes_adj_2000.txt",       0, 1 },
    };

    const vector<int> thread_counts = {1, 2, 4, 12};

    for (auto& b : cases) {
        bool first = true;
        for (int th : thread_counts) {
            try {
                auto [l1, a1] = load_tree(b.nf, b.af, b.ia);
                auto [l2, a2] = load_tree(b.nf, b.af, b.ib);
                auto cost = build_cost(l1, l2);

                auto xs = collect_xted(l1, a1, l2, a2, cost, th);
                auto rs = collect_ref(l1, a1, l2, a2, th);

                TimingRow row;
                row.name     = b.name;
                row.threads  = th;
                row.ted_xted = xs.ted;
                row.ted_ref  = rs.ted;
                row.agree    = (xs.ted == rs.ted);
                row.xted_med = xs.med_ms;
                row.xted_min = xs.min_ms;
                row.ref_med  = rs.med_ms;
                row.ref_min  = rs.min_ms;
                row.speedup  = (xs.med_ms > 0.0) ? rs.med_ms / xs.med_ms : 0.0;
                g_timing_results.push_back(row);

                cout << std::fixed << std::setprecision(2);
                cout << "  " << std::left  << std::setw(22)
                     << (first ? b.name : "")
                     << " " << std::right << std::setw(1) << th << "t"
                     << std::setw(9) << xs.med_ms << " ms"
                     << std::setw(8) << rs.med_ms << " ms"
                     << std::setw(7) << row.speedup << "x"
                     << std::setw(8) << xs.ted
                     << "  " << (row.agree ? "OK" : "MISMATCH!")
                     << "\n";
                first = false;

            } catch (const std::exception& e) {
                cout << "  " << std::left << std::setw(22) << b.name
                     << " " << th << "t  [SKIP] " << e.what() << "\n";
            }
        }
        if (!first) cout << "\n";
    }
}

// ============================================================
// Section C: Thread-Scaling Summary
// ============================================================

void section_c_scaling() {
    cout << "\n=== Section C: Thread-Scaling Summary ===\n";
    cout << "  Speedup = time(1t) / time(Nt) for each implementation.\n";
    cout << "  Derived from Section B median measurements.\n\n";

    // Collect unique dataset names in order of first appearance.
    vector<string> names;
    for (auto& r : g_timing_results) {
        if (std::find(names.begin(), names.end(), r.name) == names.end())
            names.push_back(r.name);
    }

    // Header
    cout << "  " << std::left << std::setw(22) << "Dataset"
         << std::right
         << std::setw(11) << "Impl"
         << std::setw(12) << "1t (ms)"
         << std::setw(12) << "2t (ms)"
         << std::setw(8)  << "×2t"
         << std::setw(12) << "4t (ms)"
         << std::setw(8)  << "×4t"
         << "\n";
    cout << "  " << string(22+11+12+12+8+12+8, '-') << "\n";

    for (auto& name : names) {
        // Gather 1t, 2t, 4t rows for this dataset
        TimingRow* r1 = nullptr; TimingRow* r2 = nullptr; TimingRow* r4 = nullptr;
        for (auto& r : g_timing_results) {
            if (r.name != name) continue;
            if (r.threads == 1) r1 = &r;
            if (r.threads == 2) r2 = &r;
            if (r.threads == 4) r4 = &r;
        }
        if (!r1) continue; // dataset was fully skipped

        cout << std::fixed << std::setprecision(2);

        // X-TED row
        cout << "  " << std::left  << std::setw(22) << name
             << std::right << std::setw(11) << "X-TED"
             << std::setw(12) << r1->xted_med;
        if (r2) cout << std::setw(12) << r2->xted_med
                     << std::setw(8)  << (r1->xted_med / r2->xted_med);
        else    cout << std::setw(12) << "-" << std::setw(8) << "-";
        if (r4) cout << std::setw(12) << r4->xted_med
                     << std::setw(8)  << (r1->xted_med / r4->xted_med);
        else    cout << std::setw(12) << "-" << std::setw(8) << "-";
        cout << "\n";

        // Ref row (same dataset column, blank name)
        cout << "  " << std::left  << std::setw(22) << ""
             << std::right << std::setw(11) << "Ref"
             << std::setw(12) << r1->ref_med;
        if (r2) cout << std::setw(12) << r2->ref_med
                     << std::setw(8)  << (r1->ref_med / r2->ref_med);
        else    cout << std::setw(12) << "-" << std::setw(8) << "-";
        if (r4) cout << std::setw(12) << r4->ref_med
                     << std::setw(8)  << (r1->ref_med / r4->ref_med);
        else    cout << std::setw(12) << "-" << std::setw(8) << "-";
        cout << "\n";

        // Speedup row: X-TED vs Ref at each thread count
        cout << "  " << std::left  << std::setw(22) << ""
             << std::right << std::setw(11) << "Δ (ref/xted)";
        auto ratio = [](double ref, double xted) -> string {
            if (xted <= 0.0) return "-";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << ref / xted << "x";
            return ss.str();
        };
        cout << std::setw(12) << ratio(r1->ref_med, r1->xted_med);
        if (r2) cout << std::setw(12) << ratio(r2->ref_med, r2->xted_med) << std::setw(8) << "";
        else    cout << std::setw(12) << "-" << std::setw(8) << "";
        if (r4) cout << std::setw(12) << ratio(r4->ref_med, r4->xted_med) << std::setw(8) << "";
        else    cout << std::setw(12) << "-" << std::setw(8) << "";
        cout << "\n\n";
    }
}

// ============================================================
// main
// ============================================================

int main() {
    cout << "====================================================\n"
         << "  X-TED_CPU vs TEST_XTED_CPU_IMPLEMENTATION\n"
         << "  Benchmark — " << N_RUNS << " timing runs per case\n"
         << "====================================================\n";

    section_a_correctness();
    section_b_timing();
    section_c_scaling();

    // Final verdict: non-zero exit if any TED values disagree.
    int mismatches = 0;
    for (auto& r : g_timing_results)
        if (!r.agree) ++mismatches;

    cout << "\n====================================================\n"
         << "  Benchmark Complete\n";
    if (mismatches > 0)
        cout << "  WARNING: " << mismatches
             << " timing case(s) produced mismatched TED values!\n";
    else
        cout << "  All timed cases produced matching TED values.\n";
    cout << "====================================================\n";

    return (mismatches > 0) ? 1 : 0;
}
