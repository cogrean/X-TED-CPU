"""
Run a single tree-edit-distance calculation selected by index, using
either X-TED (default) or zss (Zhang-Shasha reference).

Usage:
    python test_singular.py <n> [xted|zss]                run test n
    python test_singular.py list                          print all tests
    python test_singular.py <n> xted --threads <t>        override X-TED thread count

Examples:
    python test_singular.py 2                   # X-TED, 4 threads
    python test_singular.py 2 zss               # Zhang-Shasha
    python test_singular.py 7 xted --threads 8  # X-TED, 8 threads
"""

import sys
import ast
import time
import statistics
import ctypes
import ctypes.util
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Peak heap sampler — uses the same malloc_zone_statistics(size_in_use) metric
# as bench_compare.cpp's PeakMemSampler.  Tracks LIVE allocated bytes (not RSS
# watermark), so the delta excludes Python interpreter baseline and correctly
# returns to zero after C++ vectors are freed between calls.
# ---------------------------------------------------------------------------
_libsys = ctypes.CDLL(ctypes.util.find_library("System"))

class _MallocStats(ctypes.Structure):
    _fields_ = [
        ("blocks_in_use",       ctypes.c_uint),
        ("size_in_use",         ctypes.c_size_t),
        ("max_size_in_use",     ctypes.c_size_t),
        ("size_allocated",      ctypes.c_size_t),
        ("bytes_allocated_ever",ctypes.c_uint64),
    ]

def _heap_in_use():
    s = _MallocStats()
    _libsys.malloc_zone_statistics(None, ctypes.byref(s))
    return s.size_in_use

class PeakMemPoller:
    """Context manager that polls heap_in_use every 1 ms and reports the peak
    delta above the baseline sampled at __enter__ time."""
    def __init__(self, interval=0.001):
        self._interval = interval
        self._baseline = 0
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._baseline = _heap_in_use()
        self._peak = self._baseline
        self._stop.clear()
        def _poll():
            while not self._stop.is_set():
                v = _heap_in_use()
                if v > self._peak:
                    self._peak = v
                self._stop.wait(self._interval)
        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self):
        delta = self._peak - self._baseline
        return max(0.0, delta / 1e6)

sys.path.insert(0, str(Path(__file__).parent.parent))
from xted import x_ted_compute
import zss

BASE = Path(__file__).parent.parent / "Sampled_Dataset"
BASE_XL = Path(__file__).parent.parent / "Sampled_Dataset_Extra_Large_Trees"


def load_dataset(nodes_path, adj_path, tree_idx=0):
    with open(nodes_path) as f:
        for i, line in enumerate(f):
            if i == tree_idx:
                labels = line.strip().split()
                break
    with open(adj_path) as f:
        for i, line in enumerate(f):
            if i == tree_idx:
                adj = ast.literal_eval(line.strip())
                break
    n = len(adj)
    parent = [-1] * n
    for i in range(n):
        for child in adj[i]:
            parent[child] = i
    return parent, labels


def build_zss_tree(parent, labels):
    n = len(parent)
    nodes = [zss.Node(labels[i]) for i in range(n)]
    for i in range(1, n):
        nodes[parent[i]].addkid(nodes[i])
    return nodes[0]


def dataset_test(name, nodes_path, adj_path, idx_a=0, idx_b=1, expected=None):
    return {"name": name, "nodes": nodes_path, "adj": adj_path,
            "idx_a": idx_a, "idx_b": idx_b, "expected": expected}


TESTS = [
    # 1-3: 100-node datasets
    dataset_test("Swissport 100  (pair 0,1)",
                 BASE / "1_Swissport/swissport_nodes_100.txt",
                 BASE / "1_Swissport/swissport_nodes_adj_100.txt", expected=44),
    dataset_test("Python 100  (pair 0,1)",
                 BASE / "2_Python/python_nodes_100.txt",
                 BASE / "2_Python/python_nodes_adj_100.txt", expected=100),
    dataset_test("DBLP 100  (pair 0,1)",
                 BASE / "4_DBLP/dblp_nodes_100.txt",
                 BASE / "4_DBLP/dblp_nodes_adj_100.txt", expected=47),
    # 4-6: 500-node datasets
    dataset_test("Swissport 500  (pair 0,1)",
                 BASE / "1_Swissport/swissport_nodes_500.txt",
                 BASE / "1_Swissport/swissport_nodes_adj_500.txt", expected=424),
    dataset_test("Python 500  (pair 0,1)",
                 BASE / "2_Python/python_nodes_500.txt",
                 BASE / "2_Python/python_nodes_adj_500.txt", expected=546),
    dataset_test("DBLP 500  (pair 0,1)",
                 BASE / "4_DBLP/dblp_nodes_500.txt",
                 BASE / "4_DBLP/dblp_nodes_adj_500.txt", expected=224),
    # 7-8: 1000-node datasets
    dataset_test("Swissport 1000  (pair 0,1)",
                 BASE / "1_Swissport/swissport_nodes_1000.txt",
                 BASE / "1_Swissport/swissport_nodes_adj_1000.txt", expected=767),
    dataset_test("Python 1000  (pair 0,1)",
                 BASE / "2_Python/python_nodes_1000.txt",
                 BASE / "2_Python/python_nodes_adj_1000.txt", expected=1103),
    # 9-10: 2000-node extra-large
    dataset_test("Swissport 2000  (pair 0,1)",
                 BASE_XL / "swissport/2000_nodes/swissport_nodes_2000.txt",
                 BASE_XL / "swissport/2000_nodes/swissport_nodes_adj_2000.txt"),
    dataset_test("Python 2000  (pair 0,1)",
                 BASE_XL / "python/2000_nodes/python_nodes_2000.txt",
                 BASE_XL / "python/2000_nodes/python_nodes_adj_2000.txt"),
]


def print_list():
    print("Available tests:")
    for i, t in enumerate(TESTS, 1):
        exp = f"  expected={t['expected']}" if t["expected"] is not None else ""
        print(f"  {i:2d}.  {t['name']}{exp}")


def run_xted(t, num_threads, num_runs=25):
    print(f"Test:       {t['name']}")
    print(f"Algorithm:  X-TED  ({num_threads} threads, {num_runs} runs)")

    p1, l1 = load_dataset(t["nodes"], t["adj"], t["idx_a"])
    p2, l2 = load_dataset(t["nodes"], t["adj"], t["idx_b"])
    print(f"Tree sizes: {len(l1)} nodes vs {len(l2)} nodes")

    # Poller starts here — baseline captures heap before any X-TED allocations.
    with PeakMemPoller() as mem:
        # Warmup: primes allocator and caches at real problem size.
        for _ in range(10):
            result = x_ted_compute(p1, l1, p2, l2, num_threads=num_threads)

        timings = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            result = x_ted_compute(p1, l1, p2, l2, num_threads=num_threads)
            timings.append((time.perf_counter() - t0) * 1000)

    median_ms = statistics.median(timings)
    print(f"TED:        {result}")
    if num_runs > 1:
        print(f"Time:       median {median_ms:.3f} ms  "
              f"(min {min(timings):.3f}, max {max(timings):.3f})")
    else:
        print(f"Time:       {median_ms:.3f} ms")
    print(f"Memory:     {mem.peak_mb:.1f} MB peak working set")

    if t["expected"] is not None:
        status = "PASS" if result == t["expected"] else f"FAIL (expected {t['expected']})"
        print(f"Result:     {status}")


def run_zss(t, num_runs=25):
    print(f"Test:       {t['name']}")
    print(f"Algorithm:  Zhang-Shasha (zss, single-threaded, {num_runs} runs)")

    p1, l1 = load_dataset(t["nodes"], t["adj"], t["idx_a"])
    p2, l2 = load_dataset(t["nodes"], t["adj"], t["idx_b"])
    print(f"Tree sizes: {len(l1)} nodes vs {len(l2)} nodes")

    tree1 = build_zss_tree(p1, l1)
    tree2 = build_zss_tree(p2, l2)

    # Poller starts here — baseline captures heap before any zss allocations.
    with PeakMemPoller() as mem:
        for _ in range(10):
            result = int(zss.simple_distance(tree1, tree2))

        timings = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            result = int(zss.simple_distance(tree1, tree2))
            timings.append((time.perf_counter() - t0) * 1000)

    median_ms = statistics.median(timings)
    print(f"TED:        {result}")
    if num_runs > 1:
        print(f"Time:       median {median_ms:.3f} ms  "
              f"(min {min(timings):.3f}, max {max(timings):.3f})")
    else:
        print(f"Time:       {median_ms:.3f} ms")
    print(f"Memory:     {mem.peak_mb:.1f} MB peak working set")

    if t["expected"] is not None:
        status = "PASS" if result == t["expected"] else f"FAIL (expected {t['expected']})"
        print(f"Result:     {status}")


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if args[0] == "list":
        print_list()
        sys.exit(0)

    try:
        idx = int(args[0])
    except ValueError:
        print(f"Error: expected a test number or 'list', got '{args[0]}'")
        sys.exit(1)

    if not (1 <= idx <= len(TESTS)):
        print(f"Error: test {idx} out of range (1–{len(TESTS)})")
        print_list()
        sys.exit(1)

    # Optional algorithm selector (second positional arg).
    algo = "xted"
    if len(args) >= 2 and args[1] in ("xted", "zss"):
        algo = args[1]
    elif len(args) >= 2 and not args[1].startswith("--"):
        print(f"Error: unknown algorithm '{args[1]}' (choose xted or zss)")
        sys.exit(1)

    num_threads = 4
    if "--threads" in args:
        try:
            num_threads = int(args[args.index("--threads") + 1])
        except (IndexError, ValueError):
            print("Error: --threads requires an integer argument")
            sys.exit(1)

    num_runs = 25
    if "--runs" in args:
        try:
            num_runs = int(args[args.index("--runs") + 1])
        except (IndexError, ValueError):
            print("Error: --runs requires an integer argument")
            sys.exit(1)

    test = TESTS[idx - 1]
    if algo == "xted":
        run_xted(test, num_threads, num_runs)
    else:
        run_zss(test, num_runs)


if __name__ == "__main__":
    main()
