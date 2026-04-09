"""
Cross-implementation comparison tests for Tree Edit Distance.

Three implementations under test:
  XTED_CPU       – X-TED parallel C++ (this project)
  TEST_XTED_REF  – reference sequential Zhang-Shasha C++ (TEST_XTED_CPU_IMPLEMENTATION)
  zss            – Python Zhang-Shasha library

Cost semantics:
  xted_label  – XTED_CPU with label-matching cost  (rename = 0 same label, 1 different)
  xted_u      – XTED_CPU default all-1s cost       (rename = 1 always)
  ref         – TEST_XTED_REF                      (rename = 0 same label, 1 different)
  zss_unit    – zss with unit costs                (rename = 0 same label, 1 different)

xted_label, ref, and zss_unit must always agree.
xted_u agrees only when no two matched nodes share the same label.
"""

import ast
import os
import time
import pytest
import zss
from xted import TEST_XTED_REF
from xted import x_ted_compute


# ── Helpers ───────────────────────────────────────────────────────────────────

def label_cost_matrix(labels1, labels2):
    return [[0 if labels1[i] == labels2[j] else 1
             for j in range(len(labels2))]
            for i in range(len(labels1))]


def xted_label(parent1, labels1, parent2, labels2, num_threads=1):
    """XTED_CPU with label-matching cost (comparable to ref and zss_unit)."""
    return x_ted_compute(parent1, labels1, parent2, labels2,
                         cost_matrix=label_cost_matrix(labels1, labels2),
                         num_threads=num_threads)


def xted_u(parent1, labels1, parent2, labels2, num_threads=1):
    """XTED_CPU with default all-1s cost."""
    return x_ted_compute(parent1, labels1, parent2, labels2, num_threads=num_threads)


def ref(parent1, labels1, parent2, labels2, num_threads=1):
    """Reference Zhang-Shasha with label-matching cost."""
    return TEST_XTED_REF.compute(labels1, parent1, labels2, parent2, num_threads)


def _build_zss(parent, labels):
    nodes = [zss.Node(labels[i]) for i in range(len(labels))]
    root = None
    for i, p in enumerate(parent):
        if p == -1:
            root = nodes[i]
        else:
            nodes[p].addkid(nodes[i])
    return root


def zss_unit(parent1, labels1, parent2, labels2):
    """ZSS with unit costs: insert=1, delete=1, rename=0 if same label, 1 otherwise."""
    a = _build_zss(parent1, labels1)
    b = _build_zss(parent2, labels2)
    return int(zss.distance(
        a, b,
        get_children=zss.Node.get_children,
        insert_cost=lambda n: 1,
        remove_cost=lambda n: 1,
        update_cost=lambda x, y: 0 if x.label == y.label else 1,
    ))


# ── Dataset loader ─────────────────────────────────────────────────────────────

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "Sampled_Dataset")


def _adj_to_parent(adj):
    """Convert an adjacency list to a flat parent-index array."""
    parent = [-1] * len(adj)
    for i, children in enumerate(adj):
        for c in children:
            parent[c] = i
    return parent


def load_trees(nodes_file, adj_file):
    with open(nodes_file) as f:
        labels = [line.split() for line in f if line.strip()]
    with open(adj_file) as f:
        adjs = [ast.literal_eval(line.strip()) for line in f if line.strip()]
    assert len(labels) == len(adjs)
    parents = [_adj_to_parent(adj) for adj in adjs]
    return parents, labels


# ── Test fixtures (DFS preorder indexed, flat parent arrays) ─────────────────
# parent[i] = index of node i's parent, -1 for root

# fmt: off
SINGLE_A    = ([-1], ['a'])
SINGLE_B    = ([-1], ['b'])
SINGLE_X    = ([-1], ['x'])

# a -> b -> c
CHAIN_3     = ([-1, 0, 1], ['a', 'b', 'c'])
# a -> b
CHAIN_2     = ([-1, 0], ['a', 'b'])

# a -> (b, c)
STAR_ABC    = ([-1, 0, 0], ['a', 'b', 'c'])
# a -> (b, d)  — one label differs
STAR_ABD    = ([-1, 0, 0], ['a', 'b', 'd'])
# x -> (y, z)  — all labels differ from STAR_ABC
STAR_XYZ    = ([-1, 0, 0], ['x', 'y', 'z'])

#   a
#  / \
# b   e        depth-2 tree, 5 nodes
# |\
# c d
DEEP        = ([-1, 0, 1, 1, 0], ['a', 'b', 'c', 'd', 'e'])
# Same structure, entirely different labels
DEEP_PRIME  = ([-1, 0, 1, 1, 0], ['x', 'y', 'z', 'w', 'v'])

# a -> (b -> c, d)  vs  a -> (b, d -> e)
ASYM_1      = ([-1, 0, 1, 0], ['a', 'b', 'c', 'd'])
ASYM_2      = ([-1, 0, 0, 2], ['a', 'b', 'd', 'e'])
# fmt: on

ALL_PAIRS = [
    (SINGLE_A, SINGLE_B),
    (SINGLE_A, SINGLE_X),
    (CHAIN_3, CHAIN_2),
    (STAR_ABC, STAR_ABD),
    (STAR_ABC, STAR_XYZ),
    (DEEP, DEEP_PRIME),
    (ASYM_1, ASYM_2),
    (CHAIN_3, DEEP),
]

# Pairs where all labels across both trees are unique: all-1s == label-cost.
DISJOINT_LABEL_PAIRS = [
    (STAR_ABC, STAR_XYZ),
    (DEEP, DEEP_PRIME),
]


# ── Section 1: known TED values (label-matching cost) ─────────────────────────

class TestKnownValues:
    def test_single_node_same_label(self):
        assert xted_label(*SINGLE_A, *SINGLE_A) == 0

    def test_single_node_different_label(self):
        assert xted_label(*SINGLE_A, *SINGLE_B) == 1

    def test_chain3_vs_chain2_delete_one(self):
        assert xted_label(*CHAIN_3, *CHAIN_2) == 1

    def test_star_one_rename(self):
        # STAR_ABC vs STAR_ABD: only 'c'→'d'
        assert xted_label(*STAR_ABC, *STAR_ABD) == 1

    def test_star_all_different_labels(self):
        # STAR_ABC vs STAR_XYZ: rename all 3 nodes
        assert xted_label(*STAR_ABC, *STAR_XYZ) == 3

    def test_deep_identical(self):
        assert xted_label(*DEEP, *DEEP) == 0

    def test_deep_all_different_labels(self):
        # DEEP vs DEEP_PRIME: rename all 5 nodes
        assert xted_label(*DEEP, *DEEP_PRIME) == 5

    def test_self_distance_is_zero(self):
        for tree in (SINGLE_A, CHAIN_3, STAR_ABC, DEEP):
            assert xted_label(*tree, *tree) == 0


# ── Section 2: XTED_CPU (label cost) == TEST_XTED_REF ─────────────────────────

class TestXTEDvsREF:
    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_agrees_with_ref(self, t1, t2):
        assert xted_label(*t1, *t2) == ref(*t1, *t2)

    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_agrees_with_ref_reversed(self, t1, t2):
        assert xted_label(*t2, *t1) == ref(*t2, *t1)

    def test_ref_self_distance_zero(self):
        for tree in (SINGLE_A, CHAIN_3, STAR_ABC, DEEP):
            assert ref(*tree, *tree) == 0


# ── Section 3: XTED_CPU (label cost) == ZSS unit cost ─────────────────────────

class TestXTEDvsZSS:
    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_agrees_with_zss(self, t1, t2):
        assert xted_label(*t1, *t2) == zss_unit(*t1, *t2)

    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_agrees_with_zss_reversed(self, t1, t2):
        assert xted_label(*t2, *t1) == zss_unit(*t2, *t1)


# ── Section 4: default (label-matching in C++) vs explicit label cost ─────────

class TestDefaultCost:
    def test_self_distance_is_zero(self):
        for tree in (SINGLE_A, CHAIN_3, STAR_ABC, DEEP):
            assert xted_u(*tree, *tree) == 0

    def test_same_label_single_node_costs_0(self):
        assert xted_u(*SINGLE_A, *SINGLE_A) == 0

    def test_different_label_single_node_costs_1(self):
        assert xted_u(*SINGLE_A, *SINGLE_B) == 1

    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_agrees_with_explicit_label_cost(self, t1, t2):
        # Default C++ label-matching must equal explicit Python label-matching cost
        assert xted_u(*t1, *t2) == xted_label(*t1, *t2)


# ── Section 5: symmetry ───────────────────────────────────────────────────────

class TestSymmetry:
    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_xted_label_symmetric(self, t1, t2):
        assert xted_label(*t1, *t2) == xted_label(*t2, *t1)

    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_xted_uniform_symmetric(self, t1, t2):
        assert xted_u(*t1, *t2) == xted_u(*t2, *t1)

    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_ref_symmetric(self, t1, t2):
        assert ref(*t1, *t2) == ref(*t2, *t1)

    @pytest.mark.parametrize("t1,t2", ALL_PAIRS)
    def test_zss_symmetric(self, t1, t2):
        assert zss_unit(*t1, *t2) == zss_unit(*t2, *t1)


# ── Section 6: Bolzano dataset cross-checks ───────────────────────────────────

@pytest.fixture(scope="module")
def bolzano():
    nodes_file = os.path.join(DATASET_ROOT, "3_Bolzano", "bolzano_nodes.txt")
    adj_file   = os.path.join(DATASET_ROOT, "3_Bolzano", "bolzano_nodes_adj.txt")
    return load_trees(nodes_file, adj_file)


class TestBolzanoDataset:
    # First 10 pairs against the reference implementation
    @pytest.mark.parametrize("i,j", [(i, j) for i in range(5) for j in range(i+1, 5)])
    def test_xted_vs_ref(self, bolzano, i, j):
        parents, labels = bolzano
        assert xted_label(parents[i], labels[i], parents[j], labels[j]) == \
               ref(parents[i], labels[i], parents[j], labels[j])

    @pytest.mark.parametrize("i,j", [(i, j) for i in range(5) for j in range(i+1, 5)])
    def test_xted_vs_zss(self, bolzano, i, j):
        parents, labels = bolzano
        assert xted_label(parents[i], labels[i], parents[j], labels[j]) == \
               zss_unit(parents[i], labels[i], parents[j], labels[j])

    def test_self_distance_zero(self, bolzano):
        parents, labels = bolzano
        for i in range(10):
            assert xted_label(parents[i], labels[i], parents[i], labels[i]) == 0
            assert ref(parents[i], labels[i], parents[i], labels[i]) == 0


# ── Section 7: Swissport-100 verified values (label-matching cost) ─────────────

@pytest.fixture(scope="module")
def swissport_100():
    nodes_file = os.path.join(DATASET_ROOT, "1_Swissport", "swissport_nodes_100.txt")
    adj_file   = os.path.join(DATASET_ROOT, "1_Swissport", "swissport_nodes_adj_100.txt")
    return load_trees(nodes_file, adj_file)


class TestSwissport100:
    def test_pair_0_1_xted(self, swissport_100):
        parents, labels = swissport_100
        assert xted_label(parents[0], labels[0], parents[1], labels[1]) == 44

    def test_pair_0_1_ref(self, swissport_100):
        parents, labels = swissport_100
        assert ref(parents[0], labels[0], parents[1], labels[1]) == 44

    def test_pair_0_1_zss(self, swissport_100):
        parents, labels = swissport_100
        assert zss_unit(parents[0], labels[0], parents[1], labels[1]) == 44

    def test_pair_0_1_all_implementations_agree(self, swissport_100):
        parents, labels = swissport_100
        a = parents[0]; la = labels[0]
        b = parents[1]; lb = labels[1]
        xted_val = xted_label(a, la, b, lb)
        assert ref(a, la, b, lb)      == xted_val
        assert zss_unit(a, la, b, lb) == xted_val


# ── Section 8: timing comparison ──────────────────────────────────────────────

REPS = 5  # repetitions per timing measurement


def _timed(fn, *args, reps=REPS):
    """Return (result, best_ms) — best wall time over `reps` runs."""
    best = float("inf")
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return result, best * 1000


def _print_table(title, rows):
    """rows: list of (impl_name, ted_value, time_ms)."""
    col = max(len(r[0]) for r in rows) + 2
    print(f"\n{title}")
    print(f"  {'impl':<{col}} {'TED':>5}  {'best (ms)':>10}")
    print(f"  {'-'*col} {'-'*5}  {'-'*10}")
    for name, ted, ms in rows:
        print(f"  {name:<{col}} {ted:>5}  {ms:>10.3f}")


@pytest.fixture(scope="module")
def swissport_500():
    nodes_file = os.path.join(DATASET_ROOT, "1_Swissport", "swissport_nodes_500.txt")
    adj_file   = os.path.join(DATASET_ROOT, "1_Swissport", "swissport_nodes_adj_500.txt")
    return load_trees(nodes_file, adj_file)


class TestTiming:
    """Timing comparison across implementations on progressively larger trees.
    Run with pytest -s to see the printed tables.
    """

    def test_timing_small_trees(self):
        a_adj, a_lab = DEEP
        b_adj, b_lab = DEEP_PRIME
        cost = label_cost_matrix(a_lab, b_lab)
        rows = [
            ("xted_uniform", *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab)),
            ("xted_label",   *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab, cost)),
            ("ref",          *_timed(ref,            a_adj, a_lab, b_adj, b_lab)),
            ("zss_unit",     *_timed(zss_unit,       a_adj, a_lab, b_adj, b_lab)),
        ]
        _print_table("Small trees (5 nodes each)", rows)
        assert len({r[1] for r in rows}) == 1, "All impls must agree on TED"

    def test_timing_bolzano(self, bolzano):
        parents, labels = bolzano
        a_adj, a_lab = parents[0], labels[0]
        b_adj, b_lab = parents[1], labels[1]
        cost = label_cost_matrix(a_lab, b_lab)
        rows = [
            ("xted_uniform", *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab)),
            ("xted_label",   *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab, cost)),
            ("ref",          *_timed(ref,            a_adj, a_lab, b_adj, b_lab)),
            ("zss_unit",     *_timed(zss_unit,       a_adj, a_lab, b_adj, b_lab)),
        ]
        _print_table(f"Bolzano pair (0,1) — {len(a_lab)} vs {len(b_lab)} nodes", rows)
        assert len({r[1] for r in rows}) == 1, "All impls must agree on TED"

    def test_timing_swissport_100(self, swissport_100):
        parents, labels = swissport_100
        a_adj, a_lab = parents[0], labels[0]
        b_adj, b_lab = parents[1], labels[1]
        cost = label_cost_matrix(a_lab, b_lab)
        rows = [
            ("xted_uniform", *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab)),
            ("xted_label",   *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab, cost)),
            ("ref",          *_timed(ref,            a_adj, a_lab, b_adj, b_lab)),
            ("zss_unit",     *_timed(zss_unit,       a_adj, a_lab, b_adj, b_lab)),
        ]
        _print_table(f"Swissport-100 pair (0,1) — {len(a_lab)} vs {len(b_lab)} nodes", rows)
        assert len({r[1] for r in rows}) == 1, "All impls must agree on TED"

    def test_timing_swissport_500(self, swissport_500):
        parents, labels = swissport_500
        a_adj, a_lab = parents[0], labels[0]
        b_adj, b_lab = parents[1], labels[1]
        cost = label_cost_matrix(a_lab, b_lab)
        rows = [
            ("xted_uniform", *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab)),
            ("xted_label",   *_timed(x_ted_compute, a_adj, a_lab, b_adj, b_lab, cost)),
            ("ref",          *_timed(ref,            a_adj, a_lab, b_adj, b_lab)),
            ("zss_unit",     *_timed(zss_unit,       a_adj, a_lab, b_adj, b_lab)),
        ]
        _print_table(f"Swissport-500 pair (0,1) — {len(a_lab)} vs {len(b_lab)} nodes", rows)
        assert len({r[1] for r in rows}) == 1, "All impls must agree on TED"
