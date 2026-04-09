from xted import x_ted_compute, x_ted_compute_from_text
import zss
import spacy
import ast
import time


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_dataset(nodes_path, adj_path, tree_idx=0):
    """Load a single tree from a dataset file pair.
    Returns (parent, labels) in DFS preorder format.
    """
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

    # adj -> parent array
    n = len(adj)
    parent = [-1] * n
    for i in range(n):
        for child in adj[i]:
            parent[child] = i

    return parent, labels


def build_zss_tree(parent, labels):
    """Build a zss.Node tree from parent/label arrays."""
    n = len(parent)
    nodes = [zss.Node(labels[i]) for i in range(n)]
    for i in range(1, n):
        nodes[parent[i]].addkid(nodes[i])
    return nodes[0]


def compare(name, parent1, labels1, parent2, labels2, num_threads=1):
    """Run both X-TED and Zhang-Shasha, print results and timings."""
    # X-TED
    t0 = time.perf_counter()
    xted_result = x_ted_compute(parent1, labels1, parent2, labels2, num_threads=num_threads)
    t1 = time.perf_counter()
    xted_ms = (t1 - t0) * 1000

    # Zhang-Shasha (zss)
    zss_tree1 = build_zss_tree(parent1, labels1)
    zss_tree2 = build_zss_tree(parent2, labels2)
    t0 = time.perf_counter()
    zss_result = int(zss.simple_distance(zss_tree1, zss_tree2))
    t1 = time.perf_counter()
    zss_ms = (t1 - t0) * 1000

    match = "MATCH" if xted_result == zss_result else "MISMATCH"
    print(f"  {name}")
    print(f"    X-TED:          {xted_result:>6}   ({xted_ms:>10.3f} ms)")
    print(f"    Zhang-Shasha:   {zss_result:>6}   ({zss_ms:>10.3f} ms)")
    print(f"    {match}")
    if xted_ms > 0:
        print(f"    Speedup: {zss_ms / xted_ms:.1f}x")
    print()


# ---------------------------------------------------------------------------
# 1. Small hand-crafted trees
# ---------------------------------------------------------------------------
print("=" * 60)
print("Small hand-crafted trees")
print("=" * 60)

label1 = ["a", "b", "c"]
label2 = ["a", "b"]
parent1 = [-1, 0, 0]
parent2 = [-1, 0]

compare("3-node vs 2-node", parent1, label1, parent2, label2)


# ---------------------------------------------------------------------------
# 2. spaCy-parsed sentences
# ---------------------------------------------------------------------------
print("=" * 60)
print("spaCy-parsed sentences")
print("=" * 60)

string1 = "I am lost but I will be found. Help me. Help me. Help me."
string2 = "Lorem ipsum is an unpleasant tool to use. We must be rid of it."

nlp = spacy.load("en_core_web_sm")

from xted import x_ted_util_transfer
parent1, label1 = x_ted_util_transfer(string1, nlp=nlp)
parent2, label2 = x_ted_util_transfer(string2, nlp=nlp)

print(f"  Tree 1 ({len(label1)} nodes): {string1}")
print(f"  Tree 2 ({len(label2)} nodes): {string2}")
print()

compare("sentence TED", parent1, label1, parent2, label2)


# ---------------------------------------------------------------------------
# 3. Dataset trees (100-node)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Dataset trees (~100 nodes)")
print("=" * 60)

datasets_100 = [
    ("Swissport 100", "Sampled_Dataset/1_Swissport/swissport_nodes_100.txt",
                       "Sampled_Dataset/1_Swissport/swissport_nodes_adj_100.txt"),
    ("Python 100",     "Sampled_Dataset/2_Python/python_nodes_100.txt",
                       "Sampled_Dataset/2_Python/python_nodes_adj_100.txt"),
    ("DBLP 100",       "Sampled_Dataset/4_DBLP/dblp_nodes_100.txt",
                       "Sampled_Dataset/4_DBLP/dblp_nodes_adj_100.txt"),
]

for name, nodes_path, adj_path in datasets_100:
    p1, l1 = load_dataset(nodes_path, adj_path, tree_idx=0)
    p2, l2 = load_dataset(nodes_path, adj_path, tree_idx=1)
    compare(f"{name}  (tree 0 vs 1, {len(l1)} vs {len(l2)} nodes)", p1, l1, p2, l2)


# ---------------------------------------------------------------------------
# 4. Dataset trees (500-node)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Dataset trees (~500 nodes)")
print("=" * 60)

datasets_500 = [
    ("Swissport 500", "Sampled_Dataset/1_Swissport/swissport_nodes_500.txt",
                       "Sampled_Dataset/1_Swissport/swissport_nodes_adj_500.txt"),
    ("Python 500",     "Sampled_Dataset/2_Python/python_nodes_500.txt",
                       "Sampled_Dataset/2_Python/python_nodes_adj_500.txt"),
    ("DBLP 500",       "Sampled_Dataset/4_DBLP/dblp_nodes_500.txt",
                       "Sampled_Dataset/4_DBLP/dblp_nodes_adj_500.txt"),
]

for name, nodes_path, adj_path in datasets_500:
    p1, l1 = load_dataset(nodes_path, adj_path, tree_idx=0)
    p2, l2 = load_dataset(nodes_path, adj_path, tree_idx=1)
    compare(f"{name}  (tree 0 vs 1, {len(l1)} vs {len(l2)} nodes)", p1, l1, p2, l2)
