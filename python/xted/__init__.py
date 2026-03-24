from . import XTED_CPU
import spacy
import numpy


def _to_list(x):
    """Convert numpy arrays to nested Python lists; pass through regular lists unchanged."""
    if isinstance(x, numpy.ndarray):
        return x.tolist()
    return x


def x_ted_compute(parent_1, label_1, parent_2, label_2, cost_matrix=None, num_threads=1):
    label_1 = _to_list(label_1)
    label_2 = _to_list(label_2)
    parent_1 = _to_list(parent_1)
    parent_2 = _to_list(parent_2)

    if len(label_1) == 0 or len(label_2) == 0:
        raise ValueError("Trees must be non-empty")
    if len(label_1) != len(parent_1):
        raise ValueError(f"label_1 length ({len(label_1)}) must match parent_1 length ({len(parent_1)})")
    if len(label_2) != len(parent_2):
        raise ValueError(f"label_2 length ({len(label_2)}) must match parent_2 length ({len(parent_2)})")
    if not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError(f"num_threads must be a positive integer, got {num_threads!r}")

    if cost_matrix is None:
        return XTED_CPU.compute_tree_edit_distance_uniform(label_1, parent_1, label_2, parent_2, num_threads)

    cost_matrix = _to_list(cost_matrix)
    if len(cost_matrix) != len(label_1):
        raise ValueError(f"cost_matrix rows ({len(cost_matrix)}) must match len(label_1) ({len(label_1)})")
    if any(len(row) != len(label_2) for row in cost_matrix):
        raise ValueError(f"Every cost_matrix row must have length len(label_2) ({len(label_2)})")

    return XTED_CPU.compute_tree_edit_distance(label_1, parent_1, label_2, parent_2, cost_matrix, num_threads)


def x_ted_util_transfer(text, nlp=None):
    # default NLP usage
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    # finds the root node
    root = next(token for token in doc if token.head == token)

    label = []
    adj = []

    def dfs(token):
        idx = len(label)
        label.append(token.text)
        adj.append([])
        for child in token.children:
            adj[idx].append(len(label))  # child's DFS preorder index
            dfs(child)

    dfs(root)
    return adj, label


def x_ted_compute_from_text(text1, text2, num_threads=1):
    parent1, label1 = x_ted_util_transfer(text1)
    parent2, label2 = x_ted_util_transfer(text2)
    return x_ted_compute(parent1, label1, parent2, label2, num_threads=num_threads)
