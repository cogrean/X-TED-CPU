from . import XTED_CPU


def _to_list(x):
    """Convert numpy arrays to nested Python lists; pass through regular lists unchanged.

    Args:
        x (list or numpy.ndarray): The input to convert.

    Returns:
        list: A nested Python list representation of x's data.
    """
    try:
        import numpy
    except ImportError:
        return x
    if isinstance(x, numpy.ndarray):
        return x.tolist()
    return x

def _validate_parent_array(parent, name):
    n = len(parent)
    if parent[0] != -1:
        raise ValueError(f"{name}[0] must be -1 (root), got {parent[0]}")
    for i in range(1, n):
        p = parent[i]
        if p < 0 or p >= n:
            raise ValueError(f"{name}[{i}] = {p} is out of range [0, {n})")
        if p >= i:
            raise ValueError(
                f"{name}[{i}] = {p} is invalid: parent index must be less than "
                f"child index in DFS preorder")


def x_ted_compute(parent_1, label_1, parent_2, label_2, cost_matrix=None, num_threads=1):
    """Compute the tree edit distance between two trees.

    Sanitizes the inputs and passes them to an efficient multi-threaded
    C++ implementation of the X-TED algorithm.

    Args:
        parent_1 (list or numpy.ndarray): Flat parent-index array for the first tree.
            ``parent_1[i]`` is the index of node i's parent, with -1 for the root.
            Nodes must be ordered in DFS preorder.
        label_1 (list or numpy.ndarray): Node labels for the first tree. Ordered in DFS preorder.
        parent_2 (list or numpy.ndarray): Flat parent-index array for the second tree.
            ``parent_2[i]`` is the index of node i's parent, with -1 for the root.
            Nodes must be ordered in DFS preorder.
        label_2 (list or numpy.ndarray): Node labels for the second tree. Ordered in DFS preorder.
        cost_matrix (list or numpy.ndarray, optional): An ``m x n`` matrix where
            ``cost_matrix[i][j]`` is the rename cost from node i of tree 1 to node j
            of tree 2. If ``None``, a uniform cost matrix is used (1 for non-matching
            labels, 0 for matching).
        num_threads (int, optional): Maximum number of threads for parallel computation.
            Defaults to 1.

    Returns:
        int: The tree edit distance between the two input trees.

    Raises:
        ValueError: If trees are empty, label/parent lengths mismatch, cost_matrix
            dimensions are incorrect, or num_threads is not a positive integer.
    """
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
    _validate_parent_array(parent_1, "parent_1")
    _validate_parent_array(parent_2, "parent_2")

    if cost_matrix is None:
        return XTED_CPU.compute_tree_edit_distance_uniform(label_1, parent_1, label_2, parent_2, num_threads)

    cost_matrix = _to_list(cost_matrix)
    if len(cost_matrix) != len(label_1):
        raise ValueError(f"cost_matrix rows ({len(cost_matrix)}) must match len(label_1) ({len(label_1)})")
    if any(len(row) != len(label_2) for row in cost_matrix):
        raise ValueError(f"Every cost_matrix row must have length len(label_2) ({len(label_2)})")

    return XTED_CPU.compute_tree_edit_distance(label_1, parent_1, label_2, parent_2, cost_matrix, num_threads)

def x_ted_util_transfer(text, nlp=None):
    """Parse text into a parent-index array and label array for TED computation.

    Uses spaCy dependency parsing to build a tree from the input text,
    then returns it in DFS preorder format suitable for ``x_ted_compute``.

    Args:
        text (str): The input text to parse.
        nlp (spacy.Language, optional): A spaCy language model. If ``None``,
            loads ``en_core_web_sm``.

    Returns:
        tuple: A tuple of ``(parent, label)`` where:
            - ``parent`` (list of int): Flat parent-index array in DFS preorder.
              ``parent[i]`` is the index of node i's parent, -1 for root.
            - ``label`` (list of str): Node labels in DFS preorder.
    """
    # default NLP usage
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    # finds the root node
    root = next(token for token in doc if token.head == token)

    label = []
    parent = []

    def dfs(token, par=-1):
        label.append(token.text)
        parent.append(par)
        idx = len(label) - 1
        for child in token.children:
            dfs(child, idx)

    dfs(root)
    return parent, label


def _default_cost_matrix(labels1, labels2):
    """Generate a label-matching cost matrix: 0 if labels match, 1 otherwise."""
    return [[0 if labels1[i] == labels2[j] else 1
             for j in range(len(labels2))]
            for i in range(len(labels1))]


def x_ted_batch_compute(tree_pairs, cost_matrix=None, num_threads=1):
    """Compute tree edit distance for a batch of tree pairs.

    Args:
        tree_pairs (list of tuple): Each element is a tuple of
            ``(parent1, labels1, parent2, labels2)``.
        cost_matrix (list, optional): If ``None``, a label-matching cost matrix
            is generated per pair (0 for matching labels, 1 otherwise). Pass a
            single ``m x n`` list to reuse the same matrix for every pair, or a
            list of matrices (one per pair).
        num_threads (int, optional): Maximum number of threads passed to the
            C++ kernel for each pair. Defaults to 1.

    Returns:
        list of int: The tree edit distance for each pair.

    Raises:
        ValueError: If any element of tree_pairs does not have exactly 4
            elements, or if num_threads is not a positive integer.
    """
    if not tree_pairs:
        return []

    if not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError(f"num_threads must be a positive integer, got {num_threads!r}")

    results = []
    for idx, pair in enumerate(tree_pairs):
        if len(pair) != 4:
            raise ValueError(f"tree_pairs[{idx}] must have 4 elements (parent1, labels1, parent2, labels2), got {len(pair)}")
        parent1, labels1, parent2, labels2 = pair

        if cost_matrix is None:
            cm = _default_cost_matrix(labels1, labels2)
        elif isinstance(cost_matrix, list) and len(cost_matrix) > 0 and isinstance(cost_matrix[0], list) and not isinstance(cost_matrix[0][0], list):
            cm = cost_matrix
        else:
            cm = cost_matrix[idx]

        results.append(x_ted_compute(parent1, labels1, parent2, labels2, cost_matrix=cm, num_threads=num_threads))

    return results


def x_ted_compute_from_text(text1, text2, nlp=None, num_threads=1):
    """Compute tree edit distance directly from two text strings.

    Parses each string into a dependency tree using spaCy, then computes
    the tree edit distance with a uniform cost matrix.

    Args:
        text1 (str): The first input text.
        text2 (str): The second input text.
        num_threads (int, optional): Maximum number of threads for parallel
            computation. Defaults to 1.

    Returns:
        int: The tree edit distance between the two parsed trees.
    """
    
    import spacy
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    parent1, label1 = x_ted_util_transfer(text1, nlp=nlp)
    parent2, label2 = x_ted_util_transfer(text2, nlp=nlp)
    return x_ted_compute(parent1, label1, parent2, label2, num_threads=num_threads)