# X-TED CPU

A parallel CPU implementation of the Zhang-Shasha Tree Edit Distance (TED) algorithm, exposed as a Python package via pybind11.

## Installation (coming soon)

```bash
pip install x-ted
```

Or from source (requires a C++ compiler and CMake (version 3.23)):

```bash
git clone https://github.com/cogrean/X-TED-CPU-.git
cd X-TED-CPU
pip install .
```

## Usage

```python
from xted import x_ted_compute

# Trees must be in DFS preorder. adj[i] = list of child indices for node i.
#     a
#    / \
#   b   c
adj1    = [[1, 2], [], []]
labels1 = ['a', 'b', 'c']

#   a
#   |
#   b
adj2    = [[1], []]
labels2 = ['a', 'b']

print(x_ted_compute(adj1, labels1, adj2, labels2))  # 1 (delete c)
```

(Accepts Python lists and NumPy arrays)

### Custom cost matrix

By default, rename costs 0 for matching labels and 1 otherwise. Pass a custom `m×n` matrix to override:

```python
cost = [[0, 2],
        [1, 0]]

x_ted_compute(adj1, labels1, adj2, labels2, cost_matrix=cost)
```

TO CLARIFY: Passed-in cost matrices must have a distance of 0 for labels that are matching.

### Multithreading

```python
x_ted_compute(adj1, labels1, adj2, labels2, num_threads=4)
```

(4 seems to be the optimal amount of threads for trees under ~500 nodes due to synchronization overhead)

### Batch computation

```python
from xted import x_ted_batch_compute

pairs = [
    (adj1, labels1, adj2, labels2),
    (adj3, labels3, adj4, labels4),
]

distances = x_ted_batch_compute(pairs)                          # auto-generates label-matching cost matrices
distances = x_ted_batch_compute(pairs, num_threads=4)           # with multithreading
distances = x_ted_batch_compute(pairs, cost_matrix=my_matrix)   # single matrix reused for all pairs
distances = x_ted_batch_compute(pairs, cost_matrix=[m1, m2])    # per-pair cost matrices
```

### From text (spaCy dependency parse trees)

```python
from xted import x_ted_compute_from_text

distance = x_ted_compute_from_text("The cat sat.", "A dog ran.")
```

Requires `spacy` and the `en_core_web_sm` model (`python -m spacy download en_core_web_sm`).

## Tree format

Nodes must be indexed in **DFS preorder** — the root is node 0, and each node's subtree occupies a contiguous index range. To convert your own tree:

```python
adj, label = x_ted_util_transfer(text, nlp (optional))
```

Returns a Python List object.
