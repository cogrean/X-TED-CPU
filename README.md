# X-TED CPU

A parallel CPU implementation of the Zhang-Shasha Tree Edit Distance (TED) algorithm, exposed as a Python package via pybind11.

## Installation

```bash
pip install x-ted
```

Or from source (requires a C++ compiler and CMake (version 3.23)):

```bash
git clone https://github.com/cogrean/X-TED-CPU.git
cd X-TED-CPU
pip install .
```

For additional resources, see INSTALL.md

## Usage

```python
from xted import x_ted_compute

# Trees must be in DFS preorder. parent[i] = index of node i's parent (-1 for root).
#     a
#    / \
#   b   c
parent1 = [-1, 0, 0]
labels1 = ['a', 'b', 'c']

#   a
#   |
#   b
parent2 = [-1, 0]
labels2 = ['a', 'b']

print(x_ted_compute(parent1, labels1, parent2, labels2))  # 1 (delete c)
```

(Accepts Python lists and NumPy arrays)

### Custom cost matrix

By default, rename costs 0 for matching labels and 1 otherwise. Pass a custom `m×n` matrix to override:

```python
cost = [[0, 2],
        [1, 0]]

x_ted_compute(parent1, labels1, parent2, labels2, cost_matrix=cost)
```

### Multithreading

```python
x_ted_compute(parent1, labels1, parent2, labels2, num_threads=4)
```

### Batch computation

```python
from xted import x_ted_batch_compute

pairs = [
    (parent1, labels1, parent2, labels2),
    (parent3, labels3, parent4, labels4),
]

distances = x_ted_batch_compute(pairs)                          # auto-generates label-matching cost matrices
distances = x_ted_batch_compute(pairs, num_threads=4)           # with multithreading
distances = x_ted_batch_compute(pairs, cost_matrix=my_matrix)   # single matrix reused for all pairs
distances = x_ted_batch_compute(pairs, cost_matrix=[m1, m2])    # per-pair cost matrices
```

## Tree format

Nodes must be indexed in **DFS preorder** — the root is node 0 (with `parent[0] = -1`), and each node's subtree occupies a contiguous index range.

## Optional: NLP support

The core package (`pip install x-ted`) has no dependency on spaCy and works entirely with tree arrays. The optional `nlp` extra adds two convenience functions that automatically convert raw text into tree representations using spaCy's dependency parser:

- **`x_ted_compute_from_text(text1, text2)`** — Parses two strings into dependency trees and computes TED in one call.
- **`x_ted_util_transfer(text)`** — Converts a string into the `(parent, labels)` tree format used by `x_ted_compute`.

These functions require spaCy and a language model. To install:

```bash
pip install x-ted[nlp]
python -m spacy download en_core_web_sm
```

The `spacy download` step is required separately because spaCy language models are not standard PyPI packages.

### Usage

```python
from xted import x_ted_compute_from_text

# Compute TED directly from text (uses en_core_web_sm by default)
distance = x_ted_compute_from_text("The cat sat.", "A dog ran.")
```

To convert text into tree arrays for use with `x_ted_compute` directly:

```python
from xted import x_ted_util_transfer, x_ted_compute

parent1, labels1 = x_ted_util_transfer("The cat sat.")
parent2, labels2 = x_ted_util_transfer("A dog ran.")

distance = x_ted_compute(parent1, labels1, parent2, labels2)
```

You can also pass a custom spaCy model:

```python
import spacy
nlp = spacy.load("en_core_web_lg")

parent, labels = x_ted_util_transfer("The cat sat.", nlp=nlp)
distance = x_ted_compute_from_text("The cat sat.", "A dog ran.", nlp=nlp)
```
