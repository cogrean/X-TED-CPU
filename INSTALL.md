# Installation

## From PyPI (recommended)

```bash
pip install x-ted
```

Precompiled wheels are available for the following platforms:

Linux x86_64 & ARM64 w/ Python 3.9 - 3.12
macOS ARM64 (Apple Silicon) w/ Python 3.9 - 3.12
Windows x86_64 w/ Python 3.9 - 3.12

## From source

Building from source requires:

- A C++20-compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.23+
- Python 3.9-3.12

```bash
git clone https://github.com/cogrean/X-TED-CPU.git
cd X-TED-CPU
pip install .
```

## Dependencies

Installed automatically with the package:

- **numpy** >= 1.21

### Optional: NLP support (spaCy)

The core package works entirely with tree arrays and does not require spaCy. If you want to compute tree edit distance directly from raw text, the optional `nlp` extra provides two convenience functions:

- **`x_ted_compute_from_text`** — Parses two strings into dependency trees and computes TED in one call.
- **`x_ted_util_transfer`** — Converts a string into the `(parent, labels)` tree format used by `x_ted_compute`.

To install spaCy and download the required language model:

```bash
pip install x-ted[nlp]
python -m spacy download en_core_web_sm
```

The `spacy download` step is required separately because spaCy language models are not standard PyPI packages. The default model is `en_core_web_sm`, but any spaCy model can be passed via the `nlp` parameter.

## Verifying the installation

```python
from xted import x_ted_compute

parent1 = [-1, 0, 0]
labels1 = ['a', 'b', 'c']

parent2 = [-1, 0]
labels2 = ['a', 'b']

print(x_ted_compute(parent1, labels1, parent2, labels2))  # 1
```
