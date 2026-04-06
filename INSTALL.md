# Installation

## From PyPI (recommended)

```bash
pip install x-ted
```

Precompiled wheels are available for the following platforms:

Linux x86_64 & ARM64 w/ Python 3.9 - 3.13
macOS ARM64 (Apple Silicon) w/ Python 3.9 - 3.13
Windows x86_64 w/ Python 3.9 - 3.13

## From source

Building from source requires:

- A C++20-compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)
- CMake 3.23+
- Python 3.9+

```bash
git clone https://github.com/cogrean/X-TED-CPU.git
cd X-TED-CPU
pip install .
```

## Dependencies

Installed automatically with the package:

- **numpy** >= 1.21
- **spacy** >= 3.7.0, < 3.8.0
- **en-core-web-sm** 3.7.1 (spaCy English language model)

## Verifying the installation

```python
from xted import x_ted_compute

adj1    = [[1, 2], [], []]
labels1 = ['a', 'b', 'c']

adj2    = [[1], []]
labels2 = ['a', 'b']

print(x_ted_compute(adj1, labels1, adj2, labels2))  # 1
```
