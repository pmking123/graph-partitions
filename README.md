# Threshold Distance Index — Code Repository

Code accompanying:

> King, P. (2024). *A Geometric Topological Index via Threshold Partitions:
> Ferrers Diagrams and Molecular Graphs*.
> MATCH Communications in Mathematical and Computer Chemistry.

## Overview

This repository implements the **threshold distance** δ(G) for molecular graphs,
defined as the Euclidean distance from the degree partition d(G) to the nearest
threshold partition in TP(n):

    δ(G) = min_{τ ∈ TP(n)} ‖d(G) − τ‖₂

## Repository structure

```
src/
    partition_lib.py       Core library: TP(n) enumeration, δ computation
    alkanes.py             Alkane degree sequences, theorem verification
    cyclic_molecules.py    Cyclic molecule library, cycle formula verification
    correlation_analysis.py  BP and logP correlation analysis

tests/
    test_partition_lib.py  Pytest unit tests

results/                   (empty; populated by running the scripts)
```

## Installation

```bash
pip install rdkit scipy numpy pytest
```

RDKit is required for SMILES-based input and logP computation.
All core mathematical functions (δ computation, theorem verification) work
without RDKit.

## Quick start

```bash
# Compute delta for all alkane degree sequences up to n=15
python src/alkanes.py

# Verify all theorems (n <= 15)
python src/alkanes.py --verify

# Compute delta for all built-in cyclic molecules
python src/cyclic_molecules.py

# Verify cycle formula delta(C_n)^2 = 4n-14
python src/cyclic_molecules.py --verify-cycles

# Run full correlation analysis (C1-C15)
python src/correlation_analysis.py

# Single molecule
python src/cyclic_molecules.py --smiles "CCC(C)CCC"

# Run tests
pytest tests/
```

## Key results reproduced

| Result | Script | Function |
|--------|--------|----------|
| Theorem 1: δ(C_n)² = 4n−14 | `cyclic_molecules.py --verify-cycles` | `verify_cycle_formula()` |
| Theorem 3: δ(T)² ≥ 4 when p ≥ 4 | `alkanes.py --verify` | `verify_theorem3()` |
| Theorem 4: δ(P_n)² = 4(n−5) | `alkanes.py --verify` | `verify_path_formula()` |
| Table 2 (correlations) | `correlation_analysis.py` | `run_analysis()` |
| Table 3 (partial correlations) | `correlation_analysis.py` | `run_analysis()` |

## Python API

```python
from src.partition_lib import delta_squared, delta, threshold_partitions

# Compute delta for 3-methylhexane carbon skeleton
d = (3, 2, 2, 2, 1, 1, 1)
print(delta_squared(d))   # 4
print(delta(d))           # 2.0

# All nearest threshold partitions
from src.partition_lib import nearest_threshold_partitions
for tau in nearest_threshold_partitions(d):
    print(tau)

# From SMILES
from src.partition_lib import degree_partition_from_smiles
d = degree_partition_from_smiles("CCC(C)CCC")
print(delta_squared(d))
```

## Notes

- δ(G)² is always **even** (Parity Lemma: proved in the paper).
- TP(n) has 2^{n-1} elements; exact enumeration is feasible for n ≤ 18
  (|TP(18)| = 131072). For n > 18, beam search or branch-and-bound is needed.
- All theorems are verified computationally for n ≤ 13 (Theorem 3),
  n ≤ 15 (Theorem 4), and n ≤ 20 (Theorem 1).
- δ depends only on the degree sequence, not the full graph structure.
  Isomers sharing a carbon skeleton degree sequence (e.g. anthracene and
  phenanthrene) have identical δ.

## Citing

```bibtex
@article{king2024threshold,
  title   = {A Geometric Topological Index via Threshold Partitions:
             Ferrers Diagrams and Molecular Graphs},
  author  = {King, Paul},
  journal = {MATCH Communications in Mathematical and Computer Chemistry},
  year    = {Submitted}
}
```

## Licence

MIT License

Copyright (c) 2026 Paul M. King

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
