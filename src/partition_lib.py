"""
partition_lib.py
================
Core library for threshold partitions and the threshold distance index delta(G).

The threshold distance delta(G) of a graph G on n vertices is defined as:
    delta(G) = min_{tau in TP(n)} ||d(G) - tau||_2
where d(G) is the degree partition of G (sorted descending) and TP(n) is
the set of 2^{n-1} threshold partitions on n vertices.

Reference:
    King (2024), "A Geometric Topological Index via Threshold Partitions:
    Ferrers Diagrams and Molecular Graphs", MATCH Commun. Math. Comput. Chem.
"""

from itertools import product as _product
from math import sqrt


# ---------------------------------------------------------------------------
# Threshold partition enumeration
# ---------------------------------------------------------------------------

def threshold_partitions(n: int) -> list[tuple[int, ...]]:
    """
    Return all 2^{n-1} threshold partitions of length n.

    A threshold partition is the degree sequence of a threshold graph on n
    vertices. Threshold graphs are built by a binary creation sequence
    b = (b_1, ..., b_{n-1}) in {0,1}^{n-1}: starting from a single vertex,
    step b_k = 1 adds a *dominating* vertex (adjacent to all existing vertices)
    and b_k = 0 adds an *isolated* vertex.

    Parameters
    ----------
    n : int
        Number of vertices (n >= 1).

    Returns
    -------
    list of tuples
        Distinct threshold partitions, each of length n, sorted descending.
        Duplicates arising from different creation sequences are removed.
    """
    if n == 1:
        return [(0,)]

    seen = set()
    result = []
    for bits in _product([0, 1], repeat=n - 1):
        degrees = [0] * n
        for step, b in enumerate(bits):
            new_v = step + 1
            if b == 1:          # dominating step
                degrees[new_v] = step + 1
                for prev in range(new_v):
                    degrees[prev] += 1
            # b == 0: isolated step, degree stays 0
        tau = tuple(sorted(degrees, reverse=True))
        if tau not in seen:
            seen.add(tau)
            result.append(tau)
    return result


# ---------------------------------------------------------------------------
# Distance and delta
# ---------------------------------------------------------------------------

def squared_distance(tau: tuple[int, ...], d: tuple[int, ...]) -> int:
    """
    Return the squared Euclidean distance ||d - tau||^2.

    Both sequences must have the same length. Entries are assumed to be
    non-negative integers.
    """
    return sum((a - b) ** 2 for a, b in zip(d, tau))


def delta_squared(d: tuple[int, ...]) -> int:
    """
    Return delta(G)^2 = min_{tau in TP(n)} ||d - tau||^2.

    Parameters
    ----------
    d : tuple of int
        Degree partition of G, length n, sorted in non-increasing order.
        Must be a valid graphical partition.

    Returns
    -------
    int
        The squared threshold distance. Always a non-negative even integer
        (Parity Lemma: delta(G)^2 is always even).
    """
    n = len(d)
    tps = threshold_partitions(n)
    return min(squared_distance(tau, d) for tau in tps)


def delta(d: tuple[int, ...]) -> float:
    """
    Return delta(G) = sqrt(delta_squared(d)).

    Parameters
    ----------
    d : tuple of int
        Degree partition of G, sorted descending.

    Returns
    -------
    float
    """
    return sqrt(delta_squared(d))


def nearest_threshold_partitions(d: tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Return all threshold partitions achieving the minimum distance to d.

    There may be multiple minimisers; all are returned.

    Parameters
    ----------
    d : tuple of int
        Degree partition, sorted descending.

    Returns
    -------
    list of tuples
        All tau* in TP(n) with ||d - tau*||^2 = delta(G)^2.
    """
    n = len(d)
    tps = threshold_partitions(n)
    best = min(squared_distance(tau, d) for tau in tps)
    return [tau for tau in tps if squared_distance(tau, d) == best]


# ---------------------------------------------------------------------------
# Degree sequence utilities
# ---------------------------------------------------------------------------

def degree_partition(graph: dict[int, set[int]]) -> tuple[int, ...]:
    """
    Return the degree partition of a graph given as an adjacency dict.

    Parameters
    ----------
    graph : dict mapping vertex -> set of neighbours
        Vertices must be 0-indexed integers.

    Returns
    -------
    tuple of int
        Degrees sorted in non-increasing order.
    """
    degs = tuple(sorted((len(nbrs) for nbrs in graph.values()), reverse=True))
    return degs


def degree_partition_from_smiles(smiles: str) -> tuple[int, ...] | None:
    """
    Return the carbon-skeleton degree partition from a SMILES string.

    Requires RDKit. Returns None if RDKit is unavailable or the SMILES
    is invalid.

    Parameters
    ----------
    smiles : str
        SMILES string of an organic molecule.

    Returns
    -------
    tuple of int or None
        Carbon skeleton degree partition (sorted descending), or None.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    carbon_degs = sorted(
        [
            sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6)
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 6
        ],
        reverse=True,
    )
    return tuple(carbon_degs)


# ---------------------------------------------------------------------------
# Ferrers diagram utilities
# ---------------------------------------------------------------------------

def conjugate_partition(d: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the conjugate (transpose) of the Ferrers diagram of d.

    The conjugate partition d* satisfies d*_j = |{i : d_i >= j}|.

    Parameters
    ----------
    d : tuple of int
        Partition (non-negative integers, sorted descending or unsorted).

    Returns
    -------
    tuple of int
        Conjugate partition, sorted descending.
    """
    if not d or max(d) == 0:
        return ()
    max_val = max(d)
    return tuple(sum(1 for x in d if x >= j) for j in range(1, max_val + 1))


def ferrers_diagram(d: tuple[int, ...]) -> str:
    """
    Return a text representation of the Ferrers diagram of d.

    Parameters
    ----------
    d : tuple of int
        Partition, sorted descending.

    Returns
    -------
    str
        ASCII Ferrers diagram with rows of '■' symbols.
    """
    return "\n".join("■ " * k for k in d if k > 0)


# ---------------------------------------------------------------------------
# Support size and parity checks
# ---------------------------------------------------------------------------

def support_size(tau: tuple[int, ...]) -> int:
    """Return the number of positive entries in tau."""
    return sum(1 for x in tau if x > 0)


def is_even(x: int) -> bool:
    """Return True if x is even."""
    return x % 2 == 0


def verify_parity_lemma(d: tuple[int, ...]) -> bool:
    """
    Verify the Parity Lemma: delta(G)^2 is always even.

    Parameters
    ----------
    d : tuple of int
        Degree partition.

    Returns
    -------
    bool
        True if delta_squared(d) is even (should always be True).
    """
    return is_even(delta_squared(d))
