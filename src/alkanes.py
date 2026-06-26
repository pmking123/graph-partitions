"""
alkanes.py
==========
Enumerate alkane carbon-skeleton degree sequences and compute delta.

Alkanes are acyclic (tree) molecular graphs with all vertex degrees in {1,2,3,4}
(reflecting the valence-4 constraint of carbon). The carbon skeleton is a tree
on n vertices where the maximum degree is at most 4.

This module provides:
- Enumeration of all unique alkane degree sequences for n=1..N
- Computation of delta for each degree sequence
- Summary statistics and verification of Theorem 3

Usage
-----
    python alkanes.py              # prints summary table n=1..15
    python alkanes.py --n 12       # up to n=12
    python alkanes.py --verify     # verify all theorems
"""

import argparse
import math
from collections import defaultdict

from partition_lib import (
    threshold_partitions,
    squared_distance,
    delta_squared,
    delta,
    nearest_threshold_partitions,
    support_size,
)


# ---------------------------------------------------------------------------
# Enumeration of alkane degree sequences
# ---------------------------------------------------------------------------

def alkane_degree_sequences(n: int) -> list[tuple[int, ...]]:
    """
    Return all graphical tree degree sequences on n vertices with max degree <= 4.

    These correspond precisely to the carbon skeleton degree sequences of
    alkane isomers C_n H_{2n+2}.

    Parameters
    ----------
    n : int
        Number of carbon atoms (vertices). n >= 1.

    Returns
    -------
    list of tuples
        Distinct degree sequences, each sorted descending. Empty list for n=1
        (methane has a single carbon with no C-C bonds, degree sequence (0,)).
    """
    if n == 1:
        return [(0,)]  # methane: single vertex, degree 0

    target = 2 * (n - 1)   # sum of degrees for a tree on n vertices
    result = set()

    def _partitions(total, k, lo=1, hi=None):
        """Generate partitions of `total` into `k` parts in [lo, hi]."""
        hi = hi or min(4, total)
        if k == 1:
            if lo <= total <= hi:
                yield (total,)
            return
        for first in range(lo, min(hi, total - (k - 1) * lo) + 1):
            for rest in _partitions(total - first, k - 1, lo, first):
                yield rest + (first,)

    for p in _partitions(target, n, 1, 4):
        d = tuple(sorted(p, reverse=True))
        # Erdős–Gallai check (necessary and sufficient for graphicality)
        if _erdos_gallai(d):
            result.add(d)

    return sorted(result, reverse=True)


def _erdos_gallai(d: tuple[int, ...]) -> bool:
    """Return True if d satisfies the Erdős–Gallai conditions."""
    n = len(d)
    for k in range(1, n + 1):
        lhs = sum(d[:k])
        rhs = k * (k - 1) + sum(min(d[i], k) for i in range(k, n))
        if lhs > rhs:
            return False
    return True


def all_alkane_degree_sequences(max_n: int = 15) -> dict[int, list[tuple[int, ...]]]:
    """
    Return a dict mapping n -> list of alkane degree sequences for n=1..max_n.

    Parameters
    ----------
    max_n : int
        Maximum carbon count. Default 15.

    Returns
    -------
    dict
        Keys are carbon counts n; values are sorted lists of degree sequences.
    """
    return {n: alkane_degree_sequences(n) for n in range(1, max_n + 1)}


# ---------------------------------------------------------------------------
# Classification of tree types (Theorem 3 families)
# ---------------------------------------------------------------------------

def is_star(d: tuple[int, ...]) -> bool:
    """Return True if d is the degree sequence of the star K_{1,n-1} or methane (n=1)."""
    n = len(d)
    if n == 1:
        return True   # methane: single vertex, trivially threshold
    return n >= 2 and d == tuple([n - 1] + [1] * (n - 1))


def is_asymmetric_double_star(d: tuple[int, ...]) -> bool:
    """
    Return True if d is the degree sequence of an asymmetric double-star DS(n-2,1).

    DS(n-2,1) has degree sequence (n-2, 2, 1, ..., 1).
    It consists of two adjacent internal vertices with degrees n-2 and 2.
    """
    n = len(d)
    if n < 4:
        return False
    return (
        d[0] == n - 2
        and d[1] == 2
        and all(x == 1 for x in d[2:])
    )


def is_lollipop(d: tuple[int, ...]) -> bool:
    """
    Return True if d is the degree sequence of a lollipop L_n.

    L_n has degree sequence (n-3, 2, 2, 1, ..., 1).
    It consists of three internal vertices in a path, with pendant carbons
    on the hub.
    """
    n = len(d)
    if n < 5:
        return False
    return (
        d[0] == n - 3
        and d[1] == 2
        and d[2] == 2
        and all(x == 1 for x in d[3:])
    )


def classify_tree(d: tuple[int, ...]) -> str:
    """
    Classify a tree degree sequence according to Theorem 3 families.

    Returns
    -------
    str
        One of: 'star', 'asymmetric_double_star', 'lollipop', 'general'
    """
    if is_star(d):
        return "star"
    if is_asymmetric_double_star(d):
        return "asymmetric_double_star"
    if is_lollipop(d):
        return "lollipop"
    return "general"


# ---------------------------------------------------------------------------
# Delta computation with caching
# ---------------------------------------------------------------------------

_tp_cache: dict[int, list[tuple[int, ...]]] = {}


def _get_tp(n: int) -> list[tuple[int, ...]]:
    """Return TP(n) with caching."""
    if n not in _tp_cache:
        _tp_cache[n] = threshold_partitions(n)
    return _tp_cache[n]


def compute_delta_table(max_n: int = 15) -> list[dict]:
    """
    Compute delta for all unique alkane degree sequences up to n=max_n.

    Returns
    -------
    list of dicts, each with keys:
        n          : int   carbon count
        d          : tuple degree sequence
        ell        : int   number of pendant vertices
        p          : int   number of internal vertices (n - ell)
        delta2     : int   delta(T)^2
        delta      : float delta(T)
        family     : str   tree family classification
        tau_star   : list  all nearest threshold partitions
        s_star     : int   support size of nearest tau (first achiever)
    """
    rows = []
    seqs_by_n = all_alkane_degree_sequences(max_n)

    for n, seqs in seqs_by_n.items():
        tps = _get_tp(n)
        for d in seqs:
            ell = d.count(1)
            p = n - ell
            d2 = min(squared_distance(tau, d) for tau in tps)
            achievers = [tau for tau in tps if squared_distance(tau, d) == d2]
            rows.append(
                {
                    "n": n,
                    "d": d,
                    "ell": ell,
                    "p": p,
                    "delta2": d2,
                    "delta": math.sqrt(d2),
                    "family": classify_tree(d),
                    "tau_star": achievers,
                    "s_star": support_size(achievers[0]),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Theorem verification
# ---------------------------------------------------------------------------

def verify_theorem3(max_n: int = 15) -> dict:
    """
    Verify Theorem 3 for all alkane degree sequences up to n=max_n.

    Theorem 3 states:
      (i)  delta(T) = 0 iff T is a star.
      (ii) delta(T)^2 = 2 iff T is an asymmetric double-star or lollipop.
      (iii) delta(T)^2 >= 4 for all other trees with p = n-ell >= 4.
      Corollary: delta(T)^2 >= n-ell-3 for all trees.

    Returns
    -------
    dict with keys 'passed', 'failures', 'n_tested', 'n_sequences'
    """
    rows = compute_delta_table(max_n)
    failures = []

    for r in rows:
        n, d, ell, p = r["n"], r["d"], r["ell"], r["p"]
        d2 = r["delta2"]
        fam = r["family"]

        # Part (i): delta=0 iff star
        if (d2 == 0) != (fam == "star"):
            failures.append(
                {"check": "part_i", "n": n, "d": d, "delta2": d2, "family": fam}
            )

        # Part (ii): delta^2=2 iff asymmetric double-star or lollipop
        if (d2 == 2) != (fam in ("asymmetric_double_star", "lollipop")):
            failures.append(
                {"check": "part_ii", "n": n, "d": d, "delta2": d2, "family": fam}
            )

        # Part (iii): delta^2 >= 4 when p >= 4 and not exceptional
        if p >= 4 and fam == "general" and d2 < 4:
            failures.append(
                {"check": "part_iii", "n": n, "d": d, "delta2": d2, "family": fam}
            )

        # Corollary: delta^2 >= n-ell-3
        lb = n - ell - 3
        if d2 < lb:
            failures.append(
                {"check": "corollary", "n": n, "d": d, "delta2": d2, "lb": lb}
            )

        # Parity lemma: delta^2 is always even
        if d2 % 2 != 0:
            failures.append(
                {"check": "parity", "n": n, "d": d, "delta2": d2}
            )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_tested": max_n,
        "n_sequences": len(rows),
    }


def verify_path_formula(max_n: int = 15) -> dict:
    """
    Verify Theorem 4: delta(P_n)^2 = 4(n-5) for n=6..max_n.

    Returns
    -------
    dict with keys 'passed', 'failures', 'n_tested'
    """
    failures = []
    for n in range(6, max_n + 1):
        d = tuple([2] * (n - 2) + [1, 1])  # path P_n
        tps = _get_tp(n)
        d2 = min(squared_distance(tau, d) for tau in tps)
        expected = 4 * (n - 5)
        if d2 != expected:
            failures.append({"n": n, "delta2": d2, "expected": expected})
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_tested": max_n,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(max_n: int = 15) -> None:
    """Print a formatted summary table of delta values for all alkane degree sequences."""
    rows = compute_delta_table(max_n)

    print(
        f"{'n':>3} {'ell':>4} {'p':>3} {'delta^2':>8} {'delta':>7}  "
        f"{'family':<25} {'degree sequence'}"
    )
    print("-" * 90)

    prev_n = None
    for r in rows:
        if r["n"] != prev_n:
            if prev_n is not None:
                print()
            prev_n = r["n"]
        fam = r["family"].replace("_", " ")
        print(
            f"{r['n']:>3} {r['ell']:>4} {r['p']:>3} {r['delta2']:>8} "
            f"{r['delta']:>7.3f}  {fam:<25} {r['d']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute delta for alkane degree sequences."
    )
    parser.add_argument(
        "--n", type=int, default=15,
        help="Maximum carbon count (default: 15)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run theorem verification checks"
    )
    args = parser.parse_args()

    print_summary_table(args.n)

    if args.verify:
        print("\n=== Theorem verification ===")

        result3 = verify_theorem3(args.n)
        status3 = "PASSED" if result3["passed"] else "FAILED"
        print(
            f"Theorem 3 [{status3}]: tested {result3['n_sequences']} "
            f"sequences up to n={result3['n_tested']}"
        )
        for f in result3["failures"]:
            print(f"  FAIL: {f}")

        result4 = verify_path_formula(args.n)
        status4 = "PASSED" if result4["passed"] else "FAILED"
        print(
            f"Theorem 4 (path formula) [{status4}]: tested n=6..{result4['n_tested']}"
        )
        for f in result4["failures"]:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
