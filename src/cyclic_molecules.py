"""
cyclic_molecules.py
===================
Compute delta for cyclic molecular graphs: cycles, benzenoids, and
user-supplied SMILES strings.

Requires RDKit for SMILES-based input.

Usage
-----
    python cyclic_molecules.py                  # runs built-in cycle library
    python cyclic_molecules.py --smiles "c1ccccc1"   # benzene
    python cyclic_molecules.py --csv molecules.csv   # from CSV file
"""

import argparse
import math
import sys

from partition_lib import (
    threshold_partitions,
    squared_distance,
    delta_squared,
    delta,
    nearest_threshold_partitions,
    degree_partition_from_smiles,
)


# ---------------------------------------------------------------------------
# Cycle graphs
# ---------------------------------------------------------------------------

def cycle_degree_partition(n: int) -> tuple[int, ...]:
    """
    Return the degree partition of the cycle C_n.

    All n vertices have degree 2, so d(C_n) = (2, 2, ..., 2).
    """
    return tuple([2] * n)


def delta_cycle(n: int) -> float:
    """
    Return delta(C_n).

    By Theorem 1: delta(C_n) = sqrt(4n - 14) for n >= 4.
    """
    d = cycle_degree_partition(n)
    return delta(d)


def verify_cycle_formula(max_n: int = 20) -> dict:
    """
    Verify Theorem 1: delta(C_n)^2 = 4n - 14 for n = 4..max_n.

    Returns
    -------
    dict with keys 'passed', 'failures', 'n_tested'
    """
    failures = []
    for n in range(4, max_n + 1):
        d = cycle_degree_partition(n)
        d2 = delta_squared(d)
        expected = 4 * n - 14
        if d2 != expected:
            failures.append({"n": n, "delta2": d2, "expected": expected})
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "n_tested": max_n,
    }


# ---------------------------------------------------------------------------
# Cyclomatic number
# ---------------------------------------------------------------------------

def cyclomatic_number(n_vertices: int, n_edges: int) -> int:
    """
    Return the cyclomatic number mu = |E| - n + 1 for a connected graph.

    Parameters
    ----------
    n_vertices : int
    n_edges    : int

    Returns
    -------
    int
        mu = n_edges - n_vertices + 1
    """
    return n_edges - n_vertices + 1


# ---------------------------------------------------------------------------
# SMILES-based computation
# ---------------------------------------------------------------------------

def delta_from_smiles(smiles: str) -> dict | None:
    """
    Compute delta and related quantities for a molecule given by SMILES.

    Requires RDKit.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    dict or None
        Keys: smiles, n, d, delta2, delta, tau_star, mu (cyclomatic number),
        crippen_logp (if RDKit available).
        Returns None if the SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        print("RDKit not available. Install with: pip install rdkit", file=sys.stderr)
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    carbons = [a for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6]
    n = len(carbons)
    if n == 0:
        return None

    carbon_set = {a.GetIdx() for a in carbons}
    degs = sorted(
        [sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 6) for a in carbons],
        reverse=True,
    )
    d = tuple(degs)

    n_edges = sum(
        1 for bond in mol_h.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
    )
    mu = cyclomatic_number(n, n_edges)

    tps = threshold_partitions(n)
    d2 = min(squared_distance(tau, d) for tau in tps)
    achievers = [tau for tau in tps if squared_distance(tau, d) == d2]

    try:
        logp = Descriptors.MolLogP(Chem.RemoveHs(mol_h))
    except Exception:
        logp = None

    return {
        "smiles": smiles,
        "n": n,
        "d": d,
        "delta2": d2,
        "delta": math.sqrt(d2),
        "tau_star": achievers,
        "mu": mu,
        "crippen_logp": logp,
    }


# ---------------------------------------------------------------------------
# Built-in molecule library
# ---------------------------------------------------------------------------

# Curated SMILES for representative cyclic molecules
# Format: (name, SMILES, expected_mu)
CYCLIC_MOLECULE_LIBRARY = [
    # Cycloalkanes
    ("cyclopropane",    "C1CC1",                        1),
    ("cyclobutane",     "C1CCC1",                       1),
    ("cyclopentane",    "C1CCCC1",                      1),
    ("cyclohexane",     "C1CCCCC1",                     1),
    ("cycloheptane",    "C1CCCCCC1",                    1),
    ("cyclooctane",     "C1CCCCCCC1",                   1),
    ("cyclodecane",     "C1CCCCCCCCC1",                 1),
    # Bicyclics
    ("bicyclo[2.2.0]hexane",   "C1CCC2CCC1C2",         2),  # norbornane skeleton
    ("decalin",                "C1CCC2CCCCC2C1",        2),
    ("bicyclo[2.2.2]octane",   "C1CC2CCC1CC2",          2),
    # Benzenoids
    ("benzene",         "c1ccccc1",                     1),
    ("naphthalene",     "c1ccc2ccccc2c1",               2),
    ("anthracene",      "c1ccc2cc3ccccc3cc2c1",         3),
    ("phenanthrene",    "c1ccc2ccc3ccccc3c2c1",         3),
    ("pyrene",          "c1cc2ccc3cccc4ccc(c1)c2c34",   4),
    ("coronene",        "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", 7),
    # Heterocycles
    ("pyridine",        "c1ccncc1",                     1),
    ("pyrimidine",      "c1ccnc(c1)N",                  1),
    ("indole",          "c1ccc2[nH]ccc2c1",             2),
    ("purine",          "c1ncc2[nH]cnc2n1",             3),
    # Bioactive rings
    ("morpholine",      "C1COCCN1",                     1),
    ("piperazine",      "C1CNCCN1",                     1),
    ("piperidine",      "C1CCCCN1",                     1),
    ("thiophene",       "c1ccsc1",                      1),
    ("furan",           "c1ccoc1",                      1),
]


def compute_library_table(library=None) -> list[dict]:
    """
    Compute delta for all molecules in the library.

    Parameters
    ----------
    library : list of (name, smiles, expected_mu) or None
        Uses CYCLIC_MOLECULE_LIBRARY if None.

    Returns
    -------
    list of dicts
        Each dict has keys: name, smiles, n, d, delta2, delta, mu, crippen_logp.
    """
    if library is None:
        library = CYCLIC_MOLECULE_LIBRARY

    rows = []
    for name, smiles, expected_mu in library:
        result = delta_from_smiles(smiles)
        if result is None:
            print(f"Warning: failed to process {name} ({smiles})", file=sys.stderr)
            continue
        result["name"] = name
        result["expected_mu"] = expected_mu
        rows.append(result)
    return rows


def print_library_table(rows: list[dict]) -> None:
    """Print a formatted table of delta values for a library of molecules."""
    print(
        f"{'Name':<30} {'n':>4} {'mu':>4} {'delta^2':>8} {'delta':>7} "
        f"{'CrippenLogP':>12}  degree sequence"
    )
    print("-" * 100)
    for r in sorted(rows, key=lambda x: (x["mu"], x["n"])):
        lp = f"{r['crippen_logp']:.3f}" if r["crippen_logp"] is not None else "N/A"
        print(
            f"{r['name']:<30} {r['n']:>4} {r['mu']:>4} {r['delta2']:>8} "
            f"{r['delta']:>7.3f} {lp:>12}  {r['d']}"
        )


# ---------------------------------------------------------------------------
# Monotonicity verification
# ---------------------------------------------------------------------------

def check_mu_monotonicity(rows: list[dict]) -> dict:
    """
    Check that delta is non-decreasing with cyclomatic number mu.

    Within each (n, ell) class, higher mu should give larger delta.
    This function reports any violations.

    Parameters
    ----------
    rows : list of dicts from compute_library_table()

    Returns
    -------
    dict with 'monotone' (bool) and 'violations' (list)
    """
    from itertools import combinations

    violations = []
    for r1, r2 in combinations(rows, 2):
        if r1["n"] == r2["n"]:
            if r1["mu"] < r2["mu"] and r1["delta2"] > r2["delta2"]:
                violations.append(
                    {
                        "mol1": r1["name"],
                        "mol2": r2["name"],
                        "mu1": r1["mu"],
                        "mu2": r2["mu"],
                        "delta2_1": r1["delta2"],
                        "delta2_2": r2["delta2"],
                    }
                )
    return {"monotone": len(violations) == 0, "violations": violations}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute delta for cyclic molecular graphs."
    )
    parser.add_argument("--smiles", type=str, help="Single SMILES string")
    parser.add_argument(
        "--csv", type=str,
        help="CSV file with columns 'name' and 'smiles'"
    )
    parser.add_argument(
        "--verify-cycles", action="store_true",
        help="Verify cycle formula delta(C_n)^2 = 4n-14"
    )
    parser.add_argument(
        "--max-n", type=int, default=20,
        help="Maximum n for cycle formula verification (default: 20)"
    )
    args = parser.parse_args()

    if args.verify_cycles:
        result = verify_cycle_formula(args.max_n)
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"Cycle formula [{status}]: tested n=4..{result['n_tested']}")
        for f in result["failures"]:
            print(f"  FAIL: {f}")
        return

    if args.smiles:
        result = delta_from_smiles(args.smiles)
        if result:
            print(f"SMILES:       {result['smiles']}")
            print(f"n:            {result['n']}")
            print(f"d:            {result['d']}")
            print(f"mu:           {result['mu']}")
            print(f"delta^2:      {result['delta2']}")
            print(f"delta:        {result['delta']:.6f}")
            print(f"tau*:         {result['tau_star'][0]}")
            if result["crippen_logp"] is not None:
                print(f"Crippen logP: {result['crippen_logp']:.3f}")
        else:
            print("Invalid SMILES or RDKit unavailable.")
        return

    if args.csv:
        import csv
        library = []
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                library.append((row["name"], row["smiles"], 0))
        rows = compute_library_table(library)
        print_library_table(rows)
        return

    # Default: run built-in library
    print("=== Built-in cyclic molecule library ===\n")
    rows = compute_library_table()
    print_library_table(rows)

    print("\n=== Cycle formula verification ===")
    result = verify_cycle_formula(args.max_n)
    status = "PASSED" if result["passed"] else "FAILED"
    print(f"delta(C_n)^2 = 4n-14 [{status}]: tested n=4..{result['n_tested']}")

    print("\n=== Monotonicity check (delta vs mu) ===")
    mono = check_mu_monotonicity(rows)
    status = "PASSED" if mono["monotone"] else f"{len(mono['violations'])} violations"
    print(f"Monotonicity with mu [{status}]")
    for v in mono["violations"]:
        print(f"  {v}")


if __name__ == "__main__":
    main()
