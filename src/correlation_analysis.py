"""
correlation_analysis.py
=======================
Correlation analysis of delta with boiling point (BP) and logP for
alkane isomers C1–C15.

Computes:
- Raw Spearman and Pearson correlations of delta with BP and logP
- Partial Pearson correlations controlling for carbon count n
- Within-isomer-class correlations

Reproduces Tables 2 and 3 in the paper.

Requirements
------------
    pip install rdkit scipy numpy

Usage
-----
    python correlation_analysis.py
    python correlation_analysis.py --max-n 15
    python correlation_analysis.py --max-n 9 --experimental-only
"""

import argparse
import math
import sys

import numpy as np
from scipy import stats

from partition_lib import threshold_partitions, squared_distance
from alkanes import (
    alkane_degree_sequences,
    all_alkane_degree_sequences,
    compute_delta_table,
)


# ---------------------------------------------------------------------------
# Experimental data: C1–C9 alkanes
# ---------------------------------------------------------------------------

# Experimental boiling points (°C at 1 atm) and logP values.
# logP: experimental octanol-water partition coefficients.
# Source: Sangster (1989), J. Phys. Chem. Ref. Data 18, 1111-1229.
# BP: NIST WebBook.
#
# Keyed by degree sequence tuple to match degree-sequence level analysis.

EXPERIMENTAL_DATA: dict[tuple, dict] = {
    # d(T)                          name                        BP      logP
    (0,):                          {"name": "methane",         "bp": -161.5, "logp": 1.09},
    (1, 1):                        {"name": "ethane",          "bp":  -88.6, "logp": 1.81},
    (2, 1, 1):                     {"name": "propane",         "bp":  -42.1, "logp": 2.36},
    (2, 2, 1, 1):                  {"name": "n-butane",        "bp":   -0.5, "logp": 2.89},
    (3, 1, 1, 1):                  {"name": "isobutane",       "bp":  -11.7, "logp": 2.76},
    (2, 2, 2, 1, 1):               {"name": "n-pentane",       "bp":   36.1, "logp": 3.45},
    (3, 2, 1, 1, 1):               {"name": "isopentane",      "bp":   27.7, "logp": 3.29},
    (4, 1, 1, 1, 1):               {"name": "neopentane",      "bp":    9.5, "logp": 3.11},
    (2, 2, 2, 2, 1, 1):            {"name": "n-hexane",        "bp":   68.7, "logp": 3.90},
    (3, 2, 2, 1, 1, 1):            {"name": "2-methylpentane", "bp":   60.3, "logp": 3.74},
    # Note: 3-methylpentane shares degree sequence with 2-methylpentane
    (3, 3, 1, 1, 1, 1):            {"name": "2,3-dimethylbutane", "bp": 57.9, "logp": 3.85},
    (4, 2, 1, 1, 1, 1):            {"name": "2,2-dimethylbutane", "bp": 49.7, "logp": 3.82},
    (2, 2, 2, 2, 2, 1, 1):         {"name": "n-heptane",       "bp":   98.4, "logp": 4.50},
    (3, 2, 2, 2, 1, 1, 1):         {"name": "3-methylhexane",  "bp":   91.9, "logp": 3.91},
    (3, 3, 2, 1, 1, 1, 1):         {"name": "2,3-dimethylpentane", "bp": 89.8, "logp": 3.96},
    (4, 2, 2, 1, 1, 1, 1):         {"name": "2,2-dimethylpentane", "bp": 79.2, "logp": 4.09},
    (4, 3, 1, 1, 1, 1, 1):         {"name": "2,2,3-trimethylbutane", "bp": 80.9, "logp": 4.09},
    (2, 2, 2, 2, 2, 2, 1, 1):      {"name": "n-octane",        "bp":  125.7, "logp": 4.90},
    (3, 2, 2, 2, 2, 1, 1, 1):      {"name": "3-methylheptane", "bp":  122.7, "logp": 4.50},
    (3, 3, 2, 2, 1, 1, 1, 1):      {"name": "3-ethylhexane",   "bp":   93.5, "logp": 3.91},
    (4, 2, 2, 2, 1, 1, 1, 1):      {"name": "2-methylheptane", "bp":  117.6, "logp": 4.50},
    (4, 3, 2, 1, 1, 1, 1, 1):      {"name": "2,3,3-trimethylpentane", "bp": 114.8, "logp": 4.09},
    (3, 3, 3, 1, 1, 1, 1, 1):      {"name": "2,2,3-trimethylpentane", "bp": 109.8, "logp": 4.09},
    (4, 4, 1, 1, 1, 1, 1, 1):      {"name": "2,2,4-trimethylpentane", "bp": 99.2, "logp": 4.09},
    (2, 2, 2, 2, 2, 2, 2, 1, 1):   {"name": "n-nonane",        "bp":  150.8, "logp": 5.65},
    (3, 2, 2, 2, 2, 2, 1, 1, 1):   {"name": "3-methyloctane",  "bp":  143.6, "logp": 5.00},
    (4, 2, 2, 2, 2, 1, 1, 1, 1):   {"name": "2-methyloctane",  "bp":  142.8, "logp": 5.00},
}


# ---------------------------------------------------------------------------
# Crippen logP via RDKit
# ---------------------------------------------------------------------------

def get_crippen_logp(smiles: str) -> float | None:
    """Return Crippen-Wildman logP for a SMILES string, or None."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.MolLogP(mol) if mol else None
    except ImportError:
        return None


def build_smiles_for_degree_seq(d: tuple[int, ...]) -> str | None:
    """
    Build a representative SMILES for an alkane with given degree sequence.

    Uses a greedy tree construction: repeatedly attach the current leaf
    (degree-1 vertex) to the highest-remaining-degree vertex.
    """
    n = len(d)
    if n == 1:
        return "C"

    remaining = list(d)
    edges = []
    active = list(range(n))

    while sum(remaining) > 0:
        # Find leaves among active vertices
        leaves = [i for i in active if remaining[i] == 1]
        if not leaves:
            break
        leaf = leaves[0]

        # Find highest-degree non-leaf to attach to
        non_leaves = [i for i in active if remaining[i] > 1]
        if not non_leaves:
            # Connect two remaining leaves
            other_leaves = [i for i in active if remaining[i] == 1 and i != leaf]
            if other_leaves:
                edges.append((leaf, other_leaves[0]))
                remaining[leaf] = 0
                remaining[other_leaves[0]] = 0
            break
        hub = max(non_leaves, key=lambda i: remaining[i])
        edges.append((leaf, hub))
        remaining[leaf] = 0
        remaining[hub] -= 1
        active = [i for i in active if remaining[i] > 0]

    # Build SMILES using RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol, Atom

        mol = RWMol()
        for _ in range(n):
            mol.AddAtom(Atom(6))
        for i, j in edges:
            mol.AddBond(i, j, Chem.BondType.SINGLE)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build analysis dataset
# ---------------------------------------------------------------------------

def build_dataset(max_n: int = 15) -> list[dict]:
    """
    Build the full analysis dataset.

    For n=1..9: use experimental BP and logP where available.
    For n=1..max_n: compute Crippen logP for one representative isomer.

    Returns
    -------
    list of dicts, each with keys:
        n, d, ell, delta2, delta, bp (or None), exp_logp (or None),
        crippen_logp (or None), name (or None)
    """
    rows = compute_delta_table(max_n)
    dataset = []

    for r in rows:
        d = r["d"]
        n = r["n"]
        row = {
            "n": n,
            "d": d,
            "ell": r["ell"],
            "p": r["p"],
            "delta2": r["delta2"],
            "delta": r["delta"],
            "bp": None,
            "exp_logp": None,
            "crippen_logp": None,
            "name": None,
        }

        # Attach experimental data if available
        if d in EXPERIMENTAL_DATA:
            exp = EXPERIMENTAL_DATA[d]
            row["bp"] = exp.get("bp")
            row["exp_logp"] = exp.get("logp")
            row["name"] = exp.get("name")

        # Compute Crippen logP from representative SMILES
        smiles = build_smiles_for_degree_seq(d)
        if smiles:
            row["crippen_logp"] = get_crippen_logp(smiles)
            row["smiles"] = smiles

        dataset.append(row)

    return dataset


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------

def partial_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """
    Compute the partial Pearson correlation of x and y controlling for z.

    Parameters
    ----------
    x, y, z : np.ndarray of float, same length

    Returns
    -------
    (r, p_value)
    """
    def residualise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    rx = residualise(x, z)
    ry = residualise(y, z)
    return stats.pearsonr(rx, ry)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(max_n: int = 15, experimental_only: bool = False) -> None:
    """
    Run the full correlation analysis and print results.

    Parameters
    ----------
    max_n : int
        Maximum carbon count.
    experimental_only : bool
        If True, only use rows with experimental data (n <= 9).
    """
    print(f"Building dataset (max_n={max_n})...")
    dataset = build_dataset(max_n)

    # --- Dataset 1: experimental (C1-C9) ---
    exp_rows = [r for r in dataset if r["exp_logp"] is not None and r["bp"] is not None]
    N1 = len(exp_rows)

    # --- Dataset 2: all unique degree sequences with Crippen logP ---
    crippen_rows = [r for r in dataset if r["crippen_logp"] is not None]
    if experimental_only:
        crippen_rows = [r for r in exp_rows if r["crippen_logp"] is not None]
    N2 = len(crippen_rows)

    print(f"\nDataset 1 (experimental, C1-C9): N={N1} unique degree sequences")
    print(f"Dataset 2 (Crippen logP, C1-C{max_n}): N={N2} unique degree sequences")

    # --- Raw correlations ---
    print("\n=== Raw correlations ===")
    print(f"{'Index':<8} {'vs BP rho_s':>12} {'vs BP r':>10} "
          f"{'vs expLogP rho_s':>17} {'vs expLogP r':>13} "
          f"{'vs Crippen rho_s':>17} {'vs Crippen r':>13}")
    print("-" * 100)

    if N1 > 0:
        d_exp = np.array([r["delta"] for r in exp_rows])
        n_exp = np.array([r["n"] for r in exp_rows], dtype=float)
        bp_exp = np.array([r["bp"] for r in exp_rows])
        lp_exp = np.array([r["exp_logp"] for r in exp_rows])

        rho_bp, _ = stats.spearmanr(d_exp, bp_exp)
        r_bp, _   = stats.pearsonr(d_exp, bp_exp)
        rho_lp, _ = stats.spearmanr(d_exp, lp_exp)
        r_lp, _   = stats.pearsonr(d_exp, lp_exp)

    if N2 > 0:
        d_cr = np.array([r["delta"] for r in crippen_rows])
        n_cr = np.array([r["n"] for r in crippen_rows], dtype=float)
        lp_cr = np.array([r["crippen_logp"] for r in crippen_rows])
        rho_cr, _ = stats.spearmanr(d_cr, lp_cr)
        r_cr, _   = stats.pearsonr(d_cr, lp_cr)

    if N1 > 0:
        print(f"{'delta':<8} {rho_bp:>12.3f} {r_bp:>10.3f} "
              f"{rho_lp:>17.3f} {r_lp:>13.3f} "
              f"{rho_cr:>17.3f} {r_cr:>13.3f}")

        # Baseline: n alone
        rho_n_bp, _ = stats.spearmanr(n_exp, bp_exp)
        r_n_bp, _   = stats.pearsonr(n_exp, bp_exp)
        rho_n_lp, _ = stats.spearmanr(n_exp, lp_exp)
        r_n_lp, _   = stats.pearsonr(n_exp, lp_exp)
        rho_n_cr, _ = stats.spearmanr(n_cr, lp_cr)
        r_n_cr, _   = stats.pearsonr(n_cr, lp_cr)

        print(f"{'n (base)':<8} {rho_n_bp:>12.3f} {r_n_bp:>10.3f} "
              f"{rho_n_lp:>17.3f} {r_n_lp:>13.3f} "
              f"{rho_n_cr:>17.3f} {r_n_cr:>13.3f}")

    # --- Partial correlations ---
    print("\n=== Partial Pearson correlations (controlling for n) ===")
    print(f"{'Property':<40} {'N':>6} {'Partial r':>10} {'p-value':>12}")
    print("-" * 75)

    if N1 > 0:
        pr_bp, pp_bp = partial_pearson(d_exp, bp_exp, n_exp)
        pr_lp, pp_lp = partial_pearson(d_exp, lp_exp, n_exp)
        print(f"{'BP (°C), C1-C9 experimental':<40} {N1:>6} {pr_bp:>10.3f} {pp_bp:>12.3e}")
        print(f"{'exp logP (C1-C9)':<40} {N1:>6} {pr_lp:>10.3f} {pp_lp:>12.3e}")

    if N2 > 0:
        pr_cr, pp_cr = partial_pearson(d_cr, lp_cr, n_cr)
        print(f"{'Crippen logP (C1-C' + str(max_n) + ')':<40} {N2:>6} {pr_cr:>10.3f} {pp_cr:>12.3e}")

    # --- Within-class correlations ---
    print("\n=== Within-n Spearman correlations (delta vs exp logP) ===")
    for n_target in sorted(set(r["n"] for r in exp_rows)):
        subset = [r for r in exp_rows if r["n"] == n_target]
        deltas_n = [r["delta"] for r in subset]
        logps_n  = [r["exp_logp"] for r in subset]
        if len(subset) < 2 or len(set(deltas_n)) == 1:
            print(f"  n={n_target}: {len(subset)} seq(s), all delta={deltas_n[0]:.3f} — cannot rank")
            continue
        rho, _ = stats.spearmanr(deltas_n, logps_n)
        print(f"  n={n_target}: {len(subset)} seqs, rho(delta, logP) = {rho:+.3f}")

    # --- Summary table of unique degree sequences ---
    print(f"\n=== Delta values by degree sequence (n=2..{min(max_n, 9)}, experimental data) ===")
    print(f"{'n':>3} {'ell':>4} {'delta^2':>8} {'delta':>7} {'BP':>7} "
          f"{'expLogP':>8} {'CripLogP':>10}  name")
    print("-" * 80)
    for r in sorted(exp_rows, key=lambda x: (x["n"], -x["delta2"])):
        lp = f"{r['crippen_logp']:.3f}" if r["crippen_logp"] else "N/A"
        print(f"{r['n']:>3} {r['ell']:>4} {r['delta2']:>8} {r['delta']:>7.3f} "
              f"{r['bp']:>7.1f} {r['exp_logp']:>8.2f} {lp:>10}  {r['name']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Correlation analysis of delta with BP and logP."
    )
    parser.add_argument(
        "--max-n", type=int, default=15,
        help="Maximum carbon count for Crippen logP analysis (default: 15)"
    )
    parser.add_argument(
        "--experimental-only", action="store_true",
        help="Restrict Crippen analysis to C1-C9 (experimental data available)"
    )
    args = parser.parse_args()
    run_analysis(args.max_n, args.experimental_only)


if __name__ == "__main__":
    main()
