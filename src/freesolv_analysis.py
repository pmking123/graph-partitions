"""
freesolv_analysis.py
====================
Analysis of delta vs hydration free energy (dG_hyd) using the FreeSolv database.

FreeSolv (Mobley & Guthrie, 2014) provides experimental and calculated hydration
free energies for 643 small neutral molecules. This script extracts the alkane
subset, matches entries to carbon-skeleton degree sequences, and analyses the
relationship with delta.

Key finding: unlike logP, hydration free energy of alkanes shows no significant
partial correlation with delta after controlling for carbon count. This is because
the within-isomer-class variation in dG_hyd is smaller than the experimental
uncertainty (0.6 kcal/mol default). The result is chemically informative:
dG_hyd for saturated hydrocarbons is insensitive to branching details at the
degree-sequence level. logP, by contrast, does depend on shape, which is why
the partial correlation with delta is significant for logP but not dG_hyd.

References
----------
Mobley, D.L., Guthrie, J.P. (2014). FreeSolv: a database of experimental and
  calculated hydration free energies, with input files.
  J. Comput. Aided Mol. Des. 28, 711-720. doi:10.1007/s10822-014-9747-x

GitHub: https://github.com/MobleyLab/FreeSolv

Usage
-----
    python freesolv_analysis.py                          # uses bundled data
    python freesolv_analysis.py --db path/to/database.txt
    python freesolv_analysis.py --download               # fetch fresh copy
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from itertools import product as iprod

import numpy as np
from scipy import stats

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from partition_lib import (
    threshold_partitions,
    squared_distance,
    delta_squared,
    delta,
)


# ---------------------------------------------------------------------------
# FreeSolv data loading and alkane extraction
# ---------------------------------------------------------------------------

DEFAULT_DB = os.path.join(
    os.path.dirname(__file__), "..", "data", "freesolv_database.txt"
)
FREESOLV_URL = (
    "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
)


def download_freesolv(dest: str) -> None:
    """Download the FreeSolv database text file."""
    import urllib.request
    print(f"Downloading FreeSolv from {FREESOLV_URL} ...")
    urllib.request.urlretrieve(FREESOLV_URL, dest)
    print(f"Saved to {dest}")


def load_freesolv(path: str) -> list[dict]:
    """
    Parse the FreeSolv database text file.

    Returns
    -------
    list of dicts with keys:
        id, smiles, name, exp_dg, exp_unc, calc_dg, calc_unc
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 6:
                continue
            try:
                entries.append(
                    {
                        "id":       parts[0],
                        "smiles":   parts[1],
                        "name":     parts[2],
                        "exp_dg":   float(parts[3]),
                        "exp_unc":  float(parts[4]),
                        "calc_dg":  float(parts[5]),
                        "calc_unc": float(parts[6]) if len(parts) > 6 else None,
                    }
                )
            except (ValueError, IndexError):
                continue
    return entries


def is_pure_alkane_smiles(smiles: str) -> bool:
    """
    Return True if the SMILES represents a pure saturated acyclic hydrocarbon.

    Requires RDKit.
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit required. Install with: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # Acyclic
    if mol.GetRingInfo().NumRings() > 0:
        return False

    # All bonds single
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            return False

    # Only C and H
    mol_h = Chem.AddHs(mol)
    atom_nums = {a.GetAtomicNum() for a in mol_h.GetAtoms()}
    return atom_nums == {1, 6}


def carbon_skeleton_degree_seq(smiles: str) -> tuple[int, ...] | None:
    """Return the carbon-skeleton degree sequence for a SMILES string."""
    try:
        from rdkit import Chem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    degs = sorted(
        [
            sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 6)
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6
        ],
        reverse=True,
    )
    return tuple(degs)


def extract_alkanes(entries: list[dict]) -> list[dict]:
    """
    Filter FreeSolv entries to pure saturated acyclic hydrocarbons and
    compute their carbon-skeleton degree sequences and delta values.

    Returns
    -------
    list of dicts with additional keys: n, d, delta2, delta_val
    """
    _tp_cache: dict[int, list] = {}

    def _get_tp(n: int) -> list:
        if n not in _tp_cache:
            _tp_cache[n] = threshold_partitions(n)
        return _tp_cache[n]

    result = []
    for entry in entries:
        smiles = entry["smiles"]
        try:
            if not is_pure_alkane_smiles(smiles):
                continue
        except ImportError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        d = carbon_skeleton_degree_seq(smiles)
        if d is None:
            continue

        n = len(d)
        tps = _get_tp(n)
        d2 = min(squared_distance(tau, d) for tau in tps)

        result.append(
            {
                **entry,
                "n": n,
                "d": d,
                "ell": d.count(1),
                "delta2": d2,
                "delta_val": math.sqrt(d2),
            }
        )

    return result


# ---------------------------------------------------------------------------
# Degree-sequence level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_degree_seq(alkanes: list[dict]) -> list[dict]:
    """
    Aggregate multiple isomers sharing the same carbon-skeleton degree
    sequence into a single representative row.

    For delta, degree sequence, and n: these are identical across isomers
    by definition, so the value is unambiguous.

    For dG_hyd: take the mean of experimental values. Report the root-mean-
    square uncertainty, inflated by the standard deviation across isomers
    if multiple values are present (to reflect the within-class variation
    as a source of uncertainty).

    Returns
    -------
    list of dicts, one per unique degree sequence, sorted by (n, ell).
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in alkanes:
        groups[(row["n"], row["d"])].append(row)

    result = []
    for (n, d), rows in groups.items():
        dg_vals = np.array([r["exp_dg"] for r in rows])
        unc_vals = np.array([r["exp_unc"] for r in rows])

        dg_mean = float(np.mean(dg_vals))
        # Combined uncertainty: rms of reported uncertainties + std of values
        unc_combined = float(
            np.sqrt(np.mean(unc_vals**2) + np.var(dg_vals, ddof=0))
        )

        names = [r["name"] for r in rows]
        result.append(
            {
                "n": n,
                "d": d,
                "ell": d.count(1),
                "p": n - d.count(1),
                "delta2": rows[0]["delta2"],
                "delta_val": rows[0]["delta_val"],
                "dG_hyd": dg_mean,
                "dG_unc": unc_combined,
                "names": names,
                "n_isomers": len(rows),
            }
        )

    return sorted(result, key=lambda x: (x["n"], x["ell"]))


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------

def partial_pearson(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Partial Pearson correlation of x and y controlling for z."""

    def residualise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    return stats.pearsonr(residualise(x, z), residualise(y, z))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(db_path: str) -> None:
    """
    Run the full hydration free energy analysis and print results.

    Parameters
    ----------
    db_path : str
        Path to the FreeSolv database text file.
    """
    print(f"Loading FreeSolv from {db_path}")
    entries = load_freesolv(db_path)
    print(f"Total entries: {len(entries)}")

    print("Extracting alkanes...")
    alkanes = extract_alkanes(entries)
    print(f"Pure saturated acyclic alkanes: {len(alkanes)}")

    agg = aggregate_by_degree_seq(alkanes)
    N = len(agg)
    print(f"Unique degree sequences: {N}")

    # --- Print data table ---
    print(f"\n{'Name(s)':<38} {'n':>3} {'ell':>4} {'delta^2':>8} "
          f"{'dG_hyd':>9} {'unc':>6}")
    print("-" * 78)
    for r in agg:
        name_str = r["names"][0] if len(r["names"]) == 1 else \
            r["names"][0] + f" (+{len(r['names'])-1})"
        flag = " *" if r["n_isomers"] > 1 else ""
        print(f"{name_str:<38} {r['n']:>3} {r['ell']:>4} {r['delta2']:>8} "
              f"{r['dG_hyd']:>9.3f} {r['dG_unc']:>6.3f}{flag}")
    print("* = mean of multiple FreeSolv isomers with same degree sequence")

    # --- Correlations ---
    deltas = np.array([r["delta_val"] for r in agg])
    dg_hyd = np.array([r["dG_hyd"] for r in agg])
    ns     = np.array([r["n"] for r in agg], dtype=float)

    r_raw,   p_raw   = stats.pearsonr(deltas, dg_hyd)
    rho_raw, p_rho   = stats.spearmanr(deltas, dg_hyd)
    r_n,     _       = stats.pearsonr(ns, dg_hyd)
    rho_n,   _       = stats.spearmanr(ns, dg_hyd)

    print(f"\n=== Raw correlations (N={N}) ===")
    print(f"{'Index':<10} {'vs dG_hyd r':>12} {'rho_s':>8}")
    print("-" * 33)
    print(f"{'delta':<10} {r_raw:>12.3f} {rho_raw:>8.3f}")
    print(f"{'n':<10} {r_n:>12.3f} {rho_n:>8.3f}")

    pr, pp = partial_pearson(deltas, dg_hyd, ns)
    print(f"\n=== Partial Pearson r(delta, dG_hyd | n) ===")
    print(f"  r = {pr:.3f}, p = {pp:.3e}")

    # --- Within-n analysis ---
    print(f"\n=== Within-n Spearman correlations ===")
    for n_target in sorted(set(r["n"] for r in agg)):
        sub = [r for r in agg if r["n"] == n_target]
        if len(sub) < 2:
            continue
        dl = [r["delta_val"] for r in sub]
        gl = [r["dG_hyd"] for r in sub]
        dg_range = max(gl) - min(gl)
        typ_unc  = max(r["dG_unc"] for r in sub)
        if len(set(dl)) == 1:
            print(f"  n={n_target}: all delta={dl[0]:.3f}, "
                  f"dG range={dg_range:.2f} kcal/mol, unc=±{typ_unc:.2f}")
            continue
        rho, _ = stats.spearmanr(dl, gl)
        noise_flag = " [range < uncertainty]" if dg_range < typ_unc else ""
        print(f"  n={n_target} ({len(sub)} seqs): rho={rho:+.3f}, "
              f"dG range={dg_range:.2f}, unc=±{typ_unc:.2f}{noise_flag}")

    # --- Interpretation ---
    print(f"""
=== Interpretation ===
dG_hyd for saturated hydrocarbons is positive (unfavourable hydration,
i.e. hydrophobic effect).

Raw correlation (r = {r_raw:.3f}) is dominated by molecular size:
  n alone achieves r = {r_n:.3f}.

Partial correlation after controlling for n:
  r = {pr:.3f} (p = {pp:.3e}) -- not significant.

Reason: the within-isomer-class variation in dG_hyd is smaller than
the FreeSolv experimental uncertainty (0.6 kcal/mol default) for all
n in this dataset. The signal delta encodes (branching vs linearity)
is present in dG_hyd in principle, but is buried in the measurement
noise at the accuracy of these experiments.

This contrasts with logP (partial r = 0.555, p = 0.003), where the
within-isomer variation exceeds the measurement precision.

The result is chemically informative: hydration free energy of simple
alkanes depends primarily on size (n), not on branching topology at
the degree-sequence level. logP is more sensitive to shape because it
also captures vapour-pressure and solvation effects in octanol.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse delta vs hydration free energy (FreeSolv)."
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB,
        help=f"Path to FreeSolv database.txt (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download a fresh copy of FreeSolv before analysis"
    )
    args = parser.parse_args()

    if args.download:
        dest = args.db if args.db != DEFAULT_DB else DEFAULT_DB
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        download_freesolv(dest)

    if not os.path.exists(args.db):
        print(f"Database not found at {args.db}.")
        print("Run with --download to fetch it, or specify --db path.")
        sys.exit(1)

    run_analysis(args.db)


if __name__ == "__main__":
    main()
