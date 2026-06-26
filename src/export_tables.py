"""
export_tables.py
================
Generate supplementary data tables for:

  "A Geometric Topological Index via Threshold Partitions"
  Paul M. King, Birkbeck, University of London

Running this script reproduces the numerical data underlying
Tables 2-4 of the paper and writes four CSV files to data/:

  alkanes_C1_C15.csv       -- delta for all unique alkane degree sequences n=1..15
  experimental_C1_C9.csv   -- delta, BP, and experimental logP for C1-C9
  freesolv_alkanes.csv      -- delta and hydration free energy from FreeSolv
  cyclic_molecules.csv      -- delta for the 25-molecule cyclic library

Usage
-----
    python src/export_tables.py

Output files are written to data/ relative to the repo root.
"""

import csv
import math
import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from partition_lib import threshold_partitions, squared_distance
from alkanes import (
    all_alkane_degree_sequences,
    compute_delta_table,
    classify_tree,
)
from correlation_analysis import (
    build_dataset,
    EXPERIMENTAL_DATA,
    build_smiles_for_degree_seq,
    get_crippen_logp,
)
from cyclic_molecules import compute_library_table, CYCLIC_MOLECULE_LIBRARY
from freesolv_analysis import (
    load_freesolv,
    extract_alkanes,
    aggregate_by_degree_seq,
    DEFAULT_DB,
)

# Output directory: data/ relative to repo root
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR  = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Table 1: alkanes C1-C15
# ---------------------------------------------------------------------------

def export_alkanes(path: str) -> int:
    """
    Export delta for all unique alkane carbon-skeleton degree sequences, n=1..15.

    Columns: n, ell, p, degree_sequence, delta_sq, delta, family, crippen_logp
    """
    rows = compute_delta_table(max_n=15)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "ell", "p", "degree_sequence",
            "delta_sq", "delta", "family", "crippen_logp",
        ])
        for r in rows:
            d = r["d"]
            smiles = build_smiles_for_degree_seq(d)
            logp   = get_crippen_logp(smiles) if smiles else None
            writer.writerow([
                r["n"],
                r["ell"],
                r["p"],
                str(d),
                r["delta2"],
                f"{r['delta']:.6f}",
                r["family"],
                f"{logp:.4f}" if logp is not None else "",
            ])

    return len(rows)


# ---------------------------------------------------------------------------
# Table 2: experimental data C1-C9
# ---------------------------------------------------------------------------

def export_experimental(path: str) -> int:
    """
    Export delta alongside experimental BP and logP for C1-C9 alkanes.

    Columns: n, ell, degree_sequence, delta_sq, delta, bp_degC, exp_logp,
             crippen_logp, name
    """
    dataset = build_dataset(max_n=9)
    exp_rows = [r for r in dataset if r["exp_logp"] is not None]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "ell", "degree_sequence",
            "delta_sq", "delta",
            "bp_degC", "exp_logp", "crippen_logp",
            "name",
        ])
        for r in sorted(exp_rows, key=lambda x: (x["n"], -x["delta2"])):
            writer.writerow([
                r["n"],
                r["ell"],
                str(r["d"]),
                r["delta2"],
                f"{r['delta']:.6f}",
                f"{r['bp']:.1f}"       if r["bp"]         is not None else "",
                f"{r['exp_logp']:.2f}" if r["exp_logp"]   is not None else "",
                f"{r['crippen_logp']:.4f}" if r["crippen_logp"] is not None else "",
                r["name"] or "",
            ])

    return len(exp_rows)


# ---------------------------------------------------------------------------
# Table 3: FreeSolv hydration free energy
# ---------------------------------------------------------------------------

def export_freesolv(path: str) -> int:
    """
    Export delta and experimental hydration free energy for alkanes in FreeSolv.

    Columns: n, ell, degree_sequence, delta_sq, delta,
             dG_hyd_kcal_mol, dG_unc, n_isomers, names
    """
    if not os.path.exists(DEFAULT_DB):
        print(f"  FreeSolv database not found at {DEFAULT_DB}; skipping.")
        return 0

    entries = load_freesolv(DEFAULT_DB)
    alkanes = extract_alkanes(entries)
    agg     = aggregate_by_degree_seq(alkanes)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "ell", "degree_sequence",
            "delta_sq", "delta",
            "dG_hyd_kcal_mol", "dG_unc",
            "n_isomers", "names",
        ])
        for r in agg:
            writer.writerow([
                r["n"],
                r["ell"],
                str(r["d"]),
                r["delta2"],
                f"{r['delta_val']:.6f}",
                f"{r['dG_hyd']:.4f}",
                f"{r['dG_unc']:.4f}",
                r["n_isomers"],
                "; ".join(r["names"]),
            ])

    return len(agg)


# ---------------------------------------------------------------------------
# Table 4: cyclic molecule library
# ---------------------------------------------------------------------------

def export_cyclic(path: str) -> int:
    """
    Export delta for the 25-molecule cyclic library.

    Columns: name, smiles, n, mu, delta_sq, delta, crippen_logp
    """
    rows = compute_library_table(CYCLIC_MOLECULE_LIBRARY)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "smiles", "n", "mu",
            "delta_sq", "delta", "crippen_logp",
        ])
        for r in sorted(rows, key=lambda x: (x["mu"], x["n"])):
            lp = f"{r['crippen_logp']:.4f}" if r["crippen_logp"] is not None else ""
            writer.writerow([
                r["name"],
                r["smiles"],
                r["n"],
                r["mu"],
                r["delta2"],
                f"{r['delta']:.6f}",
                lp,
            ])

    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tables = [
        ("alkanes_C1_C15.csv",      export_alkanes,      "alkane degree sequences C1-C15"),
        ("experimental_C1_C9.csv",  export_experimental, "experimental data C1-C9"),
        ("freesolv_alkanes.csv",     export_freesolv,     "FreeSolv hydration free energy"),
        ("cyclic_molecules.csv",     export_cyclic,       "cyclic molecule library"),
    ]

    print("Exporting supplementary data tables...\n")
    for filename, func, description in tables:
        path = os.path.join(DATA_DIR, filename)
        n = func(path)
        if n:
            print(f"  {filename:<35} {n:>4} rows  ->  {path}")
        else:
            print(f"  {filename:<35} skipped")

    print("\nDone. Files written to:", DATA_DIR)
    print()
    print("To verify against paper values:")
    print("  - alkanes_C1_C15.csv:     124 rows (incl. methane n=1); 123 unique")
    print("                            degree sequences for n>=2, as reported in paper")
    print("  - experimental_C1_C9.csv:  27 rows, Spearman rho(delta,BP) ~ 0.922")
    print("  - freesolv_alkanes.csv:    23 rows, Pearson r(delta,dG) ~ 0.817")
    print("  - cyclic_molecules.csv:    25 rows, all delta increase with mu")


if __name__ == "__main__":
    main()
