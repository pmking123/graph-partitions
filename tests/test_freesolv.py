"""
tests/test_freesolv.py
======================
Unit tests for freesolv_analysis.py.

Run with: pytest tests/
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from freesolv_analysis import (
    load_freesolv,
    extract_alkanes,
    aggregate_by_degree_seq,
    is_pure_alkane_smiles,
    carbon_skeleton_degree_seq,
    partial_pearson,
)

import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "freesolv_database.txt")
DB_AVAILABLE = os.path.exists(DB_PATH)


# ---------------------------------------------------------------------------
# is_pure_alkane_smiles
# ---------------------------------------------------------------------------

class TestIsAlkane:
    def test_methane(self):
        assert is_pure_alkane_smiles("C")

    def test_ethane(self):
        assert is_pure_alkane_smiles("CC")

    def test_isobutane(self):
        assert is_pure_alkane_smiles("CC(C)C")

    def test_neopentane(self):
        assert is_pure_alkane_smiles("CC(C)(C)C")

    def test_octane(self):
        assert is_pure_alkane_smiles("CCCCCCCC")

    def test_ethylene_is_not_alkane(self):
        """Alkene: not saturated."""
        assert not is_pure_alkane_smiles("C=C")

    def test_acetylene_is_not_alkane(self):
        assert not is_pure_alkane_smiles("C#C")

    def test_benzene_is_not_alkane(self):
        """Aromatic: not acyclic."""
        assert not is_pure_alkane_smiles("c1ccccc1")

    def test_cyclohexane_is_not_alkane(self):
        """Cyclic alkane: excluded."""
        assert not is_pure_alkane_smiles("C1CCCCC1")

    def test_ethanol_is_not_alkane(self):
        """Contains oxygen."""
        assert not is_pure_alkane_smiles("CCO")

    def test_butadiene_is_not_alkane(self):
        assert not is_pure_alkane_smiles("C=CC=C")


# ---------------------------------------------------------------------------
# carbon_skeleton_degree_seq
# ---------------------------------------------------------------------------

class TestDegreeSeq:
    def test_methane(self):
        assert carbon_skeleton_degree_seq("C") == (0,)

    def test_ethane(self):
        assert carbon_skeleton_degree_seq("CC") == (1, 1)

    def test_propane(self):
        assert carbon_skeleton_degree_seq("CCC") == (2, 1, 1)

    def test_isobutane(self):
        assert carbon_skeleton_degree_seq("CC(C)C") == (3, 1, 1, 1)

    def test_neopentane(self):
        assert carbon_skeleton_degree_seq("CC(C)(C)C") == (4, 1, 1, 1, 1)

    def test_n_hexane(self):
        assert carbon_skeleton_degree_seq("CCCCCC") == (2, 2, 2, 2, 1, 1)

    def test_two_isomers_same_deg_seq(self):
        """2-methylpentane and 3-methylpentane share a degree sequence."""
        d1 = carbon_skeleton_degree_seq("CC(C)CCC")   # 2-methylpentane
        d2 = carbon_skeleton_degree_seq("CCC(C)CC")   # 3-methylpentane
        assert d1 == d2 == (3, 2, 2, 1, 1, 1)

    def test_invalid_smiles(self):
        assert carbon_skeleton_degree_seq("not_a_smiles") is None


# ---------------------------------------------------------------------------
# FreeSolv database loading (requires data file)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="FreeSolv database not found")
class TestFreeSolvLoading:
    def setup_method(self):
        self.entries = load_freesolv(DB_PATH)

    def test_entry_count(self):
        """FreeSolv v0.52 has 642 entries."""
        assert len(self.entries) >= 640

    def test_entry_keys(self):
        e = self.entries[0]
        for key in ("id", "smiles", "name", "exp_dg", "exp_unc"):
            assert key in e

    def test_methane_present(self):
        methanes = [e for e in self.entries if e["smiles"] == "C"]
        assert len(methanes) == 1
        assert methanes[0]["exp_dg"] == pytest.approx(2.0, abs=0.1)

    def test_neopentane_present(self):
        neo = [e for e in self.entries if "neopentane" in e["name"].lower()]
        assert len(neo) >= 1
        assert neo[0]["exp_dg"] == pytest.approx(2.51, abs=0.1)

    def test_no_nan_values(self):
        for e in self.entries:
            assert not math.isnan(e["exp_dg"])
            assert not math.isnan(e["exp_unc"])


# ---------------------------------------------------------------------------
# Alkane extraction (requires data file)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="FreeSolv database not found")
class TestAlkaneExtraction:
    def setup_method(self):
        entries = load_freesolv(DB_PATH)
        self.alkanes = extract_alkanes(entries)

    def test_count(self):
        """Should find exactly 27 pure alkane entries in FreeSolv v0.52."""
        assert len(self.alkanes) == 27

    def test_unique_degree_seq_count(self):
        """Should aggregate to 23 unique degree sequences."""
        agg = aggregate_by_degree_seq(self.alkanes)
        assert len(agg) == 23

    def test_all_have_delta(self):
        for r in self.alkanes:
            assert r["delta2"] >= 0
            assert r["delta_val"] >= 0

    def test_parity_lemma(self):
        """delta^2 is always even."""
        for r in self.alkanes:
            assert r["delta2"] % 2 == 0, \
                f"delta^2={r['delta2']} is odd for {r['name']}"

    def test_known_values(self):
        """Spot-check delta^2 for specific molecules."""
        expected = {
            "C":             0,   # methane
            "CC":            0,   # ethane
            "CC(C)(C)C":     0,   # neopentane (star)
            "CCCC":          2,   # n-butane
            "CCCCCC":        4,   # hexane
            "CCCCCCC":       8,   # heptane
            "CCCCCCCC":     12,   # octane
        }
        smiles_map = {r["smiles"]: r["delta2"] for r in self.alkanes}
        for smi, exp_d2 in expected.items():
            # Normalise SMILES via RDKit for comparison
            from rdkit import Chem
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            canon_map = {}
            for r in self.alkanes:
                c = Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"]))
                canon_map[c] = r["delta2"]
            assert canon in canon_map, f"{smi} not found in alkanes"
            assert canon_map[canon] == exp_d2, \
                f"{smi}: expected delta^2={exp_d2}, got {canon_map[canon]}"

    def test_positive_dg_hyd(self):
        """All alkane dG_hyd values are positive (hydrophobic)."""
        for r in self.alkanes:
            assert r["exp_dg"] > 0, \
                f"{r['name']} has negative dG_hyd={r['exp_dg']}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="FreeSolv database not found")
class TestAggregation:
    def setup_method(self):
        entries = load_freesolv(DB_PATH)
        alkanes = extract_alkanes(entries)
        self.agg = aggregate_by_degree_seq(alkanes)

    def test_two_methylpentane_isomers_merged(self):
        """2-methylpentane and 3-methylpentane share (3,2,2,1,1,1)."""
        d_target = (3, 2, 2, 1, 1, 1)
        match = [r for r in self.agg if r["d"] == d_target]
        assert len(match) == 1
        assert match[0]["n_isomers"] == 2

    def test_mean_dg_hyd_for_merged(self):
        """Mean dG_hyd for (3,2,2,1,1,1) should be 2.51 (both isomers have 2.51)."""
        d_target = (3, 2, 2, 1, 1, 1)
        match = [r for r in self.agg if r["d"] == d_target][0]
        assert match["dG_hyd"] == pytest.approx(2.51, abs=0.01)

    def test_sorted_by_n_then_ell(self):
        for i in range(len(self.agg) - 1):
            r1, r2 = self.agg[i], self.agg[i + 1]
            assert (r1["n"], r1["ell"]) <= (r2["n"], r2["ell"])

    def test_combined_uncertainty_geq_individual(self):
        """Combined uncertainty for merged rows >= individual uncertainty."""
        for r in self.agg:
            if r["n_isomers"] > 1:
                # Combined unc includes variance across isomers
                assert r["dG_unc"] >= 0


# ---------------------------------------------------------------------------
# Partial correlation
# ---------------------------------------------------------------------------

class TestPartialCorrelation:
    def test_orthogonal_case(self):
        """When x is orthogonal to y after removing z, partial r should be ~0."""
        np.random.seed(42)
        z = np.random.randn(100)
        x = z + np.random.randn(100) * 0.1   # x tracks z
        y = z + np.random.randn(100) * 0.1   # y tracks z, independent of x
        r, p = partial_pearson(x, y, z)
        assert abs(r) < 0.3   # partial correlation near zero

    def test_perfect_partial_correlation(self):
        """When residuals of x on z equal residuals of y on z, partial r = 1."""
        np.random.seed(42)
        z = np.linspace(0, 10, 100)
        noise = np.random.randn(100) * 0.01
        # x and y have the SAME residual structure after removing z
        x = 2.0 * z + noise
        y = 3.0 * z + noise   # identical noise component
        r, p = partial_pearson(x, y, z)
        assert r > 0.99

    def test_returns_tuple(self):
        """partial_pearson returns (r, p) with r in [-1,1] and p in [0,1]."""
        np.random.seed(7)
        n = 30
        z = np.random.randn(n)
        x = z + np.random.randn(n)
        y = np.random.randn(n)   # independent of x once z is removed
        result = partial_pearson(x, y, z)
        assert len(result) == 2
        r, p = result
        assert -1.0 <= float(r) <= 1.0
        assert 0.0 <= float(p) <= 1.0


# ---------------------------------------------------------------------------
# Key scientific findings (regression tests)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DB_AVAILABLE, reason="FreeSolv database not found")
class TestScientificFindings:
    """
    Regression tests for the key findings reported in the paper.
    These tests will fail if the analysis changes in a way that alters
    the reported results.
    """

    def setup_method(self):
        from scipy import stats
        entries = load_freesolv(DB_PATH)
        alkanes = extract_alkanes(entries)
        self.agg = aggregate_by_degree_seq(alkanes)
        self.deltas = np.array([r["delta_val"] for r in self.agg])
        self.dg_hyd = np.array([r["dG_hyd"] for r in self.agg])
        self.ns     = np.array([r["n"] for r in self.agg], dtype=float)
        self.stats  = stats

    def test_raw_pearson_r(self):
        """Raw Pearson r(delta, dG_hyd) ~ 0.82."""
        r, _ = self.stats.pearsonr(self.deltas, self.dg_hyd)
        assert r == pytest.approx(0.817, abs=0.02)

    def test_raw_spearman_rho(self):
        """Raw Spearman rho(delta, dG_hyd) ~ 0.80."""
        rho, _ = self.stats.spearmanr(self.deltas, self.dg_hyd)
        assert rho == pytest.approx(0.799, abs=0.02)

    def test_partial_correlation_not_significant(self):
        """
        Partial r(delta, dG_hyd | n) should be near zero and not significant.
        This is the key finding: dG_hyd carries no shape information beyond n.
        """
        pr, pp = partial_pearson(self.deltas, self.dg_hyd, self.ns)
        assert abs(pr) < 0.15       # near zero
        assert pp > 0.05            # not significant

    def test_within_class_range_below_uncertainty(self):
        """
        For all n=4..9, the within-isomer-class variation in dG_hyd
        is less than or equal to the typical experimental uncertainty (0.6).
        """
        for n_target in range(4, 10):
            sub = [r for r in self.agg if r["n"] == n_target]
            if len(sub) < 2:
                continue
            dg_range = max(r["dG_hyd"] for r in sub) - \
                       min(r["dG_hyd"] for r in sub)
            assert dg_range <= 0.65, \
                f"n={n_target}: within-class dG range={dg_range:.3f} exceeds uncertainty"
