"""
tests/test_partition_lib.py
===========================
Unit tests for partition_lib.py and related modules.

Run with: pytest tests/
"""

import math
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from partition_lib import (
    threshold_partitions,
    squared_distance,
    delta_squared,
    delta,
    nearest_threshold_partitions,
    conjugate_partition,
    support_size,
    verify_parity_lemma,
)
from alkanes import (
    alkane_degree_sequences,
    is_star,
    is_asymmetric_double_star,
    is_lollipop,
    classify_tree,
    verify_theorem3,
    verify_path_formula,
)


# ---------------------------------------------------------------------------
# threshold_partitions
# ---------------------------------------------------------------------------

class TestThresholdPartitions:
    def test_n1(self):
        assert threshold_partitions(1) == [(0,)]

    def test_n2_count(self):
        # 2^1 = 2 threshold partitions on 2 vertices
        tps = threshold_partitions(2)
        assert len(tps) == 2

    def test_n3_count(self):
        # 2^2 = 4, but some may be duplicates
        tps = threshold_partitions(3)
        assert len(tps) == 4

    def test_n4_count(self):
        tps = threshold_partitions(4)
        assert len(tps) == 8  # 2^3 = 8

    def test_sorted_descending(self):
        for n in range(1, 8):
            for tau in threshold_partitions(n):
                assert list(tau) == sorted(tau, reverse=True), \
                    f"tau={tau} not sorted descending"

    def test_length(self):
        for n in range(1, 8):
            for tau in threshold_partitions(n):
                assert len(tau) == n

    def test_non_negative(self):
        for n in range(1, 8):
            for tau in threshold_partitions(n):
                assert all(x >= 0 for x in tau)

    def test_even_weight(self):
        """All threshold partitions have even weight (degree sequence of a graph)."""
        for n in range(1, 8):
            for tau in threshold_partitions(n):
                assert sum(tau) % 2 == 0, f"Odd weight: {tau}"

    def test_star_partition_present(self):
        """The star K_{1,n-1} partition (n-1,1,...,1) is always in TP(n)."""
        for n in range(2, 8):
            star = tuple([n - 1] + [1] * (n - 1))
            assert star in threshold_partitions(n), \
                f"Star partition {star} not in TP({n})"

    def test_all_zeros_present(self):
        """The empty graph (0,...,0) is always in TP(n)."""
        for n in range(1, 8):
            zeros = tuple([0] * n)
            assert zeros in threshold_partitions(n)


# ---------------------------------------------------------------------------
# squared_distance
# ---------------------------------------------------------------------------

class TestSquaredDistance:
    def test_identical(self):
        assert squared_distance((3, 2, 1), (3, 2, 1)) == 0

    def test_known(self):
        assert squared_distance((3, 2, 1, 0), (2, 2, 2, 1)) == 1 + 0 + 1 + 1

    def test_symmetry(self):
        d1 = (3, 2, 1, 0)
        d2 = (2, 2, 2, 1)
        assert squared_distance(d1, d2) == squared_distance(d2, d1)


# ---------------------------------------------------------------------------
# delta_squared and delta
# ---------------------------------------------------------------------------

class TestDelta:
    def test_star_is_zero(self):
        """Stars are threshold graphs, so delta = 0."""
        for n in range(2, 8):
            d = tuple([n - 1] + [1] * (n - 1))
            assert delta_squared(d) == 0

    def test_parity_lemma(self):
        """delta(G)^2 is always even."""
        test_cases = [
            (2, 2, 1, 1),
            (3, 2, 1, 1, 1),
            (2, 2, 2, 2, 1, 1),
            (3, 2, 2, 2, 1, 1, 1),
            (2, 2, 2, 2, 2, 2, 1, 1),
        ]
        for d in test_cases:
            d2 = delta_squared(d)
            assert d2 % 2 == 0, f"delta_squared({d}) = {d2} is odd"

    def test_cycle_formula(self):
        """Theorem 1: delta(C_n)^2 = 4n - 14 for n >= 4."""
        for n in range(4, 12):
            d = tuple([2] * n)
            assert delta_squared(d) == 4 * n - 14, \
                f"Cycle formula failed at n={n}"

    def test_path_formula(self):
        """Theorem 4: delta(P_n)^2 = 4(n-5) for n >= 6."""
        for n in range(6, 12):
            d = tuple([2] * (n - 2) + [1, 1])
            assert delta_squared(d) == 4 * (n - 5), \
                f"Path formula failed at n={n}"

    def test_delta_nonneg(self):
        """delta is always non-negative."""
        test_cases = [
            (2, 2, 1, 1),
            (3, 2, 1, 1, 1),
            (4, 1, 1, 1, 1),
        ]
        for d in test_cases:
            assert delta(d) >= 0

    def test_three_methylhexane(self):
        """Worked example: 3-methylhexane has delta^2 = 4."""
        d = (3, 2, 2, 2, 1, 1, 1)
        assert delta_squared(d) == 4

    def test_cyclohexane(self):
        """Worked example: cyclohexane has delta^2 = 10 = 4(6)-14."""
        d = (2, 2, 2, 2, 2, 2)
        assert delta_squared(d) == 10


# ---------------------------------------------------------------------------
# nearest_threshold_partitions
# ---------------------------------------------------------------------------

class TestNearestThresholdPartitions:
    def test_returns_list(self):
        d = (2, 2, 2, 2, 1, 1)
        result = nearest_threshold_partitions(d)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_all_achieve_minimum(self):
        d = (3, 2, 2, 2, 1, 1, 1)
        result = nearest_threshold_partitions(d)
        best = delta_squared(d)
        for tau in result:
            assert squared_distance(tau, d) == best

    def test_neopentane_is_threshold(self):
        """Neopentane (star K_{1,4}) is itself threshold; tau* = d."""
        d = (4, 1, 1, 1, 1)
        achievers = nearest_threshold_partitions(d)
        assert d in achievers
        assert delta_squared(d) == 0


# ---------------------------------------------------------------------------
# conjugate_partition
# ---------------------------------------------------------------------------

class TestConjugatePartition:
    def test_simple(self):
        # (3, 2, 1) -> conjugate is (3, 2, 1)
        assert conjugate_partition((3, 2, 1)) == (3, 2, 1)

    def test_n_hexane(self):
        # d = (2,2,2,2,1,1): d*_1=6, d*_2=4
        d = (2, 2, 2, 2, 1, 1)
        conj = conjugate_partition(d)
        assert conj[0] == 6   # all 6 vertices have degree >= 1
        assert conj[1] == 4   # 4 vertices have degree >= 2

    def test_ring_junction_count(self):
        """d*_3 counts ring-junction (degree >= 3) carbons."""
        # 3-methylhexane: one carbon has degree 3, rest <= 2
        d = (3, 2, 2, 2, 1, 1, 1)
        conj = conjugate_partition(d)
        assert conj[2] == 1   # exactly 1 vertex with degree >= 3


# ---------------------------------------------------------------------------
# Tree family classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_star(self):
        for n in range(2, 8):
            d = tuple([n - 1] + [1] * (n - 1))
            assert is_star(d)
            assert classify_tree(d) == "star"

    def test_not_star(self):
        d = (2, 2, 1, 1)
        assert not is_star(d)

    def test_asymmetric_double_star(self):
        for n in range(4, 9):
            d = tuple([n - 2, 2] + [1] * (n - 2))
            assert is_asymmetric_double_star(d), f"Failed for n={n}"

    def test_lollipop(self):
        for n in range(5, 10):
            d = tuple([n - 3, 2, 2] + [1] * (n - 3))
            assert is_lollipop(d), f"Failed for n={n}"

    def test_general_path(self):
        d = (2, 2, 2, 2, 2, 1, 1)  # P_7
        assert classify_tree(d) == "general"

    def test_neopentane_is_star(self):
        d = (4, 1, 1, 1, 1)
        assert is_star(d)


# ---------------------------------------------------------------------------
# Theorem verification
# ---------------------------------------------------------------------------

class TestTheorems:
    def test_theorem3_n13(self):
        """Theorem 3 holds for all alkane degree sequences up to n=13."""
        result = verify_theorem3(max_n=13)
        assert result["passed"], \
            f"Theorem 3 failures: {result['failures']}"

    def test_path_formula_n15(self):
        """Path formula delta(P_n)^2 = 4(n-5) holds for n=6..15."""
        result = verify_path_formula(max_n=15)
        assert result["passed"], \
            f"Path formula failures: {result['failures']}"

    def test_delta_squared_always_even(self):
        """Parity lemma holds for all alkane degree sequences up to n=10."""
        from alkanes import compute_delta_table
        rows = compute_delta_table(max_n=10)
        failures = [r for r in rows if r["delta2"] % 2 != 0]
        assert len(failures) == 0, \
            f"Parity failures: {[(r['d'], r['delta2']) for r in failures]}"

    def test_theorem3_delta_4_when_p_geq_4(self):
        """Every alkane with >= 4 internal carbons has delta^2 >= 4."""
        from alkanes import compute_delta_table
        rows = compute_delta_table(max_n=12)
        failures = [
            r for r in rows
            if r["p"] >= 4
            and r["family"] == "general"
            and r["delta2"] < 4
        ]
        assert len(failures) == 0, \
            f"Theorem 3(iii) failures: {[(r['d'], r['delta2']) for r in failures]}"

    def test_alkane_count(self):
        """Correct number of unique degree sequences for small n."""
        expected = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 7, 9: 8}
        for n, exp_count in expected.items():
            seqs = alkane_degree_sequences(n)
            assert len(seqs) == exp_count, \
                f"n={n}: expected {exp_count} seqs, got {len(seqs)}"


# ---------------------------------------------------------------------------
# support_size
# ---------------------------------------------------------------------------

class TestSupportSize:
    def test_all_zero(self):
        assert support_size((0, 0, 0, 0)) == 0

    def test_mixed(self):
        assert support_size((3, 2, 1, 0, 0)) == 3

    def test_all_positive(self):
        assert support_size((4, 3, 2, 1)) == 4
