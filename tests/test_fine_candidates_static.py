import unittest

from diatomic_ea_v09.canonical_dedup import (
    CanonicalBranchRelation,
)
from diatomic_ea_v09.fine_candidates import (
    fine_relation_from_counts,
)


class FineCandidateStaticTests(
    unittest.TestCase
):

    def test_same_fine_branch(self):

        relation = (
            fine_relation_from_counts(
                n_same=11,
                n_distinct=0,
                n_ambiguous=0,
                min_same_points=5,
            )
        )

        self.assertEqual(
            relation,
            CanonicalBranchRelation
            .SAME_BRANCH,
        )


    def test_distinct_contradiction_wins(self):

        relation = (
            fine_relation_from_counts(
                n_same=10,
                n_distinct=1,
                n_ambiguous=0,
                min_same_points=5,
            )
        )

        self.assertEqual(
            relation,
            CanonicalBranchRelation
            .DISTINCT_BRANCH,
        )


    def test_ambiguous_is_retained(self):

        relation = (
            fine_relation_from_counts(
                n_same=0,
                n_distinct=0,
                n_ambiguous=11,
                min_same_points=5,
            )
        )

        self.assertEqual(
            relation,
            CanonicalBranchRelation
            .AMBIGUOUS_BRANCH,
        )


if __name__ == "__main__":
    unittest.main()


class FineEnergyOrderingTests(
    unittest.TestCase
):

    def test_a_lower_everywhere(self):

        from diatomic_ea_v09.fine_candidates import (
            fine_energy_ordering,
        )

        self.assertEqual(
            fine_energy_ordering(
                [
                    0.71,
                    0.73,
                    0.75,
                ]
            ),
            "A_LOWER_ALL",
        )


    def test_b_lower_everywhere(self):

        from diatomic_ea_v09.fine_candidates import (
            fine_energy_ordering,
        )

        self.assertEqual(
            fine_energy_ordering(
                [
                    -0.71,
                    -0.73,
                    -0.75,
                ]
            ),
            "B_LOWER_ALL",
        )


    def test_crossing_is_unresolved(self):

        from diatomic_ea_v09.fine_candidates import (
            fine_energy_ordering,
        )

        self.assertEqual(
            fine_energy_ordering(
                [
                    0.10,
                    0.02,
                    -0.03,
                ]
            ),
            "CROSS_OR_TOUCH",
        )


    def test_exact_touch_is_unresolved(self):

        from diatomic_ea_v09.fine_candidates import (
            fine_energy_ordering,
        )

        self.assertEqual(
            fine_energy_ordering(
                [
                    0.10,
                    0.00,
                    0.10,
                ]
            ),
            "CROSS_OR_TOUCH",
        )
