import unittest
from types import SimpleNamespace

from diatomic_ea_v09.canonical_dedup import (
    CanonicalBranchRelation,
)
from diatomic_ea_v09.ground_state_candidates import (
    select_ground_state_candidate_groups,
)


def fake_branch(
    branch_id,
    energy,
):
    return SimpleNamespace(
        branch_id=
            branch_id,

        valid=True,

        minimum=
            SimpleNamespace(
                energy_hartree=
                    energy,
            ),
    )


def fake_group(
    group_id,
    branch_id,
    energy,
):
    return SimpleNamespace(
        group_id=
            group_id,

        representative=
            fake_branch(
                branch_id,
                energy,
            ),
    )


class FakeComparison:

    def __init__(
        self,
        relation,
    ):
        self.relation = relation


class GroundStateCandidateTests(
    unittest.TestCase
):

    def test_ambiguous_competitor_is_retained(self):

        g1 = fake_group(
            "G1",
            "B1",
            -10.000,
        )

        g2 = fake_group(
            "G2",
            "B2",
            -9.999,
        )

        g3 = fake_group(
            "G3",
            "B3",
            -9.900,
        )


        def comparator(
            a,
            b,
            **kwargs,
        ):

            if b.branch_id == "B2":

                return FakeComparison(
                    CanonicalBranchRelation
                    .AMBIGUOUS_BRANCH
                )

            return FakeComparison(
                CanonicalBranchRelation
                .DISTINCT_BRANCH
            )


        result = (
            select_ground_state_candidate_groups(
                [
                    g1,
                    g2,
                    g3,
                ],

                comparator=
                    comparator,
            )
        )


        self.assertEqual(
            result.primary.group_id,
            "G1",
        )

        self.assertEqual(
            result.fine_candidate_ids,
            [
                "G1",
                "G2",
            ],
        )

        self.assertEqual(
            len(
                result.distinct_higher_groups
            ),
            1,
        )

        self.assertEqual(
            result.status,
            "COARSE_CANDIDATES_REQUIRE_FINE_RESOLUTION",
        )


    def test_unique_candidate(self):

        g1 = fake_group(
            "G1",
            "B1",
            -10.0,
        )

        g2 = fake_group(
            "G2",
            "B2",
            -9.0,
        )


        def comparator(
            a,
            b,
            **kwargs,
        ):

            return FakeComparison(
                CanonicalBranchRelation
                .DISTINCT_BRANCH
            )


        result = (
            select_ground_state_candidate_groups(
                [
                    g1,
                    g2,
                ],

                comparator=
                    comparator,
            )
        )


        self.assertEqual(
            result.fine_candidate_ids,
            [
                "G1"
            ],
        )

        self.assertEqual(
            result.status,
            "UNIQUE_COARSE_GROUND_STATE_CANDIDATE",
        )


if __name__ == "__main__":
    unittest.main()
