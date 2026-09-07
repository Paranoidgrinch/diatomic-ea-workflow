from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from .canonical_dedup import (
    CanonicalBranchComparison,
    CanonicalBranchGroup,
    CanonicalBranchRelation,
    compare_canonical_branches,
)


@dataclass
class CandidateGroupComparison:
    primary_group_id: str
    other_group_id: str

    relation: CanonicalBranchRelation

    primary_energy_hartree: float
    other_energy_hartree: float

    delta_energy_hartree: float

    branch_comparison: CanonicalBranchComparison


@dataclass
class GroundStateCandidateSet:
    status: str

    primary: CanonicalBranchGroup | None

    fine_candidates: List[
        CanonicalBranchGroup
    ]

    ambiguous_competitors: List[
        CanonicalBranchGroup
    ]

    same_after_dedup: List[
        CanonicalBranchGroup
    ]

    distinct_higher_groups: List[
        CanonicalBranchGroup
    ]

    comparisons: List[
        CandidateGroupComparison
    ]

    @property
    def fine_candidate_ids(self):
        return [
            group.group_id
            for group
            in self.fine_candidates
        ]


def select_ground_state_candidate_groups(
    groups: Sequence[
        CanonicalBranchGroup
    ],
    *,
    minimum_window_A: float = 0.20,
    min_same_points: int = 3,
    comparator: Callable = (
        compare_canonical_branches
    ),
) -> GroundStateCandidateSet:
    """
    Select the groups that must survive from coarse PECs into
    fine ground-state refinement.

    Universal conservative policy:

    1. The lowest canonical coarse group is the primary candidate.

    2. A higher group that is DISTINCT_BRANCH from the primary is
       not required for ground-state fine refinement.

    3. Any AMBIGUOUS_BRANCH relative to the primary is retained and
       receives its own fine PEC.

    4. SAME_BRANCH surviving the previous deduplication stage is also
       retained and explicitly reported. This should be rare and means
       that complete-link grouping deliberately prevented an unsafe
       merge elsewhere in the group.

    No energy-gap threshold is used here.

    Thus no empirically tuned "within X meV" rule is required.
    """

    eligible = [
        group
        for group in groups
        if (
            group.representative.valid
            and
            group.representative.minimum
            is not None
        )
    ]

    if not eligible:

        return GroundStateCandidateSet(
            status=
                "NO_VALID_GROUND_STATE_CANDIDATE",

            primary=None,

            fine_candidates=[],
            ambiguous_competitors=[],
            same_after_dedup=[],
            distinct_higher_groups=[],
            comparisons=[],
        )


    ordered = sorted(
        eligible,
        key=lambda group:
            group.representative
            .minimum
            .energy_hartree,
    )


    primary = ordered[0]

    fine_candidates = [
        primary
    ]

    ambiguous = []
    same_after_dedup = []
    distinct = []
    comparisons = []


    primary_branch = (
        primary.representative
    )

    primary_energy = (
        primary_branch
        .minimum
        .energy_hartree
    )


    for group in ordered[1:]:

        other_branch = (
            group.representative
        )

        comparison = comparator(
            primary_branch,
            other_branch,

            minimum_window_A=
                minimum_window_A,

            min_same_points=
                min_same_points,
        )

        other_energy = (
            other_branch
            .minimum
            .energy_hartree
        )


        comparisons.append(
            CandidateGroupComparison(
                primary_group_id=
                    primary.group_id,

                other_group_id=
                    group.group_id,

                relation=
                    comparison.relation,

                primary_energy_hartree=
                    primary_energy,

                other_energy_hartree=
                    other_energy,

                delta_energy_hartree=
                    (
                        other_energy
                        - primary_energy
                    ),

                branch_comparison=
                    comparison,
            )
        )


        if (
            comparison.relation
            == CanonicalBranchRelation
            .AMBIGUOUS_BRANCH
        ):

            ambiguous.append(
                group
            )

            fine_candidates.append(
                group
            )


        elif (
            comparison.relation
            == CanonicalBranchRelation
            .SAME_BRANCH
        ):

            same_after_dedup.append(
                group
            )

            fine_candidates.append(
                group
            )


        else:

            distinct.append(
                group
            )


    if same_after_dedup:

        status = (
            "COARSE_CANDIDATES_REQUIRE_FINE_RESOLUTION"
        )

    elif ambiguous:

        status = (
            "COARSE_CANDIDATES_REQUIRE_FINE_RESOLUTION"
        )

    else:

        status = (
            "UNIQUE_COARSE_GROUND_STATE_CANDIDATE"
        )


    return GroundStateCandidateSet(
        status=
            status,

        primary=
            primary,

        fine_candidates=
            fine_candidates,

        ambiguous_competitors=
            ambiguous,

        same_after_dedup=
            same_after_dedup,

        distinct_higher_groups=
            distinct,

        comparisons=
            comparisons,
    )
