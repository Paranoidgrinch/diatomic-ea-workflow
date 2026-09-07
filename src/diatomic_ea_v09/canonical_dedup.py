from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from .ground_state import (
    CanonicalBranch,
)
from .state_identity import (
    StateRelation,
    compare_states,
)


class CanonicalBranchRelation(
    str,
    Enum,
):
    SAME_BRANCH = "SAME_BRANCH"
    DISTINCT_BRANCH = "DISTINCT_BRANCH"
    AMBIGUOUS_BRANCH = "AMBIGUOUS_BRANCH"


@dataclass
class CanonicalBranchComparison:
    branch_a: str
    branch_b: str

    relation: CanonicalBranchRelation

    n_common_points: int
    n_same: int
    n_distinct: int
    n_ambiguous: int

    compared_r_values: List[float]


@dataclass
class CanonicalBranchGroup:
    group_id: str

    representative: CanonicalBranch
    members: List[CanonicalBranch]

    @property
    def member_ids(self):
        return [
            branch.branch_id
            for branch in self.members
        ]


def _point_map(
    branch: CanonicalBranch,
):
    return {
        round(
            point.r_A,
            10,
        ): point

        for point in branch.canonical_points

        if (
            point.converged
            and point.energy_hartree
            is not None
            and point.density_orth
            is not None
        )
    }


def compare_canonical_branches(
    branch_a: CanonicalBranch,
    branch_b: CanonicalBranch,
    *,
    minimum_window_A: float = 0.20,
    min_same_points: int = 3,
):
    """
    Compare stability-canonicalized coarse branches near both minima.

    Conservative rule:
    - any DISTINCT_STATE evidence near either minimum => DISTINCT_BRANCH
    - otherwise >= min_same_points SAME_STATE comparisons => SAME_BRANCH
    - otherwise => AMBIGUOUS_BRANCH

    This intentionally does not merge branches merely because they
    coalesce somewhere far from their minima.
    """

    if (
        not branch_a.valid
        or not branch_b.valid
        or branch_a.minimum is None
        or branch_b.minimum is None
    ):

        return CanonicalBranchComparison(
            branch_a=
                branch_a.branch_id,
            branch_b=
                branch_b.branch_id,

            relation=
                CanonicalBranchRelation.AMBIGUOUS_BRANCH,

            n_common_points=0,
            n_same=0,
            n_distinct=0,
            n_ambiguous=0,

            compared_r_values=[],
        )


    spin_a = (
        branch_a
        .source_branch
        .spin
    )

    spin_b = (
        branch_b
        .source_branch
        .spin
    )


    if spin_a != spin_b:

        return CanonicalBranchComparison(
            branch_a=
                branch_a.branch_id,
            branch_b=
                branch_b.branch_id,

            relation=
                CanonicalBranchRelation.DISTINCT_BRANCH,

            n_common_points=0,
            n_same=0,
            n_distinct=1,
            n_ambiguous=0,

            compared_r_values=[],
        )


    map_a = _point_map(
        branch_a
    )

    map_b = _point_map(
        branch_b
    )


    common = sorted(
        set(map_a)
        & set(map_b)
    )


    min_a = (
        branch_a.minimum.r_A
    )

    min_b = (
        branch_b.minimum.r_A
    )


    selected = [
        r
        for r in common
        if (
            abs(
                r - min_a
            )
            <= minimum_window_A
            + 1.0e-10

            or

            abs(
                r - min_b
            )
            <= minimum_window_A
            + 1.0e-10
        )
    ]


    relations = []

    for r in selected:

        a = map_a[r]
        b = map_b[r]

        comparison = compare_states(
            energy_a_hartree=
                a.energy_hartree,

            energy_b_hartree=
                b.energy_hartree,

            s2_a=
                a.s2,

            s2_b=
                b.s2,

            dm_orth_a=
                a.density_orth,

            dm_orth_b=
                b.density_orth,

            spin_a=
                spin_a,

            spin_b=
                spin_b,
        )

        relations.append(
            comparison.relation
        )


    n_same = sum(
        relation
        == StateRelation.SAME_STATE
        for relation in relations
    )

    n_distinct = sum(
        relation
        == StateRelation.DISTINCT_STATE
        for relation in relations
    )

    n_ambiguous = sum(
        relation
        == StateRelation.AMBIGUOUS
        for relation in relations
    )


    if n_distinct > 0:

        branch_relation = (
            CanonicalBranchRelation
            .DISTINCT_BRANCH
        )

    elif (
        n_same
        >= int(min_same_points)
    ):

        branch_relation = (
            CanonicalBranchRelation
            .SAME_BRANCH
        )

    else:

        branch_relation = (
            CanonicalBranchRelation
            .AMBIGUOUS_BRANCH
        )


    return CanonicalBranchComparison(
        branch_a=
            branch_a.branch_id,

        branch_b=
            branch_b.branch_id,

        relation=
            branch_relation,

        n_common_points=
            len(selected),

        n_same=
            n_same,

        n_distinct=
            n_distinct,

        n_ambiguous=
            n_ambiguous,

        compared_r_values=[
            float(r)
            for r in selected
        ],
    )


def deduplicate_canonical_branches(
    branches: Sequence[
        CanonicalBranch
    ],
    *,
    minimum_window_A: float = 0.20,
    min_same_points: int = 3,
):
    """
    Complete-link grouping of stability-canonicalized branches.

    A candidate joins an existing group only when it is SAME_BRANCH
    versus every member already in that group.

    Ambiguous evidence never causes an automatic merge.
    """

    valid = [
        branch
        for branch in branches
        if (
            branch.valid
            and branch.minimum
            is not None
        )
    ]


    valid = sorted(
        valid,
        key=lambda branch:
            branch.minimum
            .energy_hartree,
    )


    groups = []


    for branch in valid:

        compatible_groups = []

        for group in groups:

            comparisons = [
                compare_canonical_branches(
                    branch,
                    member,

                    minimum_window_A=
                        minimum_window_A,

                    min_same_points=
                        min_same_points,
                )

                for member
                in group.members
            ]


            if (
                comparisons
                and all(
                    comparison.relation
                    == CanonicalBranchRelation.SAME_BRANCH

                    for comparison
                    in comparisons
                )
            ):
                compatible_groups.append(
                    group
                )


        if compatible_groups:

            group = min(
                compatible_groups,
                key=lambda g:
                    g.representative
                    .minimum
                    .energy_hartree,
            )

            group.members.append(
                branch
            )

            group.representative = min(
                group.members,
                key=lambda b:
                    b.minimum
                    .energy_hartree,
            )

        else:

            groups.append(
                CanonicalBranchGroup(
                    group_id="",

                    representative=
                        branch,

                    members=[
                        branch
                    ],
                )
            )


    for index, group in enumerate(
        groups,
        start=1,
    ):

        group.group_id = (
            f"CANON_G{index:03d}"
        )


    return groups
