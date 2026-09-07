from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from .state_identity import (
    IdentityThresholds,
    StateRelation,
    compare_states,
)
from .state_scout import ScoutSolution


@dataclass
class CandidateGroup:
    scout_group_id: str
    spin: int
    multiplicity: int
    members: List[ScoutSolution] = field(
        default_factory=list
    )

    @property
    def representative(self) -> ScoutSolution:
        return min(
            self.members,
            key=lambda x: x.energy_hartree,
        )

    @property
    def equivalent_guesses(self) -> List[str]:
        guesses = set()

        for member in self.members:
            guesses.update(
                member.equivalent_guesses
            )

        return sorted(guesses)


def _pair_relation(
    a: ScoutSolution,
    b: ScoutSolution,
    thresholds: IdentityThresholds,
) -> StateRelation:

    comparison = compare_states(
        energy_a_hartree=
            a.energy_hartree,
        energy_b_hartree=
            b.energy_hartree,
        s2_a=a.s2,
        s2_b=b.s2,
        dm_orth_a=a.density_orth,
        dm_orth_b=b.density_orth,
        spin_a=a.spin,
        spin_b=b.spin,
        thresholds=thresholds,
    )

    return comparison.relation


def _same_as_entire_group(
    candidate: ScoutSolution,
    group: CandidateGroup,
    thresholds: IdentityThresholds,
) -> bool:
    """
    Conservative complete-link rule.

    A new solution is merged into a group only if it is SAME_STATE
    relative to every existing member of that group.
    """
    return all(
        _pair_relation(
            candidate,
            member,
            thresholds,
        )
        == StateRelation.SAME_STATE
        for member in group.members
    )


def build_candidate_groups(
    solutions: Iterable[ScoutSolution],
    *,
    thresholds: IdentityThresholds =
        IdentityThresholds(),
) -> List[CandidateGroup]:
    """
    Group fixed-geometry SCF solutions conservatively.

    Different spin sectors can never be grouped together.

    AMBIGUOUS solutions remain separate.
    """
    ordered = sorted(
        list(solutions),
        key=lambda x: (
            int(x.spin),
            float(x.energy_hartree),
            str(x.origin_guess),
        ),
    )

    groups: List[CandidateGroup] = []

    for solution in ordered:

        compatible = [
            group
            for group in groups
            if (
                group.spin == solution.spin
                and _same_as_entire_group(
                    solution,
                    group,
                    thresholds,
                )
            )
        ]

        if compatible:

            # Normally exactly one group should satisfy the
            # complete-link SAME_STATE condition.
            #
            # If more than one does, use the energetically closest
            # representative deterministically.
            target = min(
                compatible,
                key=lambda group:
                    abs(
                        group.representative.energy_hartree
                        - solution.energy_hartree
                    ),
            )

            target.members.append(
                solution
            )

        else:

            groups.append(
                CandidateGroup(
                    scout_group_id="",
                    spin=int(solution.spin),
                    multiplicity=
                        int(solution.multiplicity),
                    members=[solution],
                )
            )

    # Assign deterministic geometry-local group labels.
    by_spin = {}

    for group in groups:
        by_spin.setdefault(
            group.spin,
            [],
        ).append(group)

    for spin, spin_groups in by_spin.items():

        ordered_spin = sorted(
            spin_groups,
            key=lambda g:
                g.representative.energy_hartree,
        )

        for rank, group in enumerate(
            ordered_spin,
            start=1,
        ):

            group.scout_group_id = (
                f"S{spin}_G{rank:02d}"
            )

    return sorted(
        groups,
        key=lambda g:
            g.representative.energy_hartree,
    )


def compare_groups(
    a: CandidateGroup,
    b: CandidateGroup,
    *,
    thresholds: IdentityThresholds =
        IdentityThresholds(),
) -> StateRelation:
    """
    Conservative group-level relation.

    - different spin -> DISTINCT_STATE
    - all member pairs SAME -> SAME_STATE
    - all member pairs DISTINCT -> DISTINCT_STATE
    - any mixed or ambiguous evidence -> AMBIGUOUS
    """
    if a.spin != b.spin:
        return StateRelation.DISTINCT_STATE

    relations = []

    for ma in a.members:
        for mb in b.members:

            relations.append(
                _pair_relation(
                    ma,
                    mb,
                    thresholds,
                )
            )

    if relations and all(
        r == StateRelation.SAME_STATE
        for r in relations
    ):
        return StateRelation.SAME_STATE

    if relations and all(
        r == StateRelation.DISTINCT_STATE
        for r in relations
    ):
        return StateRelation.DISTINCT_STATE

    return StateRelation.AMBIGUOUS


def ambiguous_group_pairs(
    groups: Iterable[CandidateGroup],
    *,
    thresholds: IdentityThresholds =
        IdentityThresholds(),
) -> List[
    Tuple[
        CandidateGroup,
        CandidateGroup,
    ]
]:
    groups = list(groups)

    output = []

    for i, a in enumerate(groups):

        for b in groups[i + 1:]:

            if compare_groups(
                a,
                b,
                thresholds=thresholds,
            ) == StateRelation.AMBIGUOUS:

                output.append(
                    (a, b)
                )

    return output
