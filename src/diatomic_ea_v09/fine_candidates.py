from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .canonical_dedup import (
    CanonicalBranchGroup,
    CanonicalBranchRelation,
)
from .fine_pec import (
    FineGridPolicy,
    FinePECSummary,
    run_fine_pec,
)
from .fine_qc import (
    FinePECQCResult,
    qc_fine_pec,
)
from .ground_state import (
    point_at_r,
    scout_solution_from_mf,
)
from .model import (
    MoleculeSpec,
    SCFSettings,
)
from .molecule import (
    build_molecule,
)
from .scf import (
    HARTREE_TO_EV,
    run_uks,
)
from .stability import (
    optimize_internal_stability,
)
from .state_identity import (
    StateRelation,
    compare_states,
)


@dataclass
class FineCandidateOutcome:
    group: CanonicalBranchGroup

    status: str

    fine: FinePECSummary | None
    qc: FinePECQCResult | None

    seed_initially_stable: bool | None
    seed_finally_stable: bool | None
    seed_stability_delta_meV: float | None
    seed_relation: StateRelation | None

    @property
    def group_id(self):
        return self.group.group_id

    @property
    def branch_id(self):
        return self.group.representative.branch_id

    @property
    def fitted_energy_hartree(self):
        if self.fine is None:
            return None

        if self.fine.fitted_min_energy_hartree is not None:
            return self.fine.fitted_min_energy_hartree

        return self.fine.discrete_min_energy_hartree


@dataclass
class FineCandidateComparison:
    group_a: str
    group_b: str

    relation: CanonicalBranchRelation

    n_common_points: int
    n_same: int
    n_distinct: int
    n_ambiguous: int

    delta_min_energy_meV: float

    min_point_gap_meV: float | None
    max_point_gap_meV: float | None
    ordering: str

    compared_r_values: List[float]


@dataclass
class FineGroundStateResolution:
    status: str

    primary: FineCandidateOutcome | None

    same_candidates: List[FineCandidateOutcome]
    distinct_candidates: List[FineCandidateOutcome]

    ambiguous_candidates: List[FineCandidateOutcome]

    ambiguous_dominated_candidates: List[
        FineCandidateOutcome
    ]

    ambiguous_unresolved_candidates: List[
        FineCandidateOutcome
    ]

    comparisons: List[FineCandidateComparison]


def fine_relation_from_counts(
    *,
    n_same: int,
    n_distinct: int,
    n_ambiguous: int,
    min_same_points: int = 5,
):
    if n_distinct > 0:
        return (
            CanonicalBranchRelation
            .DISTINCT_BRANCH
        )

    if n_same >= int(min_same_points):
        return (
            CanonicalBranchRelation
            .SAME_BRANCH
        )

    return (
        CanonicalBranchRelation
        .AMBIGUOUS_BRANCH
    )



@dataclass
class LocalWellQC:
    """
    QC of exactly the local PEC points used to determine the
    quadratic minimum.

    This is deliberately distinct from FinePECQCResult, which
    continues to describe the entire Fine PEC window.
    """

    status: str

    r_low_A: float | None
    r_high_A: float | None

    n_points: int
    n_pass: int


def _quadratic_fit_r_values(
    fine: FinePECSummary,
):
    """
    Return the R points actually used by the local quadratic fit.

    New FinePECSummary objects persist the exact fit bounds.

    Development checkpoints created before those fields existed are
    supported by reconstructing the contiguous subset around the
    discrete minimum using quadratic_points_used.
    """

    good = sorted(
        [
            point
            for point in fine.points
            if (
                point.converged
                and point.energy_hartree
                is not None
            )
        ],
        key=lambda point:
            point.r_A,
    )

    if not good:
        return []


    r_low = getattr(
        fine,
        "quadratic_r_low_A",
        None,
    )

    r_high = getattr(
        fine,
        "quadratic_r_high_A",
        None,
    )


    if (
        r_low is not None
        and r_high is not None
    ):
        return [
            round(
                float(point.r_A),
                10,
            )
            for point in good
            if (
                float(r_low) - 1.0e-10
                <= float(point.r_A)
                <= float(r_high) + 1.0e-10
            )
        ]


    n_points = int(
        fine.quadratic_points_used
    )

    if n_points < 3:
        return []


    minimum_index = min(
        range(len(good)),
        key=lambda i:
            abs(
                float(good[i].r_A)
                - float(
                    fine.discrete_min_r_A
                )
            ),
    )


    half = (
        n_points - 1
    ) // 2


    lo = max(
        0,
        minimum_index - half,
    )

    hi = min(
        len(good),
        lo + n_points,
    )

    lo = max(
        0,
        hi - n_points,
    )


    subset = good[
        lo:hi
    ]


    return [
        round(
            float(point.r_A),
            10,
        )
        for point in subset
    ]


def evaluate_local_well_qc(
    *,
    fine: FinePECSummary | None,
    qc: FinePECQCResult | None,
) -> LocalWellQC:
    """
    Require every point actually used in the local quadratic minimum
    fit to PASS the existing strict pointwise Fine QC.

    No identity/stability threshold is changed.

    Problems outside this local fit region remain represented by the
    full FinePECQCResult as FULL_BRANCH_QC review information.
    """

    if (
        fine is None
        or qc is None
    ):
        return LocalWellQC(
            status=
                "QC_FAIL_LOCAL_WELL_DATA",

            r_low_A=None,
            r_high_A=None,

            n_points=0,
            n_pass=0,
        )


    selected_r = (
        _quadratic_fit_r_values(
            fine
        )
    )


    if len(selected_r) < 3:

        return LocalWellQC(
            status=
                "QC_FAIL_LOCAL_WELL_DATA",

            r_low_A=(
                None
                if not selected_r
                else min(selected_r)
            ),

            r_high_A=(
                None
                if not selected_r
                else max(selected_r)
            ),

            n_points=
                len(selected_r),

            n_pass=0,
        )


    qc_map = {
        round(
            float(point.r_A),
            10,
        ): point

        for point in qc.points
    }


    selected = [
        qc_map[r]
        for r in selected_r
        if r in qc_map
    ]


    if (
        len(selected)
        != len(selected_r)
    ):
        return LocalWellQC(
            status=
                "QC_FAIL_LOCAL_WELL_DATA",

            r_low_A=
                min(selected_r),

            r_high_A=
                max(selected_r),

            n_points=
                len(selected),

            n_pass=sum(
                point.status
                == "PASS"
                for point in selected
            ),
        )


    n_pass = sum(
        point.status == "PASS"
        for point in selected
    )


    clean = (
        n_pass
        == len(selected)
    )


    return LocalWellQC(
        status=(
            "PASS"
            if clean
            else "QC_FAIL_LOCAL_WELL"
        ),

        r_low_A=
            min(selected_r),

        r_high_A=
            max(selected_r),

        n_points=
            len(selected),

        n_pass=
            n_pass,
    )


def fine_candidate_effective_status(
    outcome: FineCandidateOutcome,
):
    """
    Return the status used for Fine GS resolution.

    Legacy development checkpoints that used
    QC_FAIL_FINE_POINTWISE for any non-clean 41-point PEC are
    reclassified without rerunning SCF only when:

      * the full PEC merely requires review, AND
      * the actual local quadratic-fit well passes strict pointwise QC.

    Genuine local-well failures remain failures.
    """

    if outcome.status in (
        "PASS",
        "PASS_LOCAL_WELL_FULL_BRANCH_REVIEW",
    ):
        return outcome.status


    qc = getattr(
        outcome,
        "qc",
        None,
    )

    fine = getattr(
        outcome,
        "fine",
        None,
    )


    if (
        outcome.status
        == "QC_FAIL_FINE_POINTWISE"

        and qc is not None
        and qc.status
        == "QC_REVIEW_REQUIRED"
    ):
        local = (
            evaluate_local_well_qc(
                fine=fine,
                qc=qc,
            )
        )

        if local.status == "PASS":

            return (
                "PASS_LOCAL_WELL_"
                "FULL_BRANCH_REVIEW"
            )


    return outcome.status


def fine_candidate_is_resolvable(
    outcome: FineCandidateOutcome,
):
    return (
        fine_candidate_effective_status(
            outcome
        )
        in (
            "PASS",
            "PASS_LOCAL_WELL_FULL_BRANCH_REVIEW",
        )
    )


def refine_candidate_group(
    *,
    group: CanonicalBranchGroup,

    atom: str,
    ligand: str,

    basis: str,
    settings: SCFSettings,

    fine_policy: FineGridPolicy =
        FineGridPolicy(),

    perform_pointwise_qc: bool,

    max_memory_mb: int = 2000,
):
    """
    Refine one stability-canonicalized coarse GS candidate.

    The coarse minimum is reconstructed from its stored density and
    checked for internal stability before the dense PEC is started.

    A candidate whose coarse seed changes electronic identity during
    this check is NOT silently propagated.
    """

    branch = group.representative

    if (
        not branch.valid
        or branch.minimum is None
    ):
        return FineCandidateOutcome(
            group=group,

            status=
                "QC_FAIL_INVALID_COARSE_BRANCH",

            fine=None,
            qc=None,

            seed_initially_stable=None,
            seed_finally_stable=None,
            seed_stability_delta_meV=None,
            seed_relation=None,
        )


    center_r = (
        branch.minimum.r_A
    )

    coarse_point = point_at_r(
        branch.canonical_points,
        center_r,
    )

    if (
        coarse_point is None
        or coarse_point.density_ao is None
    ):
        return FineCandidateOutcome(
            group=group,

            status=
                "QC_FAIL_COARSE_SEED",

            fine=None,
            qc=None,

            seed_initially_stable=None,
            seed_finally_stable=None,
            seed_stability_delta_meV=None,
            seed_relation=None,
        )


    spec = MoleculeSpec(
        atom=atom,
        ligand=ligand,

        charge=
            branch.source_branch.charge,

        spin=
            branch.source_branch.spin,

        basis=basis,

        r_A=
            center_r,

        max_memory_mb=
            max_memory_mb,
    )


    mol = build_molecule(
        spec
    )


    reconstructed = run_uks(
        mol,
        settings,
        dm0=
            coarse_point.density_ao,
    )


    if not reconstructed.converged:

        return FineCandidateOutcome(
            group=group,

            status=
                "QC_FAIL_COARSE_SEED_SCF",

            fine=None,
            qc=None,

            seed_initially_stable=None,
            seed_finally_stable=None,
            seed_stability_delta_meV=None,
            seed_relation=None,
        )


    stability = (
        optimize_internal_stability(
            reconstructed.mf,
            settings,
            max_iterations=6,
        )
    )


    if not stability.finally_stable:

        return FineCandidateOutcome(
            group=group,

            status=
                "QC_FAIL_COARSE_SEED_STABILITY",

            fine=None,
            qc=None,

            seed_initially_stable=
                stability.initially_stable,

            seed_finally_stable=
                stability.finally_stable,

            seed_stability_delta_meV=
                stability.total_delta_energy_meV,

            seed_relation=None,
        )


    template = (
        branch.source_branch
        .seed_solution
    )


    stable_seed = (
        scout_solution_from_mf(
            mf=
                stability.final_mf,

            template=
                template,

            origin_guess=
                "fine_candidate_seed",
        )
    )

    stable_seed.r_A = (
        center_r
    )


    seed_cmp = compare_states(
        energy_a_hartree=
            coarse_point.energy_hartree,

        energy_b_hartree=
            stable_seed.energy_hartree,

        s2_a=
            coarse_point.s2,

        s2_b=
            stable_seed.s2,

        dm_orth_a=
            coarse_point.density_orth,

        dm_orth_b=
            stable_seed.density_orth,

        spin_a=
            branch.source_branch.spin,

        spin_b=
            branch.source_branch.spin,
    )


    if (
        seed_cmp.relation
        != StateRelation.SAME_STATE
    ):

        return FineCandidateOutcome(
            group=group,

            status=
                "COARSE_SEED_CHANGED_IDENTITY",

            fine=None,
            qc=None,

            seed_initially_stable=
                stability.initially_stable,

            seed_finally_stable=
                stability.finally_stable,

            seed_stability_delta_meV=
                stability.total_delta_energy_meV,

            seed_relation=
                seed_cmp.relation,
        )


    fine = run_fine_pec(
        branch_id=(
            group.group_id
            + "_FINE"
        ),

        atom=atom,
        ligand=ligand,

        seed=
            stable_seed,

        settings=
            settings,

        center_r_A=
            center_r,

        policy=
            fine_policy,

        max_memory_mb=
            max_memory_mb,
    )


    if fine.minimum_at_boundary:

        return FineCandidateOutcome(
            group=group,

            status=
                "QC_FAIL_FINE_BOUNDARY_MINIMUM",

            fine=fine,
            qc=None,

            seed_initially_stable=
                stability.initially_stable,

            seed_finally_stable=
                stability.finally_stable,

            seed_stability_delta_meV=
                stability.total_delta_energy_meV,

            seed_relation=
                seed_cmp.relation,
        )


    qc = None

    if perform_pointwise_qc:

        qc = qc_fine_pec(
            points=
                fine.points,

            atom=atom,
            ligand=ligand,

            charge=
                branch.source_branch.charge,

            spin=
                branch.source_branch.spin,

            basis=basis,
            settings=settings,

            max_memory_mb=
                max_memory_mb,

            max_stability_iterations=6,
        )

        local_well_qc = (
            evaluate_local_well_qc(
                fine=fine,
                qc=qc,
            )
        )


        if (
            local_well_qc.status
            != "PASS"
        ):

            return FineCandidateOutcome(
                group=group,

                status=
                    "QC_FAIL_LOCAL_WELL",

                fine=fine,
                qc=qc,

                seed_initially_stable=
                    stability.initially_stable,

                seed_finally_stable=
                    stability.finally_stable,

                seed_stability_delta_meV=
                    stability.total_delta_energy_meV,

                seed_relation=
                    seed_cmp.relation,
            )


        if qc.status != "PASS":

            return FineCandidateOutcome(
                group=group,

                status=(
                    "PASS_LOCAL_WELL_"
                    "FULL_BRANCH_REVIEW"
                ),

                fine=fine,
                qc=qc,

                seed_initially_stable=
                    stability.initially_stable,

                seed_finally_stable=
                    stability.finally_stable,

                seed_stability_delta_meV=
                    stability.total_delta_energy_meV,

                seed_relation=
                    seed_cmp.relation,
            )


    return FineCandidateOutcome(
        group=group,

        status="PASS",

        fine=fine,
        qc=qc,

        seed_initially_stable=
            stability.initially_stable,

        seed_finally_stable=
            stability.finally_stable,

        seed_stability_delta_meV=
            stability.total_delta_energy_meV,

        seed_relation=
            seed_cmp.relation,
    )


def fine_energy_ordering(
    gaps_meV,
):
    """
    Determine energetic ordering from pointwise

        gap = E_B - E_A

    values.

    No empirical energy threshold is used.

    A_LOWER_ALL:
        A is below B at every compared point.

    B_LOWER_ALL:
        B is below A at every compared point.

    CROSS_OR_TOUCH:
        sign changes or at least one exact equality occurs.

    NO_COMMON_POINTS:
        no comparison data.
    """

    gaps = [
        float(x)
        for x in gaps_meV
    ]

    if not gaps:
        return "NO_COMMON_POINTS"

    if all(
        gap > 0.0
        for gap in gaps
    ):
        return "A_LOWER_ALL"

    if all(
        gap < 0.0
        for gap in gaps
    ):
        return "B_LOWER_ALL"

    return "CROSS_OR_TOUCH"


def compare_fine_candidates(
    a: FineCandidateOutcome,
    b: FineCandidateOutcome,
    *,
    identity_half_window_A: float = 0.05,
    min_same_points: int = 5,
):
    if (
        a.fine is None
        or b.fine is None
    ):
        raise ValueError(
            "Both candidates require Fine PECs."
        )


    branch_a = (
        a.group.representative
    )

    branch_b = (
        b.group.representative
    )


    spin_a = (
        branch_a.source_branch.spin
    )

    spin_b = (
        branch_b.source_branch.spin
    )


    ea = (
        a.fitted_energy_hartree
    )

    eb = (
        b.fitted_energy_hartree
    )


    delta_min_meV = (
        eb - ea
    ) * HARTREE_TO_EV * 1000.0


    if spin_a != spin_b:

        return FineCandidateComparison(
            group_a=
                a.group_id,

            group_b=
                b.group_id,

            relation=
                CanonicalBranchRelation
                .DISTINCT_BRANCH,

            n_common_points=0,
            n_same=0,
            n_distinct=1,
            n_ambiguous=0,

            delta_min_energy_meV=
                float(delta_min_meV),

            min_point_gap_meV=None,
            max_point_gap_meV=None,
            ordering="NO_COMMON_POINTS",

            compared_r_values=[],
        )


    map_a = {
        round(
            p.r_A,
            10,
        ): p

        for p in a.fine.points

        if (
            p.converged
            and p.density_orth
            is not None
        )
    }


    map_b = {
        round(
            p.r_A,
            10,
        ): p

        for p in b.fine.points

        if (
            p.converged
            and p.density_orth
            is not None
        )
    }


    common = sorted(
        set(map_a)
        & set(map_b)
    )


    min_a = (
        a.fine.discrete_min_r_A
    )

    min_b = (
        b.fine.discrete_min_r_A
    )


    selected = [
        r
        for r in common
        if (
            abs(
                r - min_a
            )
            <= identity_half_window_A
            + 1.0e-10

            or

            abs(
                r - min_b
            )
            <= identity_half_window_A
            + 1.0e-10
        )
    ]


    relations = []

    point_gaps_meV = []

    for r in selected:

        pa = map_a[r]
        pb = map_b[r]

        comparison = compare_states(
            energy_a_hartree=
                pa.energy_hartree,

            energy_b_hartree=
                pb.energy_hartree,

            s2_a=
                pa.s2,

            s2_b=
                pb.s2,

            dm_orth_a=
                pa.density_orth,

            dm_orth_b=
                pb.density_orth,

            spin_a=
                spin_a,

            spin_b=
                spin_b,
        )

        relations.append(
            comparison.relation
        )

        point_gaps_meV.append(
            (
                pb.energy_hartree
                - pa.energy_hartree
            )
            * HARTREE_TO_EV
            * 1000.0
        )


    n_same = sum(
        x == StateRelation.SAME_STATE
        for x in relations
    )

    n_distinct = sum(
        x == StateRelation.DISTINCT_STATE
        for x in relations
    )

    n_ambiguous = sum(
        x == StateRelation.AMBIGUOUS
        for x in relations
    )


    relation = fine_relation_from_counts(
        n_same=n_same,
        n_distinct=n_distinct,
        n_ambiguous=n_ambiguous,
        min_same_points=
            min_same_points,
    )


    ordering = fine_energy_ordering(
        point_gaps_meV
    )


    return FineCandidateComparison(
        group_a=
            a.group_id,

        group_b=
            b.group_id,

        relation=
            relation,

        n_common_points=
            len(selected),

        n_same=
            n_same,

        n_distinct=
            n_distinct,

        n_ambiguous=
            n_ambiguous,

        delta_min_energy_meV=
            float(delta_min_meV),

        min_point_gap_meV=(
            None
            if not point_gaps_meV
            else float(
                min(
                    point_gaps_meV
                )
            )
        ),

        max_point_gap_meV=(
            None
            if not point_gaps_meV
            else float(
                max(
                    point_gaps_meV
                )
            )
        ),

        ordering=
            ordering,

        compared_r_values=[
            float(r)
            for r in selected
        ],
    )


def resolve_fine_ground_state(
    outcomes,
    *,
    identity_half_window_A: float = 0.05,
    min_same_points: int = 5,
):
    valid = [
        x
        for x in outcomes
        if (
            fine_candidate_is_resolvable(
                x
            )
            and x.fine is not None
            and x.fitted_energy_hartree
            is not None
        )
    ]


    if not valid:

        return FineGroundStateResolution(
            status=
                "NO_VALID_FINE_GS_CANDIDATE",

            primary=None,

            same_candidates=[],
            distinct_candidates=[],
            ambiguous_candidates=[],

            ambiguous_dominated_candidates=[],
            ambiguous_unresolved_candidates=[],

            comparisons=[],
        )


    ordered = sorted(
        valid,
        key=lambda x:
            x.fitted_energy_hartree,
    )


    primary = ordered[0]

    same = []
    distinct = []
    ambiguous = []

    ambiguous_dominated = []
    ambiguous_unresolved = []

    comparisons = []


    for other in ordered[1:]:

        comparison = (
            compare_fine_candidates(
                primary,
                other,

                identity_half_window_A=
                    identity_half_window_A,

                min_same_points=
                    min_same_points,
            )
        )

        comparisons.append(
            comparison
        )


        if (
            comparison.relation
            == CanonicalBranchRelation
            .SAME_BRANCH
        ):
            same.append(
                other
            )

        elif (
            comparison.relation
            == CanonicalBranchRelation
            .DISTINCT_BRANCH
        ):
            distinct.append(
                other
            )

        else:

            ambiguous.append(
                other
            )

            # Electronic identity may remain ambiguous while
            # energetic ground-state ordering is nevertheless
            # unambiguous.
            #
            # primary = candidate A in the comparison.
            if (
                comparison.ordering
                == "A_LOWER_ALL"
            ):

                ambiguous_dominated.append(
                    other
                )

            else:

                ambiguous_unresolved.append(
                    other
                )


    if ambiguous_unresolved:

        status = (
            "FINE_GS_AMBIGUOUS"
        )

    elif ambiguous_dominated:

        status = (
            "FINE_GS_RESOLVED_WITH_"
            "AMBIGUOUS_ALTERNATIVES"
        )

    else:

        status = (
            "FINE_GS_RESOLVED"
        )


    return FineGroundStateResolution(
        status=status,

        primary=primary,

        same_candidates=
            same,

        distinct_candidates=
            distinct,

        ambiguous_candidates=
            ambiguous,

        ambiguous_dominated_candidates=
            ambiguous_dominated,

        ambiguous_unresolved_candidates=
            ambiguous_unresolved,

        comparisons=
            comparisons,
    )
