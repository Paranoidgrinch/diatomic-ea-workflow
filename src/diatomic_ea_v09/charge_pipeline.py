from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence

from .adaptive_grid import (
    ExpansionPolicy,
)
from .branch_finalize import (
    FinalizedBranch,
    finalize_ground_branch,
)
from .canonical_dedup import (
    CanonicalBranchGroup,
    deduplicate_canonical_branches,
)
from .fine_candidates import (
    FineCandidateOutcome,
    FineGroundStateResolution,
    fine_candidate_effective_status,
    fine_candidate_is_resolvable,
    refine_candidate_group,
    resolve_fine_ground_state,
)
from .fine_pec import (
    FineGridPolicy,
)
from .ground_state import (
    CanonicalBranch,
    ChargeGroundStateResult,
    run_charge_ground_state_discovery,
)
from .ground_state_candidates import (
    GroundStateCandidateSet,
    select_ground_state_candidate_groups,
)
from .model import (
    SCFSettings,
)


@dataclass
class ChargeGroundStatePipelineResult:
    molecule: str
    charge: int

    status: str

    discovery: ChargeGroundStateResult

    finalized_branches: List[
        FinalizedBranch
    ]

    finalized_canonical_views: List[
        CanonicalBranch
    ]

    canonical_groups: List[
        CanonicalBranchGroup
    ]

    coarse_candidates: GroundStateCandidateSet | None

    fine_outcomes: List[
        FineCandidateOutcome
    ]

    fine_resolution: FineGroundStateResolution | None


def canonical_view_from_finalized(
    finalized: FinalizedBranch,
) -> CanonicalBranch | None:
    """
    Convert a finalized coarse branch back into the CanonicalBranch
    interface used by deduplication and fine refinement.

    Only PASS branches are eligible.
    """

    if (
        not finalized.valid
        or finalized.status != "PASS"
        or not finalized.points
    ):
        return None

    return replace(
        finalized.source,

        canonical_points=
            list(
                finalized.points
            ),

        valid=True,
        status="PASS",
    )


def pipeline_status_from_fine(
    *,
    outcomes: Sequence[
        FineCandidateOutcome
    ],

    resolution:
        FineGroundStateResolution | None,
):
    """
    A retained coarse candidate may never disappear silently.

    A Fine candidate is resolution-eligible when its local quadratic
    well passes strict pointwise QC.

    Review findings elsewhere in the wider Fine PEC remain visible
    through a dedicated charge-level FULL_BRANCH_REVIEW status.
    """

    if not outcomes:
        return (
            "QC_FAIL_NO_FINE_CANDIDATES"
        )


    effective = [
        fine_candidate_effective_status(
            outcome
        )
        for outcome in outcomes
    ]


    if any(
        not fine_candidate_is_resolvable(
            outcome
        )
        for outcome in outcomes
    ):
        return (
            "QC_FAIL_FINE_CANDIDATE"
        )


    full_branch_review = any(
        status
        == (
            "PASS_LOCAL_WELL_"
            "FULL_BRANCH_REVIEW"
        )
        for status in effective
    )


    if resolution is None:
        return (
            "QC_FAIL_FINE_RESOLUTION"
        )


    if (
        resolution.status
        == "FINE_GS_AMBIGUOUS"
    ):
        return (
            "FINE_GS_AMBIGUOUS"
        )


    if (
        resolution.status
        == "FINE_GS_RESOLVED_WITH_"
        "AMBIGUOUS_ALTERNATIVES"
    ):
        if full_branch_review:
            return (
                "PASS_AMBIGUOUS_"
                "ALTERNATIVE_DOMINATED_"
                "FULL_BRANCH_REVIEW"
            )

        return (
            "PASS_AMBIGUOUS_"
            "ALTERNATIVE_DOMINATED"
        )


    if (
        resolution.status
        == "FINE_GS_RESOLVED"
    ):
        if full_branch_review:
            return (
                "PASS_FULL_BRANCH_REVIEW"
            )

        return "PASS"


    return (
        "QC_FAIL_FINE_RESOLUTION"
    )

def run_charge_ground_state_pipeline(
    *,
    atom: str,
    ligand: str,

    charge: int,

    seed_r_values:
        Sequence[float],

    coarse_r_values:
        Sequence[float],

    spin_max: int,

    basis: str,

    settings: SCFSettings,

    expansion_policy:
        ExpansionPolicy =
        ExpansionPolicy(),

    fine_policy:
        FineGridPolicy =
        FineGridPolicy(),

    perform_pointwise_qc: bool = True,

    max_memory_mb: int = 2000,

    precomputed_groups_by_seed=None,
):
    """
    End-to-end ground-state workflow for ONE charge state.

    Pipeline
    --------
    1. multiseed / multispin / multiguess discovery
    2. coarse branch following
    3. minimum-level stability canonicalization
    4. adaptive R expansion
    5. closed stability/expansion finalization
    6. canonical branch deduplication
    7. conservative coarse GS candidate selection
    8. independent fine PEC for every retained candidate
    9. optional pointwise fine stability/QC
    10. fine-level electronic + energetic GS resolution

    Excited states are deliberately outside this pipeline.
    """

    molecule = (
        atom + ligand
    ).upper()


    # ==========================================================
    # Discovery + initial canonicalization
    # ==========================================================

    discovery = (
        run_charge_ground_state_discovery(
            atom=atom,
            ligand=ligand,

            charge=charge,

            seed_r_values=
                seed_r_values,

            coarse_r_values=
                coarse_r_values,

            spin_max=
                spin_max,

            basis=basis,

            settings=settings,

            max_memory_mb=
                max_memory_mb,

            precomputed_groups_by_seed=
                precomputed_groups_by_seed,
        )
    )


    # ==========================================================
    # Closed expansion / stability finalization
    # ==========================================================

    finalized = []

    for branch in (
        discovery.canonical_branches
    ):

        result = finalize_ground_branch(
            branch=branch,

            atom=atom,
            ligand=ligand,

            basis=basis,

            settings=settings,

            max_memory_mb=
                max_memory_mb,

            expansion_policy=
                expansion_policy,

            max_cycles=6,

            max_stability_iterations=6,
        )

        finalized.append(
            result
        )


    canonical_views = []

    for result in finalized:

        view = (
            canonical_view_from_finalized(
                result
            )
        )

        if view is not None:
            canonical_views.append(
                view
            )


    if not canonical_views:

        return ChargeGroundStatePipelineResult(
            molecule=molecule,
            charge=charge,

            status=
                "QC_FAIL_NO_VALID_"
                "COARSE_BRANCH",

            discovery=
                discovery,

            finalized_branches=
                finalized,

            finalized_canonical_views=[],

            canonical_groups=[],

            coarse_candidates=None,

            fine_outcomes=[],

            fine_resolution=None,
        )


    # ==========================================================
    # Post-stability canonical deduplication
    # ==========================================================

    groups = (
        deduplicate_canonical_branches(
            canonical_views,

            minimum_window_A=0.20,

            min_same_points=3,
        )
    )


    coarse_candidates = (
        select_ground_state_candidate_groups(
            groups,

            minimum_window_A=0.20,

            min_same_points=3,
        )
    )


    if coarse_candidates.primary is None:

        return ChargeGroundStatePipelineResult(
            molecule=molecule,
            charge=charge,

            status=
                "QC_FAIL_NO_COARSE_"
                "GS_CANDIDATE",

            discovery=
                discovery,

            finalized_branches=
                finalized,

            finalized_canonical_views=
                canonical_views,

            canonical_groups=
                groups,

            coarse_candidates=
                coarse_candidates,

            fine_outcomes=[],

            fine_resolution=None,
        )


    # ==========================================================
    # Fine refinement of ALL retained GS candidates
    # ==========================================================

    fine_outcomes = []

    for group in (
        coarse_candidates
        .fine_candidates
    ):

        outcome = refine_candidate_group(
            group=group,

            atom=atom,
            ligand=ligand,

            basis=basis,

            settings=settings,

            fine_policy=
                fine_policy,

            perform_pointwise_qc=
                perform_pointwise_qc,

            max_memory_mb=
                max_memory_mb,
        )

        fine_outcomes.append(
            outcome
        )


    # Do not silently resolve after losing a retained candidate.
    if any(
        not fine_candidate_is_resolvable(
            outcome
        )
        for outcome in fine_outcomes
    ):

        return ChargeGroundStatePipelineResult(
            molecule=molecule,
            charge=charge,

            status=
                "QC_FAIL_FINE_CANDIDATE",

            discovery=
                discovery,

            finalized_branches=
                finalized,

            finalized_canonical_views=
                canonical_views,

            canonical_groups=
                groups,

            coarse_candidates=
                coarse_candidates,

            fine_outcomes=
                fine_outcomes,

            fine_resolution=None,
        )


    # ==========================================================
    # Fine electronic/energetic resolution
    # ==========================================================

    resolution = (
        resolve_fine_ground_state(
            fine_outcomes,

            identity_half_window_A=0.05,

            min_same_points=5,
        )
    )


    final_status = (
        pipeline_status_from_fine(
            outcomes=
                fine_outcomes,

            resolution=
                resolution,
        )
    )


    return ChargeGroundStatePipelineResult(
        molecule=molecule,
        charge=charge,

        status=
            final_status,

        discovery=
            discovery,

        finalized_branches=
            finalized,

        finalized_canonical_views=
            canonical_views,

        canonical_groups=
            groups,

        coarse_candidates=
            coarse_candidates,

        fine_outcomes=
            fine_outcomes,

        fine_resolution=
            resolution,
    )
