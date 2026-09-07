#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

from diatomic_ea_v09.canonical_dedup import (
    CanonicalBranchGroup,
)
from diatomic_ea_v09.fine_candidates import (
    evaluate_local_well_qc,
    fine_candidate_effective_status,
    refine_candidate_group,
    resolve_fine_ground_state,
)
from diatomic_ea_v09.charge_pipeline import (
    pipeline_status_from_fine,
)
from diatomic_ea_v09.fine_pec import (
    FineGridPolicy,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


OUTDIR = Path(
    "pilot_runs/feh_anion_checkpoint"
)

STAGE_C_PATH = (
    OUTDIR / "stage_c_ground_pec.pkl"
)

CHECKPOINT_PATH = (
    OUTDIR / "stage_d_fine.pkl"
)


TARGETS = [
    "SEED_BRANCH_043",
]


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


fine_policy = FineGridPolicy(
    half_width_A=0.20,
    step_A=0.01,
    quadratic_half_points=3,
)


def atomic_pickle(payload):
    tmp = Path(
        str(CHECKPOINT_PATH)
        + ".tmp"
    )

    with tmp.open("wb") as fh:
        pickle.dump(
            payload,
            fh,
            protocol=
                pickle.HIGHEST_PROTOCOL,
        )

        fh.flush()
        os.fsync(
            fh.fileno()
        )

    os.replace(
        tmp,
        CHECKPOINT_PATH,
    )


print("=" * 132)
print("FeH ANION — STAGE D: FINE PEC + LOCAL/FULL QC")
print("=" * 132)
print(flush=True)


if not STAGE_C_PATH.exists():

    raise RuntimeError(
        "Missing Stage-C checkpoint: "
        + str(STAGE_C_PATH)
    )


with STAGE_C_PATH.open(
    "rb"
) as fh:

    stage_c = pickle.load(
        fh
    )


canonical = stage_c[
    "canonical_branches"
]


for cid in TARGETS:

    if cid not in canonical:
        raise RuntimeError(
            "Missing canonical branch: "
            + cid
        )

    if not canonical[cid].valid:
        raise RuntimeError(
            cid
            + " is not a valid canonical branch."
        )


groups = {
    cid:
        CanonicalBranchGroup(
            group_id=cid,
            representative=
                canonical[cid],
            members=[
                canonical[cid]
            ],
        )

    for cid in TARGETS
}


# =====================================================================
# Resume
# =====================================================================

if CHECKPOINT_PATH.exists():

    with CHECKPOINT_PATH.open(
        "rb"
    ) as fh:

        state = pickle.load(
            fh
        )

    outcomes = dict(
        state.get(
            "outcomes",
            {},
        )
    )

    print(
        "RESUMING Stage D"
    )

    print(
        "Completed candidates:",
        sorted(
            outcomes
        ),
    )

else:

    outcomes = {}

    print(
        "Starting new Stage D"
    )


# =====================================================================
# Fine refinement
# =====================================================================

for cid in TARGETS:

    print()
    print("#" * 132)
    print(
        "FINE CANDIDATE:",
        cid,
    )
    print("#" * 132)
    print(flush=True)


    if cid in outcomes:

        print(
            "Already checkpointed — SKIP",
            flush=True,
        )

        continue


    group = (
        groups[
            cid
        ]
    )


    branch = (
        group.representative
    )


    print(
        "Coarse minimum: "
        "R={:.3f} A  "
        "E={:.12f} Eh".format(
            branch.minimum.r_A,
            branch.minimum
            .energy_hartree,
        ),
        flush=True,
    )

    print(
        "Fine grid: "
        "{:.2f} ... {:.2f} A, step {:.2f} A".format(
            branch.minimum.r_A
            - fine_policy.half_width_A,

            branch.minimum.r_A
            + fine_policy.half_width_A,

            fine_policy.step_A,
        ),
        flush=True,
    )


    outcome = (
        refine_candidate_group(
            group=
                group,

            atom="Fe",
            ligand="H",

            basis=
                "def2-qzvpd",

            settings=
                settings,

            fine_policy=
                fine_policy,

            perform_pointwise_qc=True,

            max_memory_mb=3000,
        )
    )


    outcomes[
        cid
    ] = outcome


    atomic_pickle({
        "outcomes":
            outcomes,
    })


    print()
    print(
        "STATUS:",
        outcome.status,
        flush=True,
    )


    print(
        "Seed initially stable:",
        outcome.seed_initially_stable,
    )

    print(
        "Seed finally stable:",
        outcome.seed_finally_stable,
    )

    print(
        "Seed stability dE [meV]:",
        outcome.seed_stability_delta_meV,
    )

    print(
        "Seed relation:",
        (
            None
            if outcome.seed_relation
            is None
            else outcome
            .seed_relation
            .value
        ),
    )


    if outcome.fine is not None:

        fine = (
            outcome.fine
        )

        print(
            "Discrete minimum: "
            "R={:.3f} A  "
            "E={:.12f} Eh".format(
                fine.discrete_min_r_A,
                fine.discrete_min_energy_hartree,
            )
        )

        print(
            "Fitted Re [A]:",
            fine.fitted_min_r_A,
        )

        print(
            "Fitted Emin [Eh]:",
            fine.fitted_min_energy_hartree,
        )

        print(
            "Curvature [Eh/A^2]:",
            fine.curvature_hartree_per_A2,
        )

        print(
            "Quadratic points used:",
            fine.quadratic_points_used,
        )

        print(
            "Boundary minimum:",
            fine.minimum_at_boundary,
        )


    if outcome.qc is not None:

        qc = (
            outcome.qc
        )

        print()
        print(
            "POINTWISE QC"
        )
        print("-" * 132)

        print(
            "QC status:",
            qc.status,
        )

        print(
            "PASS points: {}/{}".format(
                qc.n_pass,
                qc.n_points,
            )
        )

        print(
            "Reconstruction failures:",
            qc.n_reconstruction_fail,
        )

        print(
            "Identity failures:",
            qc.n_identity_fail,
        )

        print(
            "Initially unstable:",
            qc.n_initially_unstable,
        )

        print(
            "Finally unstable:",
            qc.n_finally_unstable,
        )

        print(
            "Stability SAME_STATE:",
            qc.n_stability_same_state,
        )

        print(
            "Stability AMBIGUOUS:",
            qc.n_stability_ambiguous,
        )

        print(
            "Stability state changes:",
            qc.n_stability_state_change,
        )

        print(
            "Max |reconstruction dE| [meV]:",
            qc.max_abs_reconstruction_delta_meV,
        )

        print(
            "Most negative stability dE [meV]:",
            qc.most_negative_stability_delta_meV,
        )


    if (
        outcome.fine is not None
        and outcome.qc is not None
    ):

        local_well_qc = (
            evaluate_local_well_qc(
                fine=outcome.fine,
                qc=outcome.qc,
            )
        )

        print()
        print(
            "LOCAL-WELL QC"
        )
        print("-" * 132)

        print(
            "Local-well status:",
            local_well_qc.status,
        )

        print(
            "Fit R range [A]:",
            local_well_qc.r_low_A,
            "to",
            local_well_qc.r_high_A,
        )

        print(
            "Local-well PASS: {}/{}".format(
                local_well_qc.n_pass,
                local_well_qc.n_points,
            )
        )


    print()
    print(
        "Effective Fine status:",
        fine_candidate_effective_status(
            outcome
        ),
    )


    print(
        "CHECKPOINT SAVED:",
        cid,
        flush=True,
    )


# =====================================================================
# Final Fine ground-state resolution
# =====================================================================

ordered_outcomes = [
    outcomes[cid]
    for cid in TARGETS
    if cid in outcomes
]


if len(
    ordered_outcomes
) != len(
    TARGETS
):

    print()
    print(
        "Not all Fine candidates completed."
    )

    raise SystemExit(0)


print()
print("=" * 132)
print("FINAL FINE GROUND-STATE RESOLUTION")
print("=" * 132)


resolution = (
    resolve_fine_ground_state(
        ordered_outcomes,

        identity_half_window_A=0.05,

        min_same_points=5,
    )
)


print(
    "Resolution status:",
    resolution.status,
)


if resolution.primary is not None:

    primary = (
        resolution.primary
    )

    print(
        "Primary group:",
        primary.group_id,
    )

    print(
        "Primary branch:",
        primary.branch_id,
    )

    print(
        "Fitted Re [A]: "
        "{:.8f}".format(
            primary.fine
            .fitted_min_r_A
        )
    )

    print(
        "Fitted Emin [Eh]: "
        "{:.12f}".format(
            primary
            .fitted_energy_hartree
        )
    )


print(
    "Same alternatives:",
    [
        x.group_id
        for x in
        resolution.same_candidates
    ],
)

print(
    "Distinct alternatives:",
    [
        x.group_id
        for x in
        resolution.distinct_candidates
    ],
)

print(
    "Ambiguous alternatives:",
    [
        x.group_id
        for x in
        resolution.ambiguous_candidates
    ],
)

print(
    "Ambiguous dominated:",
    [
        x.group_id
        for x in
        resolution
        .ambiguous_dominated_candidates
    ],
)

print(
    "Ambiguous unresolved:",
    [
        x.group_id
        for x in
        resolution
        .ambiguous_unresolved_candidates
    ],
)


pipeline_status = (
    pipeline_status_from_fine(
        outcomes=
            ordered_outcomes,

        resolution=
            resolution,
    )
)


print()
print(
    "Charge-level Fine status:",
    pipeline_status,
)


print()
print("PAIRWISE COMPARISONS")
print("-" * 132)


for comparison in (
    resolution.comparisons
):

    print(
        "{} vs {}".format(
            comparison.group_a,
            comparison.group_b,
        )
    )

    print(
        "  relation:",
        comparison.relation.value,
    )

    print(
        "  common points:",
        comparison.n_common_points,
    )

    print(
        "  SAME / DISTINCT / AMBIGUOUS:",
        (
            comparison.n_same,
            comparison.n_distinct,
            comparison.n_ambiguous,
        ),
    )

    print(
        "  delta fitted minima [meV]: "
        "{:+.6f}".format(
            comparison
            .delta_min_energy_meV
        )
    )

    print(
        "  point-gap range [meV]:",
        comparison.min_point_gap_meV,
        "to",
        comparison.max_point_gap_meV,
    )

    print(
        "  energetic ordering:",
        comparison.ordering,
    )


atomic_pickle({
    "outcomes":
        outcomes,

    "resolution":
        resolution,
})


print()
print("=" * 132)
print("FeH- STAGE-D TARGET")
print("=" * 132)

print(
    "Expected primary: SEED_BRANCH_043"
)

print(
    "Expected number of Fine candidates: 1"
)

print(
    "No pairwise alternative comparison expected."
)

print()
print("=" * 132)
print("FeH ANION STAGE D COMPLETE")
print("=" * 132)
