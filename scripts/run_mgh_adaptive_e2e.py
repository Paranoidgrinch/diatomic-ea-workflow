#!/usr/bin/env python3

from pathlib import Path

from diatomic_ea_v09.adaptive_charge_pipeline import (
    run_adaptive_charge_ground_state_pipeline,
)
from diatomic_ea_v09.adaptive_grid import (
    ExpansionPolicy,
)
from diatomic_ea_v09.discovery_policy import (
    DiscoveryPolicy,
)
from diatomic_ea_v09.fine_pec import (
    FineGridPolicy,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)
from diatomic_ea_v09.scf import (
    HARTREE_TO_EV,
)


OUTDIR = Path(
    "pilot_runs/adaptive_e2e_mgh"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


COARSE_R = [
    round(
        1.0 + 0.1 * i,
        10,
    )
    for i in range(21)
]


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


discovery_policy = DiscoveryPolicy(
    seed_spacing_A=0.20,
    seed_margin_A=0.20,

    initial_spin_sectors=5,
    max_spin_sectors=8,

    spin_tail_length=3,
)


expansion_policy = ExpansionPolicy(
    step_A=0.10,
    block_width_A=0.50,

    lower_limit_A=0.60,
    upper_limit_A=6.00,

    guard_points=2,
    max_rounds=20,
)


fine_policy = FineGridPolicy(
    half_width_A=0.20,
    step_A=0.01,
    quadratic_half_points=3,
)


CASES = [
    (
        "NEUTRAL",
        0,
    ),
    (
        "ANION",
        -1,
    ),
]


results = {}


def summarize(
    label,
    result,
):
    print()
    print("=" * 132)
    print(
        f"MgH {label}"
    )
    print("=" * 132)

    print(
        "Overall status:",
        result.status,
    )

    scout = result.scout

    print()
    print("ADAPTIVE SCOUT")
    print("-" * 132)

    print(
        "Scout status:",
        scout.status,
    )

    print(
        "Scout seeds:",
        scout.seed_r_values,
    )

    print(
        "Physical spin sectors:",
        scout.physical_spin_values,
    )

    print(
        "Development spin sectors:",
        scout.development_spin_values,
    )

    print(
        "Actually scanned spins:",
        scout.scanned_spin_values,
    )

    print(
        "Spin frontier closed:",
        scout.frontier.closed,
    )

    print(
        "Physical spin space exhausted:",
        scout.frontier
        .physical_spin_space_exhausted,
    )

    print(
        "Failing frontier seeds:",
        scout.frontier
        .failing_seed_r_values,
    )

    print(
        "Missing frontier data:",
        scout.frontier
        .missing_seed_r_values,
    )

    print(
        "Fixed-(R,spin) scout sectors calculated:",
        len(
            scout.sector_records
        ),
    )

    print(
        "Total retained local CandidateGroups:",
        sum(
            len(groups)
            for groups
            in scout.groups_by_seed.values()
        ),
    )


    if result.ground_state is None:

        print()
        print(
            "NO GROUND-STATE PIPELINE RESULT"
        )

        return


    gs = result.ground_state

    print()
    print("COARSE DISCOVERY / FINALIZATION")
    print("-" * 132)

    print(
        "Discovered branches:",
        len(
            gs.discovery
            .discovered_branches
        ),
    )

    print(
        "Seed discovery records:",
        len(
            gs.discovery
            .discovery_records
        ),
    )

    print(
        "Initial canonical branches:",
        len(
            gs.discovery
            .canonical_branches
        ),
    )

    print(
        "Finalized branches:",
        len(
            gs.finalized_branches
        ),
    )

    print(
        "Finalized PASS branches:",
        sum(
            branch.status == "PASS"
            for branch
            in gs.finalized_branches
        ),
    )

    print(
        "Canonical groups after dedup:",
        len(
            gs.canonical_groups
        ),
    )


    if gs.coarse_candidates is not None:

        print()
        print("COARSE GS CANDIDATES")
        print("-" * 132)

        print(
            "Status:",
            gs.coarse_candidates.status,
        )

        print(
            "Fine candidate groups:",
            gs.coarse_candidates
            .fine_candidate_ids,
        )


    print()
    print("FINE CANDIDATES")
    print("-" * 132)


    for outcome in (
        gs.fine_outcomes
    ):

        print()

        print(
            "Group:",
            outcome.group_id,
        )

        print(
            "Branch:",
            outcome.branch_id,
        )

        print(
            "Status:",
            outcome.status,
        )

        if outcome.fine is not None:

            print(
                "Discrete Re [A]:",
                outcome.fine
                .discrete_min_r_A,
            )

            print(
                "Fitted Re [A]:",
                "{:.8f}".format(
                    outcome.fine
                    .fitted_min_r_A
                ),
            )

            print(
                "Fitted Emin [Eh]:",
                "{:.12f}".format(
                    outcome.fitted_energy_hartree
                ),
            )

            print(
                "Curvature [Eh/A^2]:",
                "{:.9f}".format(
                    outcome.fine
                    .curvature_hartree_per_A2
                ),
            )

        if outcome.qc is not None:

            print(
                "Pointwise QC:",
                outcome.qc.status,
            )

            print(
                "QC PASS points:",
                "{}/{}".format(
                    outcome.qc.n_pass,
                    outcome.qc.n_points,
                ),
            )

            print(
                "Initially unstable fine points:",
                outcome.qc
                .n_initially_unstable,
            )

            print(
                "Fine stability root changes:",
                outcome.qc
                .n_stability_state_change,
            )


    resolution = (
        gs.fine_resolution
    )

    print()
    print("FINAL FINE GS RESOLUTION")
    print("-" * 132)


    if resolution is None:

        print(
            "Resolution: NONE"
        )

        return


    print(
        "Resolution status:",
        resolution.status,
    )


    if resolution.primary is None:

        print(
            "Primary: NONE"
        )

        return


    primary = (
        resolution.primary
    )

    branch = (
        primary.group
        .representative
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
        "Ground-state 2S:",
        branch.source_branch.spin,
    )

    print(
        "Multiplicity:",
        branch.source_branch
        .multiplicity,
    )

    print(
        "Fitted Re [A]: "
        "{:.8f}".format(
            primary.fine
            .fitted_min_r_A
        )
    )

    print(
        "Fitted ground-state energy [Eh]: "
        "{:.12f}".format(
            primary
            .fitted_energy_hartree
        )
    )

    print(
        "Same alternatives:",
        [
            x.group_id
            for x
            in resolution.same_candidates
        ],
    )

    print(
        "Distinct alternatives:",
        [
            x.group_id
            for x
            in resolution.distinct_candidates
        ],
    )

    print(
        "Ambiguous dominated alternatives:",
        [
            x.group_id
            for x
            in resolution
            .ambiguous_dominated_candidates
        ],
    )

    print(
        "Ambiguous unresolved alternatives:",
        [
            x.group_id
            for x
            in resolution
            .ambiguous_unresolved_candidates
        ],
    )


for label, charge in CASES:

    print()
    print("#" * 132)
    print(
        f"STARTING MgH {label}"
    )
    print("#" * 132)


    result = (
        run_adaptive_charge_ground_state_pipeline(
            atom="Mg",
            ligand="H",

            charge=charge,

            coarse_r_values=
                COARSE_R,

            basis=
                "def2-qzvpd",

            settings=
                settings,

            discovery_policy=
                discovery_policy,

            expansion_policy=
                expansion_policy,

            fine_policy=
                fine_policy,

            perform_pointwise_qc=True,

            max_memory_mb=2000,
        )
    )


    results[
        label
    ] = result

    summarize(
        label,
        result,
    )


# =====================================================================
# Final EA
# =====================================================================

print()
print("=" * 132)
print("MgH ADAPTIVE END-TO-END ELECTRONIC EA")
print("=" * 132)


neutral = (
    results["NEUTRAL"]
)

anion = (
    results["ANION"]
)


accepted_statuses = {
    "PASS",
    "PASS_AMBIGUOUS_ALTERNATIVE_DOMINATED",
}


for label, result in [
    (
        "NEUTRAL",
        neutral,
    ),
    (
        "ANION",
        anion,
    ),
]:

    if (
        result.status
        not in accepted_statuses
    ):

        raise RuntimeError(
            "{} did not reach an accepted "
            "ground-state status: {}".format(
                label,
                result.status,
            )
        )

    if (
        result.ground_state is None
        or
        result.ground_state
        .fine_resolution is None
        or
        result.ground_state
        .fine_resolution
        .primary is None
    ):

        raise RuntimeError(
            label
            + ": missing primary fine GS"
        )


neutral_primary = (
    neutral
    .ground_state
    .fine_resolution
    .primary
)

anion_primary = (
    anion
    .ground_state
    .fine_resolution
    .primary
)


neutral_spin = (
    neutral_primary
    .group
    .representative
    .source_branch
    .spin
)

anion_spin = (
    anion_primary
    .group
    .representative
    .source_branch
    .spin
)


neutral_energy = (
    neutral_primary
    .fitted_energy_hartree
)

anion_energy = (
    anion_primary
    .fitted_energy_hartree
)


ea_el = (
    neutral_energy
    - anion_energy
) * HARTREE_TO_EV


print(
    "Neutral final status:",
    neutral.status,
)

print(
    "Anion final status:",
    anion.status,
)

print(
    "Neutral GS 2S:",
    neutral_spin,
)

print(
    "Anion GS 2S:",
    anion_spin,
)

print(
    "Neutral fitted Re [A]: "
    "{:.8f}".format(
        neutral_primary
        .fine
        .fitted_min_r_A
    )
)

print(
    "Anion fitted Re [A]: "
    "{:.8f}".format(
        anion_primary
        .fine
        .fitted_min_r_A
    )
)

print(
    "Neutral fitted Emin [Eh]: "
    "{:.12f}".format(
        neutral_energy
    )
)

print(
    "Anion fitted Emin [Eh]: "
    "{:.12f}".format(
        anion_energy
    )
)

print(
    "EA_el [eV]: "
    "{:.8f}".format(
        ea_el
    )
)


print()
print("=" * 132)
print("REGRESSION EXPECTATIONS")
print("=" * 132)

print(
    "Expected neutral GS 2S = 1"
)

print(
    "Expected anion GS 2S = 0"
)

print(
    "Previous neutral fitted Re ~= 1.75343 A"
)

print(
    "Previous anion fitted Re ~= 1.90503 A"
)

print(
    "Previous EA_el ~= 0.725974 eV"
)


print()
print("=" * 132)
print("MgH ADAPTIVE END-TO-END TEST COMPLETE")
print("=" * 132)
