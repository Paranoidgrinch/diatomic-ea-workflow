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
    "pilot_runs/adaptive_e2e_feh_neutral"
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


print("=" * 132)
print("FeH NEUTRAL ADAPTIVE END-TO-END GROUNDSTATE WORKFLOW")
print("=" * 132)

print(
    "Method: PBE / def2-QZVPD"
)

print(
    "Initial coarse grid: 1.0-3.0 A, step 0.1 A"
)

print(
    "Pointwise fine stability QC: ENABLED"
)

print()


result = (
    run_adaptive_charge_ground_state_pipeline(
        atom="Fe",
        ligand="H",

        charge=0,

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

        max_memory_mb=3000,
    )
)


# =====================================================================
# Adaptive scout
# =====================================================================

print()
print("=" * 132)
print("ADAPTIVE SCOUT")
print("=" * 132)

scout = result.scout

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
    "Frontier tail:",
    scout.frontier.tail_spins,
)

print(
    "Non-rising seeds:",
    scout.frontier
    .failing_seed_r_values,
)

print(
    "Missing-tail-data seeds:",
    scout.frontier
    .missing_seed_r_values,
)

print(
    "Fixed-(R,spin) sectors calculated:",
    len(
        scout.sector_records
    ),
)

print(
    "Total retained CandidateGroups:",
    sum(
        len(groups)
        for groups
        in scout.groups_by_seed.values()
    ),
)


failed_guess_records = [
    record
    for record
    in scout.sector_records
    if record.failed_guesses
]

print(
    "Sectors with unavailable/failed guesses:",
    len(
        failed_guess_records
    ),
)


if failed_guess_records:

    print()
    print(
        "FAILED-GUESS AUDIT"
    )
    print("-" * 132)

    for record in failed_guess_records:

        print(
            "R={:.2f} 2S={} failed={}".format(
                record.seed_r_A,
                record.spin,
                len(
                    record.failed_guesses
                ),
            )
        )

        for failure in (
            record.failed_guesses
        ):
            print(
                "   ",
                failure,
            )


# =====================================================================
# Ground-state pipeline
# =====================================================================

if result.ground_state is None:

    print()
    print("=" * 132)
    print("NO GROUND-STATE PIPELINE RESULT")
    print("=" * 132)

    print(
        "Overall status:",
        result.status,
    )

    raise SystemExit(0)


gs = result.ground_state


print()
print("=" * 132)
print("COARSE DISCOVERY")
print("=" * 132)

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


print()
print("DISCOVERED BRANCHES")
print("-" * 132)

for branch in (
    gs.discovery
    .discovered_branches
):

    minimum = (
        branch.raw_minimum
    )

    print(
        "{}  2S={}  seed={:.2f}  "
        "raw_Rmin={:.3f}  "
        "raw_Emin={:.12f}".format(
            branch.branch_id,
            branch.spin,
            branch.source_seed_r_A,
            minimum.r_A,
            minimum.energy_hartree,
        )
    )


# =====================================================================
# Closed coarse finalization
# =====================================================================

print()
print("=" * 132)
print("CLOSED COARSE FINALIZATION")
print("=" * 132)


for branch in (
    gs.finalized_branches
):

    print()
    print(
        branch.source.branch_id
    )

    print(
        "  status:",
        branch.status,
    )

    print(
        "  valid:",
        branch.valid,
    )

    print(
        "  cycles:",
        len(
            branch.history
        ),
    )

    if branch.minimum is not None:

        print(
            "  final minimum: "
            "R={:.3f} A "
            "E={:.12f} Eh".format(
                branch.minimum.r_A,
                branch.minimum.energy_hartree,
            )
        )

    for cycle in (
        branch.history
    ):

        print(
            "    cycle {}: "
            "expanded={} "
            "rounds={} "
            "pre_R={:.3f} "
            "stable_initial={} "
            "stable_final={} "
            "reopts={} "
            "stab_dE={:+.6f} meV "
            "post_R={:.3f} "
            "shift={:+.3f}".format(
                cycle.cycle,

                cycle.expanded,
                cycle.expansion_rounds,

                cycle.pre_stability_min_r_A,

                cycle.stability_initially_stable,
                cycle.stability_finally_stable,
                cycle.stability_reoptimizations,
                cycle.stability_delta_energy_meV,

                cycle.post_stability_min_r_A,
                cycle.minimum_shift_A,
            )
        )


# =====================================================================
# Canonical groups
# =====================================================================

print()
print("=" * 132)
print("POST-STABILITY CANONICAL GROUPS")
print("=" * 132)

print(
    "Number of groups:",
    len(
        gs.canonical_groups
    ),
)


for group in (
    gs.canonical_groups
):

    branch = (
        group.representative
    )

    print()

    print(
        group.group_id
    )

    print(
        "  representative:",
        branch.branch_id,
    )

    print(
        "  2S:",
        branch.source_branch.spin,
    )

    print(
        "  members:",
        group.member_ids,
    )

    print(
        "  minimum: "
        "R={:.3f} "
        "E={:.12f}".format(
            branch.minimum.r_A,
            branch.minimum.energy_hartree,
        )
    )


# =====================================================================
# Coarse candidate selection
# =====================================================================

print()
print("=" * 132)
print("COARSE GROUND-STATE CANDIDATES")
print("=" * 132)


if gs.coarse_candidates is None:

    print(
        "No coarse candidate set."
    )

else:

    candidates = (
        gs.coarse_candidates
    )

    print(
        "Status:",
        candidates.status,
    )

    print(
        "Primary:",
        (
            None
            if candidates.primary
            is None
            else candidates.primary
            .group_id
        ),
    )

    print(
        "Fine candidates:",
        candidates.fine_candidate_ids,
    )

    print(
        "Ambiguous competitors:",
        [
            group.group_id
            for group
            in candidates
            .ambiguous_competitors
        ],
    )

    print(
        "Distinct higher groups:",
        [
            group.group_id
            for group
            in candidates
            .distinct_higher_groups
        ],
    )


    print()
    print("PRIMARY COMPARISONS")
    print("-" * 132)

    for comparison in (
        candidates.comparisons
    ):

        delta_meV = (
            comparison
            .delta_energy_hartree
            * HARTREE_TO_EV
            * 1000.0
        )

        print(
            "{} vs {}: {}  "
            "dEmin={:+.6f} meV".format(
                comparison
                .primary_group_id,

                comparison
                .other_group_id,

                comparison
                .relation.value,

                delta_meV,
            )
        )


# =====================================================================
# Fine candidates + pointwise QC
# =====================================================================

print()
print("=" * 132)
print("FINE CANDIDATES")
print("=" * 132)


for outcome in (
    gs.fine_outcomes
):

    print()

    print(
        outcome.group_id
    )

    print(
        "  branch:",
        outcome.branch_id,
    )

    print(
        "  status:",
        outcome.status,
    )

    print(
        "  seed stability: "
        "initial={} final={} "
        "dE={}".format(
            outcome
            .seed_initially_stable,

            outcome
            .seed_finally_stable,

            (
                "-"
                if outcome
                .seed_stability_delta_meV
                is None
                else
                "{:+.6f} meV".format(
                    outcome
                    .seed_stability_delta_meV
                )
            ),
        )
    )


    if outcome.fine is not None:

        fine = outcome.fine

        print(
            "  discrete minimum: "
            "R={:.3f} A "
            "E={:.12f} Eh".format(
                fine.discrete_min_r_A,
                fine.discrete_min_energy_hartree,
            )
        )

        print(
            "  fitted Re [A]: "
            "{:.8f}".format(
                fine.fitted_min_r_A
            )
        )

        print(
            "  fitted Emin [Eh]: "
            "{:.12f}".format(
                fine.fitted_min_energy_hartree
            )
        )

        print(
            "  curvature [Eh/A^2]: "
            "{:.9f}".format(
                fine.curvature_hartree_per_A2
            )
        )

        print(
            "  boundary minimum:",
            fine.minimum_at_boundary,
        )


    if outcome.qc is not None:

        qc = (
            outcome.qc
        )

        print(
            "  pointwise QC:",
            qc.status,
        )

        print(
            "  QC PASS points: {}/{}".format(
                qc.n_pass,
                qc.n_points,
            )
        )

        print(
            "  reconstruction failures:",
            qc.n_reconstruction_fail,
        )

        print(
            "  identity failures:",
            qc.n_identity_fail,
        )

        print(
            "  initially unstable:",
            qc.n_initially_unstable,
        )

        print(
            "  finally unstable:",
            qc.n_finally_unstable,
        )

        print(
            "  stability SAME_STATE reopts:",
            qc.n_stability_same_state,
        )

        print(
            "  stability AMBIGUOUS reopts:",
            qc.n_stability_ambiguous,
        )

        print(
            "  stability root changes:",
            qc.n_stability_state_change,
        )

        print(
            "  max |reconstruction dE| [meV]:",
            qc.max_abs_reconstruction_delta_meV,
        )

        print(
            "  most negative stability dE [meV]:",
            qc.most_negative_stability_delta_meV,
        )


# =====================================================================
# Final fine resolution
# =====================================================================

print()
print("=" * 132)
print("FINAL FINE GROUND-STATE RESOLUTION")
print("=" * 132)

resolution = (
    gs.fine_resolution
)


if resolution is None:

    print(
        "Fine resolution: NONE"
    )

else:

    print(
        "Resolution status:",
        resolution.status,
    )


    if resolution.primary is not None:

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
            in resolution
            .same_candidates
        ],
    )

    print(
        "Distinct alternatives:",
        [
            x.group_id
            for x
            in resolution
            .distinct_candidates
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


    print()
    print("FINE PRIMARY COMPARISONS")
    print("-" * 132)

    for comparison in (
        resolution.comparisons
    ):

        print(
            "{} vs {}: "
            "{}  "
            "dEmin={:+.6f} meV  "
            "point-gap=[{}, {}] meV  "
            "ordering={}  "
            "same={} distinct={} ambiguous={}".format(
                comparison.group_a,
                comparison.group_b,

                comparison.relation.value,

                comparison
                .delta_min_energy_meV,

                (
                    "-"
                    if comparison
                    .min_point_gap_meV
                    is None
                    else
                    "{:+.6f}".format(
                        comparison
                        .min_point_gap_meV
                    )
                ),

                (
                    "-"
                    if comparison
                    .max_point_gap_meV
                    is None
                    else
                    "{:+.6f}".format(
                        comparison
                        .max_point_gap_meV
                    )
                ),

                comparison.ordering,

                comparison.n_same,
                comparison.n_distinct,
                comparison.n_ambiguous,
            )
        )


# =====================================================================
# Final status
# =====================================================================

print()
print("=" * 132)
print("FeH NEUTRAL FINAL RESULT")
print("=" * 132)

print(
    "Overall status:",
    result.status,
)


accepted = {
    "PASS",
    "PASS_AMBIGUOUS_ALTERNATIVE_DOMINATED",
}


print(
    "Accepted ground-state result:",
    result.status in accepted,
)


if (
    resolution is not None
    and resolution.primary is not None
):

    print(
        "Final neutral GS 2S:",
        resolution.primary
        .group
        .representative
        .source_branch
        .spin,
    )

    print(
        "Final neutral fitted Re [A]: "
        "{:.8f}".format(
            resolution.primary
            .fine
            .fitted_min_r_A
        )
    )

    print(
        "Final neutral Emin [Eh]: "
        "{:.12f}".format(
            resolution.primary
            .fitted_energy_hartree
        )
    )


print()
print("=" * 132)
print("REGRESSION EXPECTATION")
print("=" * 132)

print(
    "Expected lowest FeH neutral manifold: 2S=3"
)

print(
    "Previous fine primary Re ~= 1.56741 A"
)

print(
    "Previous FeH result contained a ~0.73 meV "
    "electronically ambiguous but energetically dominated "
    "quartet alternative."
)


print()
print("=" * 132)
print("FeH NEUTRAL ADAPTIVE END-TO-END TEST COMPLETE")
print("=" * 132)
