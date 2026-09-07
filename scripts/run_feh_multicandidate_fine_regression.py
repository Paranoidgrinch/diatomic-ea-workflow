#!/usr/bin/env python3

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
)
from diatomic_ea_v09.canonical_dedup import (
    deduplicate_canonical_branches,
)
from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.fine_candidates import (
    refine_candidate_group,
    resolve_fine_ground_state,
)
from diatomic_ea_v09.fine_pec import (
    FineGridPolicy,
)
from diatomic_ea_v09.ground_state import (
    DiscoveredBranch,
    canonicalize_branch_at_minimum,
)
from diatomic_ea_v09.ground_state_candidates import (
    select_ground_state_candidate_groups,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


COARSE_R = [
    1.40,
    1.50,
    1.60,
    1.70,
    1.80,
]


fine_policy = FineGridPolicy(
    half_width_A=0.20,
    step_A=0.01,
    quadratic_half_points=3,
)


print("=" * 128)
print("FeH QUARTET MULTI-CANDIDATE FINE-PEC REGRESSION")
print("=" * 128)

print(
    "Method: PBE / def2-QZVPD"
)

print(
    "Discovery seed: R=1.40 A"
)

print(
    "Spin: 2S=3 only"
)

print(
    "Coarse diagnostic grid:",
    COARSE_R,
)

print()


# ======================================================================
# 1. Independent quartet scout at R = 1.40 A
# ======================================================================

solutions = []

for guess in DEFAULT_GUESSES:

    spec = MoleculeSpec(
        atom="Fe",
        ligand="H",
        charge=0,
        spin=3,
        basis="def2-qzvpd",
        r_A=1.40,
        max_memory_mb=2000,
    )

    try:

        solution = run_guess(
            spec,
            settings,
            guess,
        )

    except Exception as exc:

        print(
            "Guess unavailable:",
            guess,
            repr(exc),
        )

        continue

    if solution is None:
        continue

    solution.mf = None

    solutions.append(
        solution
    )

    print(
        "{:>8s}: "
        "E={:.12f} Eh "
        "<S2>={:.7f}".format(
            guess,
            solution.energy_hartree,
            solution.s2,
        )
    )


groups = build_candidate_groups(
    solutions
)


print()
print(
    "Local quartet groups:",
    len(groups),
)


# ======================================================================
# 2. Propagate each local group only across the small diagnostic grid
# ======================================================================

discovered = []

for index, group in enumerate(
    groups,
    start=1,
):

    representative = (
        group.representative
    )

    branch_id = (
        f"FEH_Q_FINE_DISC{index:02d}"
    )

    points = follow_scout_solution(
        branch_id=
            branch_id,

        atom="Fe",
        ligand="H",

        seed=
            representative,

        settings=
            settings,

        r_values=
            COARSE_R,

        max_memory_mb=2000,
    )

    branch = DiscoveredBranch(
        branch_id=
            branch_id,

        charge=0,

        spin=
            representative.spin,

        multiplicity=
            representative.multiplicity,

        source_seed_r_A=1.40,

        source_scout_group_id=
            group.scout_group_id,

        source_guesses=
            group.equivalent_guesses,

        seed_solution=
            representative,

        raw_points=
            points,
    )

    discovered.append(
        branch
    )

    minimum = (
        branch.raw_minimum
    )

    print()
    print(
        branch_id,
        "guesses=",
        ",".join(
            group.equivalent_guesses
        ),
    )

    print(
        "  raw min: "
        "R={:.3f} "
        "E={:.12f}".format(
            minimum.r_A,
            minimum.energy_hartree,
        )
    )


# ======================================================================
# 3. Stability canonicalization
# ======================================================================

canonical = []

print()
print("=" * 128)
print("STABILITY CANONICALIZATION")
print("=" * 128)


for branch in discovered:

    result = (
        canonicalize_branch_at_minimum(
            branch=
                branch,

            atom="Fe",
            ligand="H",

            basis=
                "def2-qzvpd",

            settings=
                settings,

            coarse_r_values=
                COARSE_R,

            max_memory_mb=2000,

            max_stability_iterations=6,
        )
    )

    print()
    print(
        result.branch_id
    )

    print(
        "  status:",
        result.status,
    )

    if result.stability_result is not None:

        s = result.stability_result

        print(
            "  stability: "
            "initial={} "
            "final={} "
            "reopts={} "
            "dE={:+.6f} meV".format(
                s.initially_stable,
                s.finally_stable,
                s.reoptimizations,
                s.total_delta_energy_meV,
            )
        )

    if result.minimum is not None:

        print(
            "  canonical min: "
            "R={:.3f} "
            "E={:.12f}".format(
                result.minimum.r_A,
                result.minimum.energy_hartree,
            )
        )

    if result.valid:
        canonical.append(
            result
        )


# ======================================================================
# 4. Dedup and select only coarse GS candidates
# ======================================================================

canon_groups = (
    deduplicate_canonical_branches(
        canonical,

        minimum_window_A=0.20,
        min_same_points=3,
    )
)


candidate_set = (
    select_ground_state_candidate_groups(
        canon_groups,

        minimum_window_A=0.20,
        min_same_points=3,
    )
)


print()
print("=" * 128)
print("COARSE CANDIDATE SET")
print("=" * 128)

print(
    "Canonical groups:",
    len(canon_groups),
)

print(
    "Candidate status:",
    candidate_set.status,
)

print(
    "Fine candidates:",
    candidate_set.fine_candidate_ids,
)


for group in (
    candidate_set.fine_candidates
):

    branch = (
        group.representative
    )

    print(
        "{}  representative={}  "
        "Rmin={:.3f}  "
        "Emin={:.12f}".format(
            group.group_id,
            branch.branch_id,
            branch.minimum.r_A,
            branch.minimum.energy_hartree,
        )
    )


# ======================================================================
# 5. Fine PEC for every retained candidate
#
# Pointwise QC deliberately deferred until candidate structure is known.
# ======================================================================

fine_outcomes = []

print()
print("=" * 128)
print("FINE PECs")
print("=" * 128)


for group in (
    candidate_set.fine_candidates
):

    outcome = refine_candidate_group(
        group=group,

        atom="Fe",
        ligand="H",

        basis=
            "def2-qzvpd",

        settings=
            settings,

        fine_policy=
            fine_policy,

        perform_pointwise_qc=False,

        max_memory_mb=2000,
    )

    fine_outcomes.append(
        outcome
    )

    print()
    print(
        group.group_id
    )

    print(
        "  status:",
        outcome.status,
    )

    print(
        "  seed stability: "
        "initial={} final={} dE={}".format(
            outcome.seed_initially_stable,
            outcome.seed_finally_stable,

            (
                "-"
                if outcome.seed_stability_delta_meV
                is None
                else
                "{:+.6f} meV".format(
                    outcome.seed_stability_delta_meV
                )
            ),
        )
    )

    if outcome.fine is not None:

        fine = outcome.fine

        print(
            "  discrete min: "
            "R={:.3f} "
            "E={:.12f}".format(
                fine.discrete_min_r_A,
                fine.discrete_min_energy_hartree,
            )
        )

        print(
            "  fitted Re: "
            "{:.8f}".format(
                fine.fitted_min_r_A
            )
        )

        print(
            "  fitted Emin: "
            "{:.12f}".format(
                fine.fitted_min_energy_hartree
            )
        )

        print(
            "  curvature: "
            "{:.9f} Eh/A^2".format(
                fine.curvature_hartree_per_A2
            )
        )


# ======================================================================
# 6. Fine-level electronic resolution
# ======================================================================

resolution = (
    resolve_fine_ground_state(
        fine_outcomes,

        identity_half_window_A=0.05,
        min_same_points=5,
    )
)


print()
print("=" * 128)
print("FINE GROUND-STATE RESOLUTION")
print("=" * 128)

print(
    "Status:",
    resolution.status,
)


if resolution.primary is not None:

    print(
        "Primary:",
        resolution.primary.group_id,
    )

    print(
        "Primary fitted Emin [Eh]: "
        "{:.12f}".format(
            resolution.primary
            .fitted_energy_hartree
        )
    )


for comparison in (
    resolution.comparisons
):

    print(
        "{} vs {}: "
        "{}  "
        "dEmin={:+.6f} meV  "
        "common={} "
        "same={} "
        "distinct={} "
        "ambiguous={} "
        "R={}".format(
            comparison.group_a,
            comparison.group_b,

            comparison.relation.value,

            comparison.delta_min_energy_meV,

            comparison.n_common_points,
            comparison.n_same,
            comparison.n_distinct,
            comparison.n_ambiguous,

            comparison.compared_r_values,
        )
    )


print()
print(
    "Same candidates:",
    [
        x.group_id
        for x in resolution.same_candidates
    ],
)

print(
    "Distinct candidates:",
    [
        x.group_id
        for x in resolution.distinct_candidates
    ],
)

print(
    "Ambiguous candidates:",
    [
        x.group_id
        for x in resolution.ambiguous_candidates
    ],
)


print()
print("=" * 128)
print("FeH MULTI-CANDIDATE FINE REGRESSION COMPLETE")
print("=" * 128)
