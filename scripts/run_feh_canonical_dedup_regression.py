#!/usr/bin/env python3

from diatomic_ea_v09.canonical_dedup import (
    compare_canonical_branches,
    deduplicate_canonical_branches,
)
from diatomic_ea_v09.ground_state import (
    run_charge_ground_state_discovery,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


R_GRID = [
    round(
        1.30 + 0.10 * i,
        10,
    )
    for i in range(9)
]

SEEDS = [
    1.40,
    1.70,
    2.00,
]


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


print("=" * 128)
print("FeH POST-STABILITY CANONICAL-BRANCH DEDUP REGRESSION")
print("=" * 128)

print(
    "R grid:",
    R_GRID,
)

print(
    "Seeds:",
    SEEDS,
)

print(
    "Spins included: 2S <= 3"
)

print()


result = (
    run_charge_ground_state_discovery(
        atom="Fe",
        ligand="H",

        charge=0,

        seed_r_values=
            SEEDS,

        coarse_r_values=
            R_GRID,

        spin_max=3,

        basis=
            "def2-qzvpd",

        settings=
            settings,

        max_memory_mb=2000,
    )
)


print("=" * 128)
print("RAW DISCOVERY")
print("=" * 128)

print(
    "Discovered branches:",
    len(
        result.discovered_branches
    ),
)

for branch in (
    result.discovered_branches
):

    minimum = (
        branch.raw_minimum
    )

    print(
        "{}  2S={}  "
        "seed={:.2f}  "
        "raw_min_R={:.2f}  "
        "raw_min_E={:.12f}".format(
            branch.branch_id,

            branch.spin,

            branch.source_seed_r_A,

            minimum.r_A,

            minimum.energy_hartree,
        )
    )


print()
print("=" * 128)
print("STABILITY-CANONICALIZED BRANCHES")
print("=" * 128)


valid = []

for branch in (
    result.canonical_branches
):

    stability = (
        branch.stability_result
    )

    minimum = (
        branch.minimum
    )

    print()

    print(
        branch.branch_id
    )

    print(
        "  2S:",
        branch.source_branch.spin,
    )

    print(
        "  status:",
        branch.status,
    )

    if stability is not None:

        print(
            "  initially stable:",
            stability.initially_stable,
        )

        print(
            "  finally stable:",
            stability.finally_stable,
        )

        print(
            "  reoptimizations:",
            stability.reoptimizations,
        )

        print(
            "  stability dE [meV]: "
            "{:+.6f}".format(
                stability.total_delta_energy_meV
            )
        )

    if minimum is not None:

        print(
            "  canonical minimum: "
            "R={:.3f} A  "
            "E={:.12f} Eh".format(
                minimum.r_A,
                minimum.energy_hartree,
            )
        )

    if (
        branch.valid
        and minimum is not None
    ):
        valid.append(
            branch
        )


print()
print("=" * 128)
print("PAIRWISE CANONICAL COMPARISONS")
print("=" * 128)


for i, a in enumerate(valid):

    for b in valid[
        i + 1:
    ]:

        comparison = (
            compare_canonical_branches(
                a,
                b,

                minimum_window_A=0.20,
                min_same_points=3,
            )
        )

        print(
            "{} vs {}: "
            "{}  "
            "common={} "
            "same={} "
            "distinct={} "
            "ambiguous={} "
            "R={}".format(
                a.branch_id,
                b.branch_id,

                comparison.relation.value,

                comparison.n_common_points,
                comparison.n_same,
                comparison.n_distinct,
                comparison.n_ambiguous,

                comparison.compared_r_values,
            )
        )


groups = (
    deduplicate_canonical_branches(
        valid,

        minimum_window_A=0.20,
        min_same_points=3,
    )
)


print()
print("=" * 128)
print("DEDUPLICATED CANONICAL GROUPS")
print("=" * 128)

print(
    "Valid canonical branches before dedup:",
    len(valid),
)

print(
    "Canonical groups after dedup:",
    len(groups),
)


for group in groups:

    print()

    print(
        group.group_id
    )

    print(
        "  representative:",
        group.representative.branch_id,
    )

    print(
        "  spin 2S:",
        group.representative
        .source_branch.spin,
    )

    print(
        "  members:",
        ",".join(
            group.member_ids
        ),
    )

    print(
        "  minimum: "
        "R={:.3f} A  "
        "E={:.12f} Eh".format(
            group.representative
            .minimum.r_A,

            group.representative
            .minimum.energy_hartree,
        )
    )


eligible_groups = [
    group
    for group in groups
    if (
        group.representative.valid
        and group.representative.minimum
        is not None
    )
]


if eligible_groups:

    gs_group = min(
        eligible_groups,

        key=lambda group:
            group.representative
            .minimum
            .energy_hartree,
    )

    print()
    print("=" * 128)
    print("LOWEST DEDUPLICATED CANONICAL GROUP")
    print("=" * 128)

    print(
        "Group:",
        gs_group.group_id,
    )

    print(
        "Representative:",
        gs_group.representative.branch_id,
    )

    print(
        "Spin 2S:",
        gs_group.representative
        .source_branch.spin,
    )

    print(
        "Members:",
        gs_group.member_ids,
    )

    print(
        "Minimum R [A]:",
        gs_group.representative
        .minimum.r_A,
    )

    print(
        "Minimum E [Eh]: "
        "{:.12f}".format(
            gs_group.representative
            .minimum.energy_hartree
        )
    )


print()
print("=" * 128)
print("FeH CANONICAL DEDUP REGRESSION COMPLETE")
print("=" * 128)
