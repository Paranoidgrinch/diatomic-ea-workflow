#!/usr/bin/env python3

import gc

from diatomic_ea_v09.adaptive_grid import (
    ExpansionPolicy,
)
from diatomic_ea_v09.branch_finalize import (
    finalize_ground_branch,
)
from diatomic_ea_v09.ground_state import (
    run_charge_ground_state_discovery,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)
from diatomic_ea_v09.scf import (
    HARTREE_TO_EV,
)


COARSE_R = [
    round(
        1.0 + 0.1 * i,
        10,
    )
    for i in range(21)
]

SEED_R = [
    1.2,
    1.8,
    2.4,
]


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


policy = ExpansionPolicy(
    step_A=0.10,
    block_width_A=0.50,
    lower_limit_A=0.60,
    upper_limit_A=6.00,
    guard_points=2,
    max_rounds=20,
)


def run_charge(
    *,
    charge,
    spin_max,
    label,
):

    print()
    print("=" * 124)
    print(label)
    print("=" * 124)

    discovery = (
        run_charge_ground_state_discovery(
            atom="Mg",
            ligand="H",

            charge=charge,

            seed_r_values=
                SEED_R,

            coarse_r_values=
                COARSE_R,

            spin_max=
                spin_max,

            basis=
                "def2-qzvpd",

            settings=
                settings,

            max_memory_mb=2000,
        )
    )


    finalized = []

    for branch in (
        discovery.canonical_branches
    ):

        print()
        print("#" * 124)
        print(branch.branch_id)
        print("#" * 124)

        result = finalize_ground_branch(
            branch=branch,

            atom="Mg",
            ligand="H",

            basis=
                "def2-qzvpd",

            settings=
                settings,

            max_memory_mb=2000,

            expansion_policy=
                policy,

            max_cycles=6,

            max_stability_iterations=6,
        )

        finalized.append(
            result
        )

        print(
            "Final status:",
            result.status,
        )

        print(
            "Finalization cycles:",
            result.cycles,
        )

        for event in result.history:

            print(
                "  cycle {}: "
                "expanded={} "
                "exp_rounds={} "
                "pre_R={:.3f} "
                "stable_initial={} "
                "stable_final={} "
                "reopts={} "
                "stab_dE={:+.6f} meV "
                "post_R={} "
                "shift={}".format(
                    event.cycle,
                    event.expanded,
                    event.expansion_rounds,
                    event.pre_stability_min_r_A,
                    event.initially_stable,
                    event.finally_stable,
                    event.stability_reoptimizations,
                    (
                        0.0
                        if event.stability_delta_energy_meV
                        is None
                        else
                        event.stability_delta_energy_meV
                    ),
                    (
                        "-"
                        if event.post_stability_min_r_A
                        is None
                        else
                        "{:.3f}".format(
                            event.post_stability_min_r_A
                        )
                    ),
                    (
                        "-"
                        if event.minimum_shift_A
                        is None
                        else
                        "{:+.3f}".format(
                            event.minimum_shift_A
                        )
                    ),
                )
            )

        minimum = (
            result.minimum
        )

        if minimum is not None:

            print(
                "Final minimum: "
                "R={:.3f} A "
                "E={:.12f} Eh".format(
                    minimum.r_A,
                    minimum.energy_hartree,
                )
            )


    eligible = [
        branch
        for branch
        in finalized
        if (
            branch.valid
            and branch.status == "PASS"
            and branch.minimum
            is not None
        )
    ]

    if not eligible:

        print(
            "NO VALID FINALIZED BRANCH"
        )

        return None


    gs = min(
        eligible,
        key=lambda x:
            x.minimum.energy_hartree,
    )

    minimum = gs.minimum

    print()
    print("=" * 124)
    print("SELECTED FINALIZED GROUND STATE")
    print("=" * 124)

    print(
        "Branch:",
        gs.source.branch_id,
    )

    print(
        "Spin 2S:",
        gs.source.source_branch.spin,
    )

    print(
        "Multiplicity:",
        gs.source.source_branch.multiplicity,
    )

    print(
        "Coarse Re [A]:",
        minimum.r_A,
    )

    print(
        "Energy [Eh]: "
        "{:.12f}".format(
            minimum.energy_hartree
        )
    )

    return {
        "branch_id":
            gs.source.branch_id,

        "spin":
            gs.source.source_branch.spin,

        "multiplicity":
            gs.source.source_branch.multiplicity,

        "r_A":
            minimum.r_A,

        "energy_hartree":
            minimum.energy_hartree,
    }


print("=" * 124)
print("v0.9 MgH CLOSED COARSE-GROUNDSTATE FINALIZATION")
print("=" * 124)


neutral = run_charge(
    charge=0,
    spin_max=3,
    label="MgH NEUTRAL",
)

gc.collect()


anion = run_charge(
    charge=-1,
    spin_max=4,
    label="MgH ANION",
)


print()
print("=" * 124)
print("FINALIZED COARSE EA")
print("=" * 124)

if (
    neutral is None
    or anion is None
):

    print(
        "EA unavailable."
    )

else:

    ea_eV = (
        neutral["energy_hartree"]
        - anion["energy_hartree"]
    ) * HARTREE_TO_EV

    print(
        "Neutral:",
        neutral,
    )

    print(
        "Anion:",
        anion,
    )

    print(
        "EA_el coarse [eV]: "
        "{:.8f}".format(
            ea_eV
        )
    )


print()
print("=" * 124)
print("CLOSED FINALIZATION TEST COMPLETE")
print("=" * 124)
