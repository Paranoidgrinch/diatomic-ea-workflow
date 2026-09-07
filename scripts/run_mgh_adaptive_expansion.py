#!/usr/bin/env python3

import csv
import gc
from pathlib import Path

from diatomic_ea_v09.adaptive_grid import (
    ExpansionPolicy,
    adaptive_expand_branch,
    converged_points,
)
from diatomic_ea_v09.ground_state import (
    run_charge_ground_state_discovery,
)
from diatomic_ea_v09.model import SCFSettings
from diatomic_ea_v09.scf import HARTREE_TO_EV


OUTDIR = Path(
    "pilot_runs/adaptive_expansion_mgh"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


COARSE_R = [
    round(1.0 + 0.1 * i, 10)
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

    result = (
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

    expanded = []

    for branch in result.canonical_branches:

        print()
        print("#" * 124)
        print(branch.branch_id)
        print("#" * 124)

        if not branch.valid:

            print(
                "Canonicalization status:",
                branch.status,
            )

            expanded.append({
                "branch":
                    branch,

                "expansion":
                    None,
            })

            continue


        before = branch.minimum

        print(
            "Before expansion:"
        )

        print(
            "  minimum R={:.3f} A".format(
                before.r_A
            )
        )

        print(
            "  minimum E={:.12f} Eh".format(
                before.energy_hartree
            )
        )

        print(
            "  initial R coverage: "
            "{:.3f} - {:.3f} A".format(
                min(
                    p.r_A
                    for p
                    in branch.canonical_points
                    if p.converged
                ),
                max(
                    p.r_A
                    for p
                    in branch.canonical_points
                    if p.converged
                ),
            )
        )


        expansion = adaptive_expand_branch(
            points=
                branch.canonical_points,

            atom="Mg",
            ligand="H",

            charge=
                charge,

            spin=
                branch.source_branch.spin,

            basis=
                "def2-qzvpd",

            settings=
                settings,

            max_memory_mb=2000,

            policy=
                policy,
        )


        good = converged_points(
            expansion.points
        )

        after_r = (
            expansion.minimum_r_A
        )

        after_e = (
            expansion.minimum_energy_hartree
        )

        print()
        print(
            "Expansion status:",
            expansion.status,
        )

        print(
            "Expansion rounds:",
            expansion.rounds,
        )

        print(
            "Lower blocks added:",
            expansion.lower_extensions,
        )

        print(
            "Upper blocks added:",
            expansion.upper_extensions,
        )

        print(
            "Expanded:",
            expansion.expanded,
        )

        if good:

            print(
                "Final R coverage: "
                "{:.3f} - {:.3f} A".format(
                    good[0].r_A,
                    good[-1].r_A,
                )
            )

        if after_r is not None:

            print(
                "After expansion minimum: "
                "R={:.3f} A  "
                "E={:.12f} Eh".format(
                    after_r,
                    after_e,
                )
            )

            delta_min_meV = (
                after_e
                - before.energy_hartree
            ) * HARTREE_TO_EV * 1000.0

            print(
                "Minimum-energy change "
                "[meV]: {:+.6f}".format(
                    delta_min_meV
                )
            )

            print(
                "Minimum moved [A]: "
                "{:+.3f}".format(
                    after_r - before.r_A
                )
            )


        # Print several final points so an asymptotic / still-descending
        # curve is immediately visible.
        print()
        print(
            "Last converged points:"
        )

        for point in good[-8:]:

            rel_meV = (
                point.energy_hartree
                - min(
                    p.energy_hartree
                    for p in good
                )
            ) * HARTREE_TO_EV * 1000.0

            print(
                "  R={:.3f}  "
                "E={:.12f}  "
                "rel={:+.6f} meV".format(
                    point.r_A,
                    point.energy_hartree,
                    rel_meV,
                )
            )


        # Save full expanded PEC.
        csv_path = (
            OUTDIR
            / (
                branch.branch_id
                + "_expanded.csv"
            )
        )

        with csv_path.open(
            "w",
            newline="",
        ) as fh:

            fields = [
                "branch_id",
                "r_A",
                "parent_r_A",
                "converged",
                "energy_hartree",
                "relative_energy_meV",
                "s2",
                "scf_path",
                "point_source",
            ]

            writer = csv.DictWriter(
                fh,
                fieldnames=fields,
            )

            writer.writeheader()

            emin = (
                min(
                    p.energy_hartree
                    for p in good
                )
                if good
                else None
            )

            for point in expansion.points:

                writer.writerow({
                    "branch_id":
                        point.branch_id,

                    "r_A":
                        point.r_A,

                    "parent_r_A":
                        point.parent_r_A,

                    "converged":
                        point.converged,

                    "energy_hartree":
                        point.energy_hartree,

                    "relative_energy_meV":
                        (
                            ""
                            if (
                                not point.converged
                                or emin is None
                            )
                            else (
                                point.energy_hartree
                                - emin
                            )
                            * HARTREE_TO_EV
                            * 1000.0
                        ),

                    "s2":
                        point.s2,

                    "scf_path":
                        point.scf_path,

                    "point_source":
                        point.point_source,
                })


        expanded.append({
            "branch":
                branch,

            "expansion":
                expansion,
        })


    # ------------------------------------------------------------
    # Diagnostic lowest valid expanded branch
    # ------------------------------------------------------------

    eligible = [
        item
        for item in expanded
        if (
            item["expansion"] is not None
            and
            item["expansion"].status
            == "PASS"
            and
            item["expansion"]
            .minimum_energy_hartree
            is not None
        )
    ]

    print()
    print("=" * 124)
    print(
        "EXPANDED CHARGE-LEVEL SUMMARY"
    )
    print("=" * 124)

    if not eligible:

        print(
            "No PASS branch after expansion."
        )

        return None


    selected = min(
        eligible,
        key=lambda item:
            item["expansion"]
            .minimum_energy_hartree,
    )

    branch = selected[
        "branch"
    ]

    expansion = selected[
        "expansion"
    ]

    print(
        "Diagnostic lowest PASS branch:",
        branch.branch_id,
    )

    print(
        "Spin 2S:",
        branch.source_branch.spin,
    )

    print(
        "Multiplicity:",
        branch.source_branch.multiplicity,
    )

    print(
        "Minimum R [A]:",
        expansion.minimum_r_A,
    )

    print(
        "Minimum E [Eh]: "
        "{:.12f}".format(
            expansion.minimum_energy_hartree
        )
    )

    print()
    print(
        "NOTE: if expansion moved the minimum, "
        "production logic must repeat stability "
        "canonicalization at that new minimum."
    )

    return {
        "branch_id":
            branch.branch_id,

        "spin":
            branch.source_branch.spin,

        "multiplicity":
            branch.source_branch.multiplicity,

        "r_A":
            expansion.minimum_r_A,

        "energy_hartree":
            expansion.minimum_energy_hartree,
    }


print("=" * 124)
print("v0.9 MgH ADAPTIVE COARSE-GRID EXPANSION REGRESSION")
print("=" * 124)

print(
    "Initial coarse grid: "
    "1.0 - 3.0 A / 0.1 A"
)

print(
    "Expansion block: 0.5 A"
)

print(
    "Development limits: "
    "0.6 - 6.0 A"
)

print(
    "Required guard points around minimum:",
    policy.guard_points,
)


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
print("EXPANDED MgH ELECTRONIC EA DIAGNOSTIC")
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

    print()

    print(
        "EA_el expanded coarse [eV]: "
        "{:.8f}".format(
            ea_eV
        )
    )


print()
print("=" * 124)
print("ADAPTIVE EXPANSION REGRESSION COMPLETE")
print("=" * 124)
