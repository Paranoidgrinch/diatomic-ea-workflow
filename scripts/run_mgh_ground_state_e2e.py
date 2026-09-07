#!/usr/bin/env python3

import csv
import gc
from pathlib import Path

from diatomic_ea_v09.ground_state import (
    run_charge_ground_state_discovery,
)
from diatomic_ea_v09.model import SCFSettings
from diatomic_ea_v09.scf import HARTREE_TO_EV


OUTDIR = Path(
    "pilot_runs/ground_state_mgh_e2e"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Development grid.
#
# Not frozen production policy yet.
# Seeds are deliberately members of the coarse grid.
# ----------------------------------------------------------------------

COARSE_R = [
    round(1.0 + 0.1 * i, 10)
    for i in range(21)
]

SEED_R = [
    1.2,
    1.8,
    2.4,
]


for seed in SEED_R:
    if seed not in COARSE_R:
        raise RuntimeError(
            f"Scout seed {seed} A is not "
            "contained in coarse R grid."
        )


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def is_boundary_minimum(
    branch,
):
    minimum = branch.minimum

    if minimum is None:
        return None

    good_r = [
        p.r_A
        for p in branch.canonical_points
        if p.converged
    ]

    if not good_r:
        return None

    return (
        abs(
            minimum.r_A - min(good_r)
        ) < 1e-8
        or abs(
            minimum.r_A - max(good_r)
        ) < 1e-8
    )


def summarize_charge(
    result,
    label,
):

    print()
    print("=" * 120)
    print(label)
    print("=" * 120)

    print(
        "Discovered raw branches:",
        len(result.discovered_branches),
    )

    print(
        "Canonical branches:",
        len(result.canonical_branches),
    )

    print()
    print("DISCOVERY RECORDS")
    print("-" * 120)

    for record in result.discovery_records:

        print(
            "seed R={:.2f}  "
            "2S={}  "
            "{:>7s}  "
            "E={:.12f}  "
            "guesses={:<24s}  "
            "action={}{}".format(
                record.seed_r_A,
                record.spin,
                record.scout_group_id,
                record.energy_hartree,
                ",".join(
                    record.guesses
                ),
                record.action,
                (
                    ""
                    if record.matched_branch_id
                    is None
                    else " -> "
                    + record.matched_branch_id
                ),
            )
        )


    print()
    print("CANONICAL BRANCHES")
    print("-" * 120)

    rows = []

    for branch in result.canonical_branches:

        raw_min = (
            branch.source_branch.raw_minimum
        )

        canon_min = (
            branch.minimum
        )

        stability = (
            branch.stability_result
        )

        boundary = (
            is_boundary_minimum(
                branch
            )
            if branch.valid
            else None
        )

        print()
        print(
            branch.branch_id
        )

        print(
            "  spin 2S={} mult={}".format(
                branch.source_branch.spin,
                branch.source_branch.multiplicity,
            )
        )

        print(
            "  source seed R={:.2f} A  "
            "guesses={}".format(
                branch.source_branch.source_seed_r_A,
                ",".join(
                    branch.source_branch.source_guesses
                ),
            )
        )

        print(
            "  raw minimum: "
            "R={:.3f} A  "
            "E={:.12f} Eh".format(
                raw_min.r_A,
                raw_min.energy_hartree,
            )
        )

        print(
            "  status:",
            branch.status,
        )

        print(
            "  reconstruction relation:",
            (
                "-"
                if branch.reconstruction_relation
                is None
                else
                branch.reconstruction_relation.value
            ),
        )

        if stability is not None:

            print(
                "  stability: "
                "initial={} final={} "
                "reopts={} "
                "DeltaE={:+.6f} meV".format(
                    stability.initially_stable,
                    stability.finally_stable,
                    stability.reoptimizations,
                    stability.total_delta_energy_meV,
                )
            )

        print(
            "  stability relation:",
            (
                "-"
                if branch.stability_relation
                is None
                else
                branch.stability_relation.value
            ),
        )

        if canon_min is not None:

            print(
                "  canonical minimum: "
                "R={:.3f} A  "
                "E={:.12f} Eh".format(
                    canon_min.r_A,
                    canon_min.energy_hartree,
                )
            )

            print(
                "  minimum at boundary:",
                boundary,
            )

        rows.append({
            "branch_id":
                branch.branch_id,

            "spin":
                branch.source_branch.spin,

            "multiplicity":
                branch.source_branch.multiplicity,

            "source_seed_r_A":
                branch.source_branch.source_seed_r_A,

            "source_guesses":
                ",".join(
                    branch.source_branch.source_guesses
                ),

            "raw_min_r_A":
                raw_min.r_A,

            "raw_min_energy_hartree":
                raw_min.energy_hartree,

            "status":
                branch.status,

            "reconstruction_relation":
                (
                    ""
                    if branch.reconstruction_relation
                    is None
                    else
                    branch.reconstruction_relation.value
                ),

            "initially_stable":
                (
                    ""
                    if stability is None
                    else
                    stability.initially_stable
                ),

            "finally_stable":
                (
                    ""
                    if stability is None
                    else
                    stability.finally_stable
                ),

            "stability_reoptimizations":
                (
                    ""
                    if stability is None
                    else
                    stability.reoptimizations
                ),

            "stability_delta_energy_meV":
                (
                    ""
                    if stability is None
                    else
                    stability.total_delta_energy_meV
                ),

            "stability_relation":
                (
                    ""
                    if branch.stability_relation
                    is None
                    else
                    branch.stability_relation.value
                ),

            "canonical_min_r_A":
                (
                    ""
                    if canon_min is None
                    else
                    canon_min.r_A
                ),

            "canonical_min_energy_hartree":
                (
                    ""
                    if canon_min is None
                    else
                    canon_min.energy_hartree
                ),

            "canonical_min_boundary":
                (
                    ""
                    if boundary is None
                    else
                    boundary
                ),
        })


    csv_path = (
        OUTDIR
        / (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_branches.csv"
        )
    )

    if rows:

        with csv_path.open(
            "w",
            newline="",
        ) as fh:

            writer = csv.DictWriter(
                fh,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)


    gs = result.ground_state_branch

    print()
    print("SELECTED GROUND STATE")
    print("-" * 120)

    if gs is None:

        print(
            "NO VALID GROUND STATE FOUND"
        )

        return None

    minimum = gs.minimum

    print(
        "Branch:",
        gs.branch_id,
    )

    print(
        "Spin 2S:",
        gs.source_branch.spin,
    )

    print(
        "Multiplicity:",
        gs.source_branch.multiplicity,
    )

    print(
        "Coarse Re [A]:",
        minimum.r_A,
    )

    print(
        "Emin [Eh]:",
        "{:.12f}".format(
            minimum.energy_hartree
        ),
    )

    print(
        "Boundary minimum:",
        is_boundary_minimum(gs),
    )

    return {
        "branch_id":
            gs.branch_id,

        "spin":
            gs.source_branch.spin,

        "multiplicity":
            gs.source_branch.multiplicity,

        "r_A":
            minimum.r_A,

        "energy_hartree":
            minimum.energy_hartree,

        "boundary":
            is_boundary_minimum(gs),
    }


print("=" * 120)
print("v0.9 MgH GROUND-STATE END-TO-END PILOT")
print("=" * 120)

print(
    "Method: PBE / def2-QZVPD"
)

print(
    "Coarse R:",
    f"{min(COARSE_R):.1f} - "
    f"{max(COARSE_R):.1f} A, "
    "step 0.1 A",
)

print(
    "Scout seeds:",
    SEED_R,
)


# ======================================================================
# Neutral
# ======================================================================

print()
print("#" * 120)
print("RUNNING MgH NEUTRAL")
print("#" * 120)

neutral = run_charge_ground_state_discovery(
    atom="Mg",
    ligand="H",
    charge=0,

    seed_r_values=
        SEED_R,

    coarse_r_values=
        COARSE_R,

    spin_max=3,

    basis=
        "def2-qzvpd",

    settings=
        settings,

    max_memory_mb=2000,
)

neutral_summary = summarize_charge(
    neutral,
    "MgH neutral",
)


# Release the detailed neutral object before running anion.
del neutral
gc.collect()


# ======================================================================
# Anion
# ======================================================================

print()
print("#" * 120)
print("RUNNING MgH ANION")
print("#" * 120)

anion = run_charge_ground_state_discovery(
    atom="Mg",
    ligand="H",
    charge=-1,

    seed_r_values=
        SEED_R,

    coarse_r_values=
        COARSE_R,

    spin_max=4,

    basis=
        "def2-qzvpd",

    settings=
        settings,

    max_memory_mb=2000,
)

anion_summary = summarize_charge(
    anion,
    "MgH anion",
)


# ======================================================================
# Coarse electronic EA
# ======================================================================

print()
print("=" * 120)
print("MgH COARSE ELECTRONIC EA")
print("=" * 120)

if (
    neutral_summary is None
    or anion_summary is None
):

    print(
        "EA unavailable because at least "
        "one charge has no valid ground state."
    )

else:

    ea_hartree = (
        neutral_summary[
            "energy_hartree"
        ]
        - anion_summary[
            "energy_hartree"
        ]
    )

    ea_eV = (
        ea_hartree
        * HARTREE_TO_EV
    )

    print(
        "Neutral GS:",
        neutral_summary,
    )

    print(
        "Anion GS:",
        anion_summary,
    )

    print()

    print(
        "EA_el coarse [Eh]:",
        "{:.12f}".format(
            ea_hartree
        ),
    )

    print(
        "EA_el coarse [eV]:",
        "{:.8f}".format(
            ea_eV
        ),
    )

    print()
    print(
        "IMPORTANT: this is a coarse-grid "
        "electronic diagnostic only."
    )

    print(
        "No fine PEC and no ZPE correction "
        "have been applied."
    )


print()
print("=" * 120)
print("MgH END-TO-END GROUND-STATE PILOT COMPLETE")
print("=" * 120)

