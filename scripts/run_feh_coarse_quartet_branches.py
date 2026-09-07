#!/usr/bin/env python3

import csv
import gc
from pathlib import Path

import numpy as np

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
)
from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.scf import (
    HARTREE_TO_EV,
)
from diatomic_ea_v09.state_identity import (
    compare_states,
)
from diatomic_ea_v09.state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


OUTDIR = Path(
    "pilot_runs/coarse_branch_feh"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


# Development grid only.
# This is NOT yet the universal production coarse grid.
R_GRID = [
    1.30,
    1.35,
    1.40,
    1.45,
    1.50,
    1.54,
    1.55,
    1.60,
    1.65,
    1.70,
    1.75,
    1.80,
    1.90,
    2.00,
    2.10,
]


print("=" * 118)
print("FeH COARSE QUARTET-BRANCH REGRESSION")
print("=" * 118)

print(
    "Seed: R=1.54 A, neutral, 2S=3"
)

print(
    "R grid:",
    R_GRID,
)


# ============================================================
# Recreate local quartet scout at the seed geometry
# ============================================================

solutions = []

for guess in DEFAULT_GUESSES:

    spec = MoleculeSpec(
        atom="Fe",
        ligand="H",
        charge=0,
        spin=3,
        basis="def2-qzvpd",
        r_A=1.54,
        max_memory_mb=2000,
    )

    sol = run_guess(
        spec,
        settings,
        guess,
    )

    if sol is None:
        print(
            "{}: seed SCF failed".format(
                guess
            )
        )
        continue

    print(
        "seed {:>8s}: "
        "E={:.12f} Eh "
        "<S2>={:.7f}".format(
            guess,
            sol.energy_hartree,
            sol.s2,
        )
    )

    solutions.append(sol)


groups = build_candidate_groups(
    solutions
)

print()
print(
    "Quartet scout groups:",
    len(groups),
)

for group in groups:

    print(
        "{}: E={:.12f} guesses={}".format(
            group.scout_group_id,
            group.representative.energy_hartree,
            ",".join(
                group.equivalent_guesses
            ),
        )
    )


if len(groups) != 2:
    raise RuntimeError(
        "Regression expectation failed: "
        "FeH quartet should form exactly "
        "two local scout groups."
    )


# ============================================================
# Follow both branches
# ============================================================

all_branches = {}

for group in groups:

    branch_id = (
        "FEH_N_"
        + group.scout_group_id
    )

    seed = group.representative

    print()
    print("#" * 118)
    print("FOLLOW", branch_id)
    print("#" * 118)

    points = follow_scout_solution(
        branch_id=branch_id,
        atom="Fe",
        ligand="H",
        seed=seed,
        settings=settings,
        r_values=R_GRID,
        max_memory_mb=2000,
    )

    all_branches[
        branch_id
    ] = points

    good = [
        p
        for p in points
        if p.converged
    ]

    for point in points:

        if point.converged:

            print(
                "R={:.3f}  "
                "E={:.12f}  "
                "<S2>={:.7f}  "
                "parent={}  "
                "source={}".format(
                    point.r_A,
                    point.energy_hartree,
                    point.s2,
                    (
                        "-"
                        if point.parent_r_A
                        is None
                        else "{:.3f}".format(
                            point.parent_r_A
                        )
                    ),
                    point.point_source,
                )
            )

        else:

            print(
                "R={:.3f}  FAILED  "
                "parent={}  {}".format(
                    point.r_A,
                    point.parent_r_A,
                    point.scf_path,
                )
            )

    if good:

        minimum = min(
            good,
            key=lambda p:
                p.energy_hartree,
        )

        print()
        print(
            "Coarse minimum:",
            "R={:.3f} A".format(
                minimum.r_A
            ),
            "E={:.12f} Eh".format(
                minimum.energy_hartree
            ),
        )

        r_good = [
            p.r_A
            for p in good
        ]

        boundary = (
            minimum.r_A
            == min(r_good)
            or minimum.r_A
            == max(r_good)
        )

        print(
            "Minimum at current "
            "converged boundary:",
            boundary,
        )


    # Save CSV
    csv_path = (
        OUTDIR
        / (
            branch_id
            + ".csv"
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
            "spin",
            "multiplicity",
            "converged",
            "energy_hartree",
            "relative_energy_meV",
            "s2",
            "observed_multiplicity",
            "homo_eV",
            "lumo_eV",
            "gap_eV",
            "scf_path",
            "point_source",
        ]

        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )

        writer.writeheader()

        if good:
            emin = min(
                p.energy_hartree
                for p in good
            )
        else:
            emin = None

        for p in points:

            row = {
                "branch_id":
                    p.branch_id,
                "r_A":
                    p.r_A,
                "parent_r_A":
                    p.parent_r_A,
                "spin":
                    p.spin,
                "multiplicity":
                    p.multiplicity,
                "converged":
                    p.converged,
                "energy_hartree":
                    p.energy_hartree,
                "relative_energy_meV":
                    (
                        None
                        if (
                            not p.converged
                            or emin is None
                        )
                        else (
                            p.energy_hartree
                            - emin
                        )
                        * HARTREE_TO_EV
                        * 1000.0
                    ),
                "s2":
                    p.s2,
                "observed_multiplicity":
                    p.observed_multiplicity,
                "homo_eV":
                    p.homo_eV,
                "lumo_eV":
                    p.lumo_eV,
                "gap_eV":
                    p.gap_eV,
                "scf_path":
                    p.scf_path,
                "point_source":
                    p.point_source,
            }

            writer.writerow(row)


    # Save densities for reuse/audit.
    good_for_density = [
        p
        for p in points
        if (
            p.converged
            and p.density_ao
            is not None
        )
    ]

    if good_for_density:

        np.savez_compressed(
            OUTDIR
            / (
                branch_id
                + "_densities.npz"
            ),
            r_A=np.array([
                p.r_A
                for p
                in good_for_density
            ]),
            density_ao=np.stack([
                p.density_ao
                for p
                in good_for_density
            ]),
            density_orth=np.stack([
                p.density_orth
                for p
                in good_for_density
            ]),
        )


# ============================================================
# Compare the two branches at every common R
# ============================================================

print()
print("=" * 118)
print("PAIRWISE BRANCH IDENTITY AT COMMON R")
print("=" * 118)

branch_ids = list(
    all_branches.keys()
)

a_id, b_id = branch_ids

a_map = {
    round(p.r_A, 10): p
    for p in all_branches[a_id]
    if p.converged
}

b_map = {
    round(p.r_A, 10): p
    for p in all_branches[b_id]
    if p.converged
}

common = sorted(
    set(a_map)
    & set(b_map)
)

for r in common:

    a = a_map[r]
    b = b_map[r]

    comparison = compare_states(
        energy_a_hartree=
            a.energy_hartree,
        energy_b_hartree=
            b.energy_hartree,
        s2_a=a.s2,
        s2_b=b.s2,
        dm_orth_a=
            a.density_orth,
        dm_orth_b=
            b.density_orth,
        spin_a=a.spin,
        spin_b=b.spin,
    )

    print(
        "R={:.3f}  "
        "dE={:.6f} meV  "
        "dS2={:.6f}  "
        "dTot={:.6e}  "
        "dSpin={:.6e}  "
        "{}".format(
            r,
            comparison.delta_energy_meV,
            comparison.delta_s2,
            comparison.total_spectrum_max,
            comparison.spin_spectrum_max,
            comparison.relation.value,
        )
    )


print()
print("=" * 118)
print("FEH COARSE BRANCH REGRESSION COMPLETE")
print("=" * 118)

