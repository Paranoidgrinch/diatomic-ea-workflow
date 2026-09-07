#!/usr/bin/env python3

import csv
import gc
from collections import Counter
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
from diatomic_ea_v09.state_identity import (
    StateRelation,
    compare_states,
)
from diatomic_ea_v09.state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


REFDIR = Path(
    "pilot_runs/coarse_branch_feh"
)

OUTDIR = Path(
    "pilot_runs/branch_resolution_feh_R1p40"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


REFERENCE_BRANCHES = [
    "FEH_N_S3_G01",
    "FEH_N_S3_G02",
]

SEED_R = 1.40

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


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def load_reference_branch(
    branch_id,
):
    csv_path = (
        REFDIR
        / f"{branch_id}.csv"
    )

    npz_path = (
        REFDIR
        / f"{branch_id}_densities.npz"
    )

    with csv_path.open() as fh:
        rows = list(
            csv.DictReader(fh)
        )

    archive = np.load(
        npz_path
    )

    density_r = np.asarray(
        archive["r_A"],
        dtype=float,
    )

    density_orth = np.asarray(
        archive["density_orth"]
    )

    output = {}

    for row in rows:

        if row["converged"] != "True":
            continue

        r = float(
            row["r_A"]
        )

        idx = np.where(
            np.isclose(
                density_r,
                r,
                atol=1.0e-8,
                rtol=0.0,
            )
        )[0]

        if len(idx) != 1:
            raise RuntimeError(
                f"{branch_id}: density lookup "
                f"failed at R={r}"
            )

        output[
            round(r, 10)
        ] = {
            "energy_hartree":
                float(
                    row[
                        "energy_hartree"
                    ]
                ),
            "s2":
                float(
                    row["s2"]
                ),
            "density_orth":
                np.array(
                    density_orth[
                        int(idx[0])
                    ],
                    copy=True,
                ),
        }

    return output


references = {
    branch_id:
        load_reference_branch(
            branch_id
        )
    for branch_id
    in REFERENCE_BRANCHES
}


print("=" * 120)
print("FeH R=1.40 A CANDIDATE-BRANCH RESOLUTION")
print("=" * 120)


# ======================================================================
# Independent local scout at R=1.40 A
# ======================================================================

raw = []

for guess in DEFAULT_GUESSES:

    spec = MoleculeSpec(
        atom="Fe",
        ligand="H",
        charge=0,
        spin=3,
        basis="def2-qzvpd",
        r_A=SEED_R,
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
            f"{guess:>8s}: ERROR {exc!r}"
        )

        continue

    if solution is None:

        print(
            f"{guess:>8s}: "
            "NOT CONVERGED"
        )

        continue

    print(
        f"{guess:>8s}: "
        f"E={solution.energy_hartree:.12f} Eh  "
        f"<S2>={solution.s2:.7f}"
    )

    solution.mf = None

    raw.append(solution)

    gc.collect()


groups = build_candidate_groups(
    raw
)


print()
print(
    "Local groups:",
    len(groups),
)


unresolved = []


# ======================================================================
# Match every local group against existing propagated branches at R=1.40
# ======================================================================

for group in groups:

    rep = group.representative

    print()
    print(
        "{}  E={:.12f}  guesses={}".format(
            group.scout_group_id,
            rep.energy_hartree,
            ",".join(
                group.equivalent_guesses
            ),
        )
    )

    same_matches = []

    for branch_id in REFERENCE_BRANCHES:

        ref = references[
            branch_id
        ][round(SEED_R, 10)]

        comparison = compare_states(
            energy_a_hartree=
                rep.energy_hartree,
            energy_b_hartree=
                ref[
                    "energy_hartree"
                ],
            s2_a=
                rep.s2,
            s2_b=
                ref["s2"],
            dm_orth_a=
                rep.density_orth,
            dm_orth_b=
                ref[
                    "density_orth"
                ],
            spin_a=
                rep.spin,
            spin_b=3,
        )

        print(
            "  vs {:>15s}: "
            "dE={:9.4f} meV  "
            "dTot={:.3e}  "
            "dSpin={:.3e}  "
            "{}".format(
                branch_id,
                comparison.delta_energy_meV,
                comparison.total_spectrum_max,
                comparison.spin_spectrum_max,
                comparison.relation.value,
            )
        )

        if (
            comparison.relation
            == StateRelation.SAME_STATE
        ):
            same_matches.append(
                branch_id
            )

    if len(same_matches) == 1:

        print(
            "  => already represented by",
            same_matches[0],
        )

    else:

        print(
            "  => unresolved/new candidate: "
            "propagate"
        )

        unresolved.append(group)


print()
print(
    "Candidates requiring PEC-level resolution:",
    len(unresolved),
)


# ======================================================================
# Propagate every unresolved candidate
# ======================================================================

candidate_branches = {}


for number, group in enumerate(
    unresolved,
    start=1,
):

    rep = group.representative

    branch_id = (
        f"FEH_N_R140_CAND{number:02d}_"
        f"S{rep.spin}_"
        f"{rep.origin_guess.upper()}"
    )

    print()
    print("#" * 120)
    print(
        "FOLLOW",
        branch_id,
    )
    print("#" * 120)

    points = follow_scout_solution(
        branch_id=branch_id,
        atom="Fe",
        ligand="H",
        seed=rep,
        settings=settings,
        r_values=R_GRID,
        max_memory_mb=2000,
    )

    candidate_branches[
        branch_id
    ] = points

    good = [
        p
        for p in points
        if p.converged
    ]

    for p in points:

        if p.converged:

            print(
                "R={:.3f} "
                "E={:.12f} "
                "<S2>={:.7f}".format(
                    p.r_A,
                    p.energy_hartree,
                    p.s2,
                )
            )

        else:

            print(
                "R={:.3f} FAILED {}".format(
                    p.r_A,
                    p.scf_path,
                )
            )

    if good:

        minimum = min(
            good,
            key=lambda p:
                p.energy_hartree,
        )

        print(
            "Coarse minimum: "
            "R={:.3f} A "
            "E={:.12f} Eh".format(
                minimum.r_A,
                minimum.energy_hartree,
            )
        )


# ======================================================================
# PEC-level comparison against both established branches
# ======================================================================

print()
print("=" * 120)
print("PEC-LEVEL BRANCH COMPARISON")
print("=" * 120)


for candidate_id, points in candidate_branches.items():

    candidate_map = {
        round(p.r_A, 10): p
        for p in points
        if p.converged
    }

    print()
    print("-" * 120)
    print(candidate_id)
    print("-" * 120)

    for reference_id in REFERENCE_BRANCHES:

        ref_map = references[
            reference_id
        ]

        common = sorted(
            set(candidate_map)
            & set(ref_map)
        )

        relations = []

        print()
        print(
            "Against",
            reference_id,
        )

        for r in common:

            candidate = candidate_map[
                r
            ]

            ref = ref_map[
                r
            ]

            comparison = compare_states(
                energy_a_hartree=
                    candidate.energy_hartree,
                energy_b_hartree=
                    ref[
                        "energy_hartree"
                    ],
                s2_a=
                    candidate.s2,
                s2_b=
                    ref["s2"],
                dm_orth_a=
                    candidate.density_orth,
                dm_orth_b=
                    ref[
                        "density_orth"
                    ],
                spin_a=
                    candidate.spin,
                spin_b=3,
            )

            relations.append(
                comparison.relation.value
            )

            print(
                "R={:.3f} "
                "dE={:9.4f} meV "
                "dTot={:.3e} "
                "dSpin={:.3e} "
                "{}".format(
                    r,
                    comparison.delta_energy_meV,
                    comparison.total_spectrum_max,
                    comparison.spin_spectrum_max,
                    comparison.relation.value,
                )
            )

        counts = Counter(
            relations
        )

        print(
            "Summary vs {}: {}".format(
                reference_id,
                dict(counts),
            )
        )


print()
print("=" * 120)
print("BRANCH RESOLUTION DIAGNOSTIC COMPLETE")
print("=" * 120)
