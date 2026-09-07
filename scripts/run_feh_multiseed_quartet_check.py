#!/usr/bin/env python3

import csv
import gc
from pathlib import Path

import numpy as np

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.state_identity import (
    compare_states,
)
from diatomic_ea_v09.state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


REFDIR = Path(
    "pilot_runs/coarse_branch_feh"
)

REFERENCE_BRANCHES = [
    "FEH_N_S3_G01",
    "FEH_N_S3_G02",
]

SEED_R_VALUES = [
    1.40,
    1.70,
]


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def load_reference_at_r(
    branch_id,
    target_r,
):

    csv_path = (
        REFDIR
        / f"{branch_id}.csv"
    )

    npz_path = (
        REFDIR
        / f"{branch_id}_densities.npz"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    if not npz_path.exists():
        raise FileNotFoundError(
            npz_path
        )

    with csv_path.open() as fh:

        rows = list(
            csv.DictReader(fh)
        )

    matching = [
        row
        for row in rows
        if (
            row["converged"]
            == "True"
            and abs(
                float(row["r_A"])
                - float(target_r)
            ) < 1.0e-8
        )
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"{branch_id}: expected one "
            f"CSV point at R={target_r}, "
            f"found {len(matching)}"
        )

    row = matching[0]

    archive = np.load(
        npz_path
    )

    r_values = np.asarray(
        archive["r_A"],
        dtype=float,
    )

    indices = np.where(
        np.isclose(
            r_values,
            target_r,
            atol=1.0e-8,
            rtol=0.0,
        )
    )[0]

    if len(indices) != 1:
        raise RuntimeError(
            f"{branch_id}: expected one "
            f"density at R={target_r}, "
            f"found {len(indices)}"
        )

    index = int(
        indices[0]
    )

    return {
        "branch_id":
            branch_id,

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
                archive[
                    "density_orth"
                ][index],
                copy=True,
            ),
    }


print("=" * 120)
print("FeH MULTI-SEED QUARTET DISCOVERY CHECK")
print("=" * 120)

print(
    "Independent local scouts at:",
    SEED_R_VALUES,
)

print(
    "Reference branches:",
    REFERENCE_BRANCHES,
)


for seed_r in SEED_R_VALUES:

    print()
    print("#" * 120)
    print(
        "INDEPENDENT LOCAL SCOUT "
        f"AT R={seed_r:.2f} A"
    )
    print("#" * 120)

    raw = []

    for guess in DEFAULT_GUESSES:

        spec = MoleculeSpec(
            atom="Fe",
            ligand="H",
            charge=0,
            spin=3,
            basis="def2-qzvpd",
            r_A=seed_r,
            max_memory_mb=2000,
        )

        try:

            sol = run_guess(
                spec,
                settings,
                guess,
            )

        except Exception as exc:

            print(
                f"{guess:>8s}: "
                f"ERROR {exc!r}"
            )

            continue

        if sol is None:

            print(
                f"{guess:>8s}: "
                "NOT CONVERGED"
            )

            continue

        print(
            f"{guess:>8s}: "
            f"E={sol.energy_hartree:.12f} Eh  "
            f"<S2>={sol.s2:.7f}"
        )

        sol.mf = None

        raw.append(sol)

        gc.collect()


    groups = build_candidate_groups(
        raw
    )

    print()
    print(
        "Local candidate groups:",
        len(groups),
    )

    references = {
        branch_id:
            load_reference_at_r(
                branch_id,
                seed_r,
            )
        for branch_id
        in REFERENCE_BRANCHES
    }


    for group in groups:

        rep = group.representative

        print()
        print(
            f"{group.scout_group_id}: "
            f"E={rep.energy_hartree:.12f} Eh  "
            f"guesses="
            f"{','.join(group.equivalent_guesses)}"
        )

        relations = []

        for branch_id in REFERENCE_BRANCHES:

            ref = references[
                branch_id
            ]

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

                spin_b=
                    3,
            )

            relations.append(
                (
                    branch_id,
                    comparison.relation.value,
                )
            )

            print(
                "  vs {:>15s}: "
                "dE={:10.6f} meV  "
                "dS2={:10.6f}  "
                "dTot={:.6e}  "
                "dSpin={:.6e}  "
                "{}".format(
                    branch_id,
                    comparison.delta_energy_meV,
                    comparison.delta_s2,
                    comparison.total_spectrum_max,
                    comparison.spin_spectrum_max,
                    comparison.relation.value,
                )
            )

        same_matches = [
            branch_id
            for branch_id, relation
            in relations
            if relation
            == "SAME_STATE"
        ]

        if len(same_matches) == 1:

            print(
                "  => maps uniquely to",
                same_matches[0],
            )

        elif len(same_matches) == 0:

            print(
                "  => NO unique SAME_STATE "
                "reference match"
            )

        else:

            print(
                "  => MULTIPLE SAME_STATE "
                "matches:",
                same_matches,
            )


print()
print("=" * 120)
print("MULTI-SEED CHECK COMPLETE")
print("=" * 120)

