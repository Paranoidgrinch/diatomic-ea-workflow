#!/usr/bin/env python3

import json
import os
import pickle
from pathlib import Path

from diatomic_ea_v09.adaptive_scout import (
    run_adaptive_local_scout,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


OUTDIR = Path(
    "pilot_runs/feh_anion_checkpoint"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCOUT_PATH = (
    OUTDIR / "scout.pkl"
)

SUMMARY_PATH = (
    OUTDIR / "scout_summary.json"
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


def atomic_pickle(
    path,
    payload,
):
    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as fh:

        pickle.dump(
            payload,
            fh,
            protocol=
                pickle.HIGHEST_PROTOCOL,
        )

        fh.flush()

        os.fsync(
            fh.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def progress(
    message,
):
    print(
        message,
        flush=True,
    )


print("=" * 132)
print("FeH ANION — STAGE A: ADAPTIVE LOCAL SCOUT")
print("=" * 132)

print(
    "Molecule: FeH-"
)

print(
    "Charge: -1"
)

print(
    "Method: PBE / def2-QZVPD"
)

print(
    "Coarse grid: 1.0--3.0 A / 0.1 A"
)

print(
    "Scout geometry policy: adaptive_scout default"
)

print(flush=True)


result = run_adaptive_local_scout(
    atom="Fe",
    ligand="H",

    charge=-1,

    coarse_r_values=
        COARSE_R,

    basis=
        "def2-qzvpd",

    settings=
        settings,

    max_memory_mb=3000,

    progress_callback=
        progress,
)


# PySCF mf objects are already removed by adaptive_scout.
atomic_pickle(
    SCOUT_PATH,
    result,
)


n_groups = sum(
    len(groups)
    for groups
    in result.groups_by_seed.values()
)


print()
print("=" * 132)
print("STAGE A RESULT")
print("=" * 132)

print(
    "Status:",
    result.status,
)

print(
    "Physical spin values:",
    result.physical_spin_values,
)

print(
    "Development spin values:",
    result.development_spin_values,
)

print(
    "Scanned spin values:",
    result.scanned_spin_values,
)

print(
    "Seed R values:",
    result.seed_r_values,
)

print(
    "Total sectors:",
    len(
        result.sector_records
    ),
)

print(
    "Total CandidateGroups:",
    n_groups,
)


frontier = result.frontier

if frontier is not None:

    print(
        "Frontier closed:",
        frontier.closed,
    )

    print(
        "Frontier tail:",
        frontier.tail_spins,
    )

    print(
        "Failing seeds:",
        frontier.failing_seed_r_values,
    )

    print(
        "Missing seeds:",
        frontier.missing_seed_r_values,
    )


print()
print("GROUP COUNTS / MINIMA BY SEED")
print("-" * 132)


counts_by_seed = {}


for seed_r in result.seed_r_values:

    seed_key = round(
        float(seed_r),
        10,
    )

    groups = (
        result.groups_by_seed[
            seed_key
        ]
    )


    by_spin = {}


    for group in groups:

        spin = int(
            group.spin
        )

        entry = by_spin.setdefault(
            spin,
            {
                "groups": 0,
                "minimum_energy_hartree":
                    None,
            },
        )

        entry[
            "groups"
        ] += 1


        energy = float(
            group.representative
            .energy_hartree
        )


        if (
            entry[
                "minimum_energy_hartree"
            ] is None

            or energy
            < entry[
                "minimum_energy_hartree"
            ]
        ):
            entry[
                "minimum_energy_hartree"
            ] = energy


    counts_by_seed[
        str(seed_key)
    ] = by_spin


    printable = {
        spin: (
            entry[
                "groups"
            ],
            entry[
                "minimum_energy_hartree"
            ],
        )
        for spin, entry
        in sorted(
            by_spin.items()
        )
    }


    print(
        "R={:.2f}: {}".format(
            seed_key,
            printable,
        )
    )


summary = {
    "molecule":
        result.molecule,

    "charge":
        result.charge,

    "status":
        result.status,

    "physical_spin_values":
        list(
            result.physical_spin_values
        ),

    "development_spin_values":
        list(
            result.development_spin_values
        ),

    "scanned_spin_values":
        list(
            result.scanned_spin_values
        ),

    "seed_r_values":
        list(
            result.seed_r_values
        ),

    "n_sectors":
        len(
            result.sector_records
        ),

    "n_candidate_groups":
        n_groups,

    "frontier": (
        None
        if frontier is None
        else {
            "closed":
                frontier.closed,

            "tail_spins":
                list(
                    frontier.tail_spins
                ),

            "failing_seed_r_values":
                list(
                    frontier
                    .failing_seed_r_values
                ),

            "missing_seed_r_values":
                list(
                    frontier
                    .missing_seed_r_values
                ),
        }
    ),

    "counts_by_seed":
        counts_by_seed,
}


with SUMMARY_PATH.open(
    "w"
) as fh:

    json.dump(
        summary,
        fh,
        indent=2,
    )


print()
print(
    "Checkpoint:",
    SCOUT_PATH,
)

print(
    "Checkpoint size [MB]: "
    "{:.2f}".format(
        SCOUT_PATH.stat().st_size
        / 1024**2
    )
)

print(
    "Summary:",
    SUMMARY_PATH,
)

print("=" * 132)
print("FeH ANION STAGE A COMPLETE")
print("=" * 132)
