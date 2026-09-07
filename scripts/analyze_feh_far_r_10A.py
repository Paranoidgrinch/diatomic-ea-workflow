#!/usr/bin/env python3

import json
import pickle
from pathlib import Path

from pyscf.data import nist

from diatomic_ea_v09.coarse_branch import (
    extend_branch_points,
)
from diatomic_ea_v09.fine_qc import (
    qc_fine_point,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


INPUT_PATH = Path(
    "pilot_runs/feh_far_r.pkl"
)

CHECKPOINT_PATH = Path(
    "pilot_runs/feh_far_r_10A.pkl"
)

OUTPUT_PATH = Path(
    "pilot_runs/feh_far_r_10A.json"
)


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


EXTENSION_R = [
    round(
        6.1 + 0.1 * i,
        10,
    )
    for i in range(40)
]


MILESTONES = (
    6.0,
    6.5,
    7.0,
    8.0,
    9.0,
    10.0,
)


def point_at(
    points,
    r_A,
):
    matches = [
        point
        for point in points
        if (
            point.converged
            and point.energy_hartree
            is not None
            and abs(
                float(point.r_A)
                - float(r_A)
            ) < 1.0e-8
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one converged "
            "point at {:.2f} A; found {}".format(
                r_A,
                len(matches),
            )
        )

    return matches[0]


def extend_one(
    *,
    existing,
    label,
    charge,
    spin,
):
    old_points = list(
        existing["points"]
    )

    old_max = max(
        point.r_A
        for point in old_points
        if point.converged
    )

    if abs(
        old_max - 6.0
    ) > 1.0e-8:
        raise RuntimeError(
            "{} checkpoint does not end at "
            "6.0 A: {}".format(
                label,
                old_max,
            )
        )


    print()
    print("=" * 132)
    print(label)
    print("=" * 132)

    print(
        "Existing converged endpoint:",
        old_max,
        "A",
    )


    points = extend_branch_points(
        points=old_points,

        atom="Fe",
        ligand="H",

        charge=charge,
        spin=spin,

        basis="def2-qzvpd",

        settings=settings,

        new_r_values=EXTENSION_R,

        direction="upper",

        max_memory_mb=3000,
    )


    good = [
        point
        for point in points
        if (
            point.converged
            and point.energy_hartree
            is not None
        )
    ]


    endpoint = max(
        point.r_A
        for point in good
    )


    print(
        "New converged endpoint:",
        endpoint,
        "A",
    )


    if endpoint < 10.0 - 1.0e-8:
        raise RuntimeError(
            label
            + " did not propagate continuously "
            "to 10.0 A."
        )


    ref = point_at(
        points,
        10.0,
    )


    print()
    print(
        "MILESTONES RELATIVE TO R=10.0 A"
    )
    print("-" * 132)


    milestones = []


    for r_A in MILESTONES:

        point = point_at(
            points,
            r_A,
        )


        qc = qc_fine_point(
            point=point,

            atom="Fe",
            ligand="H",

            charge=charge,
            spin=spin,

            basis="def2-qzvpd",

            settings=settings,

            max_memory_mb=3000,

            max_stability_iterations=6,
        )


        delta_eV = (
            (
                point.energy_hartree
                - ref.energy_hartree
            )
            * nist.HARTREE2EV
        )


        record = {
            "r_A":
                float(r_A),

            "energy_hartree":
                float(
                    point.energy_hartree
                ),

            "delta_vs_10A_eV":
                float(
                    delta_eV
                ),

            "homo_eV":
                float(
                    point.homo_eV
                ),

            "lumo_eV":
                float(
                    point.lumo_eV
                ),

            "gap_eV":
                float(
                    point.gap_eV
                ),

            "qc_status":
                qc.status,

            "initially_stable":
                qc.initially_stable,

            "finally_stable":
                qc.finally_stable,

            "stability_delta_meV":
                qc.stability_delta_energy_meV,

            "stability_relation": (
                None
                if qc.stability_relation
                is None
                else qc.stability_relation.value
            ),
        }


        milestones.append(
            record
        )


        print(
            "R={:.1f} A  "
            "E={:.12f} Eh  "
            "E(R)-E(10A)={:+.8f} eV  "
            "gap={:.6f} eV  "
            "QC={}  "
            "stable={}/{}".format(
                r_A,
                point.energy_hartree,
                delta_eV,
                point.gap_eV,
                qc.status,
                qc.initially_stable,
                qc.finally_stable,
            )
        )


    print()
    print(
        "TAIL STEP ENERGIES"
    )
    print("-" * 132)


    tail = [
        point
        for point in good
        if point.r_A >= 8.0 - 1.0e-8
    ]


    tail_steps = []


    for previous, current in zip(
        tail[:-1],
        tail[1:],
    ):

        delta_meV = (
            (
                current.energy_hartree
                - previous.energy_hartree
            )
            * nist.HARTREE2EV
            * 1000.0
        )


        tail_steps.append({
            "r_from_A":
                float(
                    previous.r_A
                ),

            "r_to_A":
                float(
                    current.r_A
                ),

            "delta_energy_meV":
                float(
                    delta_meV
                ),
        })


        print(
            "{:.1f}->{:.1f} A  "
            "dE={:+.6f} meV".format(
                previous.r_A,
                current.r_A,
                delta_meV,
            )
        )


    return {
        "label":
            label,

        "charge":
            charge,

        "spin":
            spin,

        "points":
            points,

        "milestones":
            milestones,

        "tail_steps":
            tail_steps,

        "r10_energy_hartree":
            float(
                ref.energy_hartree
            ),
    }


if not INPUT_PATH.exists():
    raise RuntimeError(
        "Missing existing 6-A checkpoint: "
        + str(INPUT_PATH)
    )


with INPUT_PATH.open(
    "rb"
) as fh:

    existing = pickle.load(
        fh
    )


neutral = extend_one(
    existing=
        existing["neutral"],

    label=
        "FeH NEUTRAL FAR-R EXTENSION",

    charge=0,
    spin=3,
)


anion = extend_one(
    existing=
        existing["anion"],

    label=
        "FeH ANION FAR-R EXTENSION",

    charge=-1,
    spin=4,
)


with CHECKPOINT_PATH.open(
    "wb"
) as fh:

    pickle.dump(
        {
            "neutral":
                neutral,

            "anion":
                anion,
        },
        fh,
        protocol=
            pickle.HIGHEST_PROTOCOL,
    )


json_payload = {
    "neutral": {
        key: value
        for key, value
        in neutral.items()
        if key != "points"
    },

    "anion": {
        key: value
        for key, value
        in anion.items()
        if key != "points"
    },
}


with OUTPUT_PATH.open(
    "w"
) as fh:

    json.dump(
        json_payload,
        fh,
        indent=2,
    )


print()
print("=" * 132)
print("FeH FAR-R 10-A EXTENSION COMPLETE")
print("=" * 132)

print(
    "Checkpoint:",
    CHECKPOINT_PATH,
)

print(
    "Summary:",
    OUTPUT_PATH,
)
