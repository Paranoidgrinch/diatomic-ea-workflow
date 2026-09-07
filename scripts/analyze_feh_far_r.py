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


NEUTRAL_PATH = Path(
    "pilot_runs/feh_neutral_checkpoint/"
    "stage_c_quartet_pecs.pkl"
)

ANION_PATH = Path(
    "pilot_runs/feh_anion_checkpoint/"
    "stage_c_ground_pec.pkl"
)

CHECKPOINT_PATH = Path(
    "pilot_runs/feh_far_r.pkl"
)

OUTPUT_PATH = Path(
    "pilot_runs/feh_far_r.json"
)


NEUTRAL_ID = "SEED_BRANCH_016"
ANION_ID = "SEED_BRANCH_043"


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
        3.1 + 0.1 * i,
        10,
    )
    for i in range(30)
]


MILESTONES = (
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
)


def load_branch(
    path,
    candidate_id,
):
    with path.open("rb") as fh:
        payload = pickle.load(fh)

    branch = payload[
        "canonical_branches"
    ][candidate_id]

    if not branch.valid:
        raise RuntimeError(
            candidate_id
            + " is not a valid canonical branch."
        )

    if not branch.canonical_points:
        raise RuntimeError(
            candidate_id
            + " contains no canonical points."
        )

    return branch


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
            )
            < 1.0e-8
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected one converged point at "
            "{} A; found {}".format(
                r_A,
                len(matches),
            )
        )

    return matches[0]


def extend_and_qc(
    *,
    branch,
    label,
    charge,
    spin,
):
    print()
    print("=" * 132)
    print(label)
    print("=" * 132)

    original = list(
        branch.canonical_points
    )

    print(
        "Starting branch range: "
        "{:.2f}--{:.2f} A".format(
            min(
                point.r_A
                for point in original
                if point.converged
            ),
            max(
                point.r_A
                for point in original
                if point.converged
            ),
        )
    )

    extended = extend_branch_points(
        points=original,

        atom="Fe",
        ligand="H",

        charge=charge,
        spin=spin,

        basis="def2-qzvpd",

        settings=settings,

        new_r_values=
            EXTENSION_R,

        direction="upper",

        max_memory_mb=3000,
    )

    good = [
        point
        for point in extended
        if (
            point.converged
            and point.energy_hartree
            is not None
        )
    ]

    print(
        "Extended converged range: "
        "{:.2f}--{:.2f} A".format(
            min(
                point.r_A
                for point in good
            ),
            max(
                point.r_A
                for point in good
            ),
        )
    )

    if (
        max(
            point.r_A
            for point in good
        )
        < 6.0 - 1.0e-8
    ):
        raise RuntimeError(
            label
            + " did not propagate continuously to 6.0 A."
        )


    ref = point_at(
        extended,
        6.0,
    )

    minimum = min(
        good,
        key=lambda point:
            point.energy_hartree,
    )


    records = []


    print()
    print(
        "Milestone energies relative to R=6.0 A"
    )
    print("-" * 132)


    for r_A in MILESTONES:

        point = point_at(
            extended,
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


        relative_eV = (
            (
                point.energy_hartree
                - ref.energy_hartree
            )
            * nist.HARTREE2EV
        )


        binding_to_r6_eV = (
            (
                ref.energy_hartree
                - minimum.energy_hartree
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

            "delta_vs_r6_eV":
                float(
                    relative_eV
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
                else qc
                .stability_relation
                .value
            ),
        }

        records.append(
            record
        )


        print(
            "R={:.1f} A  "
            "E={:.12f} Eh  "
            "E(R)-E(6A)={:+.8f} eV  "
            "QC={}  "
            "stable={}/{}".format(
                r_A,
                point.energy_hartree,
                relative_eV,
                qc.status,
                qc.initially_stable,
                qc.finally_stable,
            )
        )


    binding_to_r6_eV = (
        (
            ref.energy_hartree
            - minimum.energy_hartree
        )
        * nist.HARTREE2EV
    )


    print()
    print(
        "Branch minimum: "
        "R={:.3f} A  "
        "E={:.12f} Eh".format(
            minimum.r_A,
            minimum.energy_hartree,
        )
    )

    print(
        "D_e relative to R=6.0 A: "
        "{:.8f} eV".format(
            binding_to_r6_eV
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
            extended,

        "milestones":
            records,

        "minimum_r_A":
            float(
                minimum.r_A
            ),

        "minimum_energy_hartree":
            float(
                minimum.energy_hartree
            ),

        "r6_energy_hartree":
            float(
                ref.energy_hartree
            ),

        "de_vs_r6_eV":
            float(
                binding_to_r6_eV
            ),
    }


neutral = load_branch(
    NEUTRAL_PATH,
    NEUTRAL_ID,
)

anion = load_branch(
    ANION_PATH,
    ANION_ID,
)


neutral_result = (
    extend_and_qc(
        branch=neutral,

        label="FeH NEUTRAL FAR-R",

        charge=0,
        spin=3,
    )
)


anion_result = (
    extend_and_qc(
        branch=anion,

        label="FeH ANION FAR-R",

        charge=-1,
        spin=4,
    )
)


with CHECKPOINT_PATH.open(
    "wb"
) as fh:

    pickle.dump(
        {
            "neutral":
                neutral_result,

            "anion":
                anion_result,
        },
        fh,
        protocol=
            pickle.HIGHEST_PROTOCOL,
    )


json_payload = {
    "neutral": {
        key: value
        for key, value
        in neutral_result.items()
        if key != "points"
    },

    "anion": {
        key: value
        for key, value
        in anion_result.items()
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
print("FeH FAR-R ANALYSIS COMPLETE")
print("=" * 132)

print(
    "Checkpoint:",
    CHECKPOINT_PATH,
)

print(
    "Summary:",
    OUTPUT_PATH,
)
