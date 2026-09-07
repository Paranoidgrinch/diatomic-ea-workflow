#!/usr/bin/env python3

import csv
from pathlib import Path

from diatomic_ea_v09.fine_pec import (
    FineGridPolicy,
    run_fine_pec,
)
from diatomic_ea_v09.fine_qc import (
    qc_fine_pec,
)
from diatomic_ea_v09.ground_state import (
    scout_solution_from_mf,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.stability import (
    optimize_internal_stability,
)
from diatomic_ea_v09.state_scout import (
    run_guess,
)


OUTDIR = Path(
    "pilot_runs/fine_pointwise_qc_mgh"
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


policy = FineGridPolicy(
    half_width_A=0.20,
    step_A=0.01,
    quadratic_half_points=3,
)


CASES = [
    {
        "label":
            "MGH_NEUTRAL_GS",

        "charge":
            0,

        "spin":
            1,

        "center_r":
            1.80,
    },

    {
        "label":
            "MGH_ANION_GS",

        "charge":
            -1,

        "spin":
            0,

        "center_r":
            1.90,
    },
]


for case in CASES:

    print()
    print("=" * 124)
    print(case["label"])
    print("=" * 124)


    # ------------------------------------------------------------
    # Stable canonical seed
    # ------------------------------------------------------------

    spec = MoleculeSpec(
        atom="Mg",
        ligand="H",

        charge=
            case["charge"],

        spin=
            case["spin"],

        basis=
            "def2-qzvpd",

        r_A=
            case["center_r"],

        max_memory_mb=2000,
    )

    seed = run_guess(
        spec,
        settings,
        "minao",
    )

    if seed is None:
        raise RuntimeError(
            case["label"]
            + ": seed failed"
        )


    seed_stability = (
        optimize_internal_stability(
            seed.mf,
            settings,
            max_iterations=6,
        )
    )

    if not seed_stability.finally_stable:
        raise RuntimeError(
            case["label"]
            + ": seed did not become stable"
        )


    canonical_seed = (
        scout_solution_from_mf(
            mf=
                seed_stability.final_mf,

            template=
                seed,

            origin_guess=
                "fine_qc_seed",
        )
    )

    canonical_seed.r_A = (
        case["center_r"]
    )


    # ------------------------------------------------------------
    # Fine PEC
    # ------------------------------------------------------------

    fine = run_fine_pec(
        branch_id=
            case["label"],

        atom="Mg",
        ligand="H",

        seed=
            canonical_seed,

        settings=
            settings,

        center_r_A=
            case["center_r"],

        policy=
            policy,

        max_memory_mb=2000,
    )


    print(
        "Fine PEC points:",
        len(fine.points),
    )

    print(
        "Discrete minimum: "
        "R={:.3f} A".format(
            fine.discrete_min_r_A
        )
    )

    print(
        "Fitted minimum: "
        "R={:.8f} A".format(
            fine.fitted_min_r_A
        )
    )


    # ------------------------------------------------------------
    # Pointwise stability / identity QC
    # ------------------------------------------------------------

    qc = qc_fine_pec(
        points=
            fine.points,

        atom="Mg",
        ligand="H",

        charge=
            case["charge"],

        spin=
            case["spin"],

        basis=
            "def2-qzvpd",

        settings=
            settings,

        max_memory_mb=2000,

        max_stability_iterations=6,
    )


    print()
    print("POINTWISE QC SUMMARY")
    print("-" * 124)

    print(
        "Overall status:",
        qc.status,
    )

    print(
        "Total points:",
        qc.n_points,
    )

    print(
        "PASS:",
        qc.n_pass,
    )

    print(
        "Reconstruction failures:",
        qc.n_reconstruction_fail,
    )

    print(
        "Identity failures:",
        qc.n_identity_fail,
    )

    print(
        "Initially unstable:",
        qc.n_initially_unstable,
    )

    print(
        "Finally unstable:",
        qc.n_finally_unstable,
    )

    print(
        "Stability reopt SAME_STATE:",
        qc.n_stability_same_state,
    )

    print(
        "Stability reopt AMBIGUOUS:",
        qc.n_stability_ambiguous,
    )

    print(
        "Stability root changes:",
        qc.n_stability_state_change,
    )

    print(
        "Max |reconstruction dE| [meV]:",
        (
            "-"
            if qc.max_abs_reconstruction_delta_meV
            is None
            else
            "{:.9f}".format(
                qc.max_abs_reconstruction_delta_meV
            )
        ),
    )

    print(
        "Most negative stability dE [meV]:",
        (
            "-"
            if qc.most_negative_stability_delta_meV
            is None
            else
            "{:+.9f}".format(
                qc.most_negative_stability_delta_meV
            )
        ),
    )


    unusual = [
        record
        for record in qc.points
        if record.status != "PASS"
    ]


    if unusual:

        print()
        print("NON-PASS POINTS")
        print("-" * 124)

        for record in unusual:

            print(
                "R={:.3f}  "
                "status={}  "
                "recon={}  "
                "initial_stable={}  "
                "final_stable={}  "
                "stab_relation={}  "
                "stab_dE_meV={}".format(
                    record.r_A,

                    record.status,

                    (
                        "-"
                        if record.reconstruction_relation
                        is None
                        else
                        record.reconstruction_relation.value
                    ),

                    record.initially_stable,
                    record.finally_stable,

                    (
                        "-"
                        if record.stability_relation
                        is None
                        else
                        record.stability_relation.value
                    ),

                    (
                        "-"
                        if record.stability_delta_energy_meV
                        is None
                        else
                        "{:+.6f}".format(
                            record.stability_delta_energy_meV
                        )
                    ),
                )
            )


    # ------------------------------------------------------------
    # Audit CSV
    # ------------------------------------------------------------

    csv_path = (
        OUTDIR
        / (
            case["label"]
            + "_pointwise_qc.csv"
        )
    )

    with csv_path.open(
        "w",
        newline="",
    ) as fh:

        fields = [
            "r_A",
            "status",

            "stored_energy_hartree",
            "reconstructed_energy_hartree",
            "stabilized_energy_hartree",

            "reconstruction_delta_meV",
            "reconstruction_relation",

            "initially_stable",
            "finally_stable",

            "stability_reoptimizations",
            "stability_delta_energy_meV",
            "stability_relation",
        ]

        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in qc.points:

            writer.writerow({
                "r_A":
                    record.r_A,

                "status":
                    record.status,

                "stored_energy_hartree":
                    record.stored_energy_hartree,

                "reconstructed_energy_hartree":
                    record.reconstructed_energy_hartree,

                "stabilized_energy_hartree":
                    record.stabilized_energy_hartree,

                "reconstruction_delta_meV":
                    record.reconstruction_delta_meV,

                "reconstruction_relation":
                    (
                        ""
                        if record.reconstruction_relation
                        is None
                        else
                        record.reconstruction_relation.value
                    ),

                "initially_stable":
                    record.initially_stable,

                "finally_stable":
                    record.finally_stable,

                "stability_reoptimizations":
                    record.stability_reoptimizations,

                "stability_delta_energy_meV":
                    record.stability_delta_energy_meV,

                "stability_relation":
                    (
                        ""
                        if record.stability_relation
                        is None
                        else
                        record.stability_relation.value
                    ),
            })


print()
print("=" * 124)
print("MgH FINE POINTWISE QC COMPLETE")
print("=" * 124)
