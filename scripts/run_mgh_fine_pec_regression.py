#!/usr/bin/env python3

import csv
from pathlib import Path

from diatomic_ea_v09.fine_pec import (
    FineGridPolicy,
    run_fine_pec,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.scf import (
    HARTREE_TO_EV,
)
from diatomic_ea_v09.stability import (
    optimize_internal_stability,
)
from diatomic_ea_v09.state_scout import (
    run_guess,
)
from diatomic_ea_v09.ground_state import (
    scout_solution_from_mf,
)


OUTDIR = Path(
    "pilot_runs/fine_pec_mgh"
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

        "coarse_r":
            1.80,

        "guess":
            "minao",
    },

    {
        "label":
            "MGH_ANION_GS",

        "charge":
            -1,

        "spin":
            0,

        "coarse_r":
            1.90,

        "guess":
            "minao",
    },
]


summaries = {}


for case in CASES:

    print()
    print("=" * 120)
    print(case["label"])
    print("=" * 120)

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
            case["coarse_r"],

        max_memory_mb=2000,
    )

    seed = run_guess(
        spec,
        settings,
        case["guess"],
    )

    if seed is None:
        raise RuntimeError(
            case["label"]
            + ": seed failed"
        )

    stability = (
        optimize_internal_stability(
            seed.mf,
            settings,
            max_iterations=6,
        )
    )

    print(
        "Coarse seed R={:.3f} A".format(
            case["coarse_r"]
        )
    )

    print(
        "Initial stability:",
        stability.initially_stable,
    )

    print(
        "Final stability:",
        stability.finally_stable,
    )

    print(
        "Stability DeltaE [meV]: "
        "{:+.6f}".format(
            stability.total_delta_energy_meV
        )
    )

    if not stability.finally_stable:
        raise RuntimeError(
            case["label"]
            + ": canonical seed not stable"
        )

    canonical_seed = (
        scout_solution_from_mf(
            mf=
                stability.final_mf,

            template=
                seed,

            origin_guess=
                "fine_canonical_seed",
        )
    )

    # Preserve exact coarse-center metadata.
    canonical_seed.r_A = (
        case["coarse_r"]
    )

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
            case["coarse_r"],

        policy=
            policy,

        max_memory_mb=2000,
    )

    print()
    print(
        "Fine grid: {:.3f} - {:.3f} A".format(
            min(
                p.r_A
                for p in fine.points
            ),
            max(
                p.r_A
                for p in fine.points
            ),
        )
    )

    print(
        "Converged points:",
        sum(
            p.converged
            for p in fine.points
        ),
    )

    print(
        "Discrete minimum: "
        "R={:.3f} A  "
        "E={:.12f} Eh".format(
            fine.discrete_min_r_A,
            fine.discrete_min_energy_hartree,
        )
    )

    print(
        "Minimum at boundary:",
        fine.minimum_at_boundary,
    )

    print(
        "Quadratic diagnostic points:",
        fine.quadratic_points_used,
    )

    print(
        "Quadratic fitted Re [A]:",
        (
            "-"
            if fine.fitted_min_r_A
            is None
            else
            "{:.8f}".format(
                fine.fitted_min_r_A
            )
        ),
    )

    print(
        "Quadratic fitted Emin [Eh]:",
        (
            "-"
            if fine.fitted_min_energy_hartree
            is None
            else
            "{:.12f}".format(
                fine.fitted_min_energy_hartree
            )
        ),
    )

    print(
        "Curvature [Eh/A^2]:",
        (
            "-"
            if fine.curvature_hartree_per_A2
            is None
            else
            "{:.8f}".format(
                fine.curvature_hartree_per_A2
            )
        ),
    )


    good = [
        p
        for p in fine.points
        if p.converged
    ]

    emin = min(
        p.energy_hartree
        for p in good
    )

    csv_path = (
        OUTDIR
        / (
            case["label"]
            + ".csv"
        )
    )

    with csv_path.open(
        "w",
        newline="",
    ) as fh:

        fields = [
            "r_A",
            "energy_hartree",
            "relative_energy_meV",
            "s2",
            "observed_multiplicity",
            "homo_eV",
            "lumo_eV",
            "gap_eV",
            "parent_r_A",
            "point_source",
            "scf_path",
        ]

        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )

        writer.writeheader()

        for p in good:

            writer.writerow({
                "r_A":
                    p.r_A,

                "energy_hartree":
                    p.energy_hartree,

                "relative_energy_meV":
                    (
                        p.energy_hartree
                        - emin
                    )
                    * HARTREE_TO_EV
                    * 1000.0,

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

                "parent_r_A":
                    p.parent_r_A,

                "point_source":
                    p.point_source,

                "scf_path":
                    p.scf_path,
            })


    summaries[
        case["label"]
    ] = fine


print()
print("=" * 120)
print("FINE-GRID ELECTRONIC EA DIAGNOSTIC")
print("=" * 120)


neutral = summaries[
    "MGH_NEUTRAL_GS"
]

anion = summaries[
    "MGH_ANION_GS"
]


ea_discrete = (
    neutral.discrete_min_energy_hartree
    - anion.discrete_min_energy_hartree
) * HARTREE_TO_EV


print(
    "EA_el from discrete fine minima [eV]: "
    "{:.8f}".format(
        ea_discrete
    )
)


if (
    neutral.fitted_min_energy_hartree
    is not None
    and
    anion.fitted_min_energy_hartree
    is not None
):

    ea_fit = (
        neutral.fitted_min_energy_hartree
        - anion.fitted_min_energy_hartree
    ) * HARTREE_TO_EV

    print(
        "EA_el from quadratic fitted minima [eV]: "
        "{:.8f}".format(
            ea_fit
        )
    )


print()
print(
    "No pointwise stability QC or ZPE "
    "has yet been applied."
)

print("=" * 120)
print("MgH FINE PEC REGRESSION COMPLETE")
print("=" * 120)
