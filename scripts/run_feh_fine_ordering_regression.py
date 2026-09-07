#!/usr/bin/env python3

from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.ground_state import (
    scout_solution_from_mf,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.molecule import (
    build_molecule,
)
from diatomic_ea_v09.scf import (
    HARTREE_TO_EV,
    run_uks,
)
from diatomic_ea_v09.stability import (
    optimize_internal_stability,
)
from diatomic_ea_v09.state_scout import (
    run_guess,
)


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


R_VALUES = [
    round(
        1.52 + 0.01 * i,
        10,
    )
    for i in range(11)
]


def spec(
    r_A,
):
    return MoleculeSpec(
        atom="Fe",
        ligand="H",
        charge=0,
        spin=3,
        basis="def2-qzvpd",
        r_A=float(r_A),
        max_memory_mb=2000,
    )


def make_stable_branch(
    guess,
):
    # Start from the known discovery geometry.
    seed140 = run_guess(
        spec(1.40),
        settings,
        guess,
    )

    if seed140 is None:
        raise RuntimeError(
            guess
            + " seed failed"
        )

    # Follow to the known coarse minimum region.
    coarse = follow_scout_solution(
        branch_id=
            guess.upper()
            + "_COARSE",

        atom="Fe",
        ligand="H",

        seed=
            seed140,

        settings=
            settings,

        r_values=[
            1.40,
            1.50,
            1.60,
        ],

        max_memory_mb=2000,
    )

    point160 = [
        p
        for p in coarse
        if (
            p.converged
            and abs(
                p.r_A - 1.60
            ) < 1.0e-8
        )
    ]

    if len(point160) != 1:
        raise RuntimeError(
            guess
            + ": no R=1.60 point"
        )

    mol160 = build_molecule(
        spec(1.60)
    )

    reconstruction = run_uks(
        mol160,
        settings,
        dm0=
            point160[0].density_ao,
    )

    if not reconstruction.converged:
        raise RuntimeError(
            guess
            + ": R=1.60 reconstruction failed"
        )

    stability = (
        optimize_internal_stability(
            reconstruction.mf,
            settings,
            max_iterations=6,
        )
    )

    if not stability.finally_stable:
        raise RuntimeError(
            guess
            + ": stability failed"
        )

    stable_seed = (
        scout_solution_from_mf(
            mf=
                stability.final_mf,

            template=
                seed140,

            origin_guess=
                guess
                + "_stable_R160",
        )
    )

    stable_seed.r_A = 1.60

    points = follow_scout_solution(
        branch_id=
            guess.upper()
            + "_FINE_ORDERING",

        atom="Fe",
        ligand="H",

        seed=
            stable_seed,

        settings=
            settings,

        r_values=
            R_VALUES
            + [1.60],

        max_memory_mb=2000,
    )

    return {
        round(
            p.r_A,
            10,
        ): p
        for p in points
        if p.converged
    }


print("=" * 128)
print("FeH FINE GROUND-STATE ENERGETIC ORDERING REGRESSION")
print("=" * 128)

print(
    "R range:",
    R_VALUES,
)

print()


minao = make_stable_branch(
    "minao"
)

huckel = make_stable_branch(
    "huckel"
)


print(
    " R[A]       E_MINAO [Eh]          "
    "E_HUCKEL [Eh]         "
    "E_HUCKEL-E_MINAO [meV]"
)

print("-" * 128)


gaps = []

for r in R_VALUES:

    a = minao[
        round(r, 10)
    ]

    b = huckel[
        round(r, 10)
    ]

    gap = (
        b.energy_hartree
        - a.energy_hartree
    ) * HARTREE_TO_EV * 1000.0

    gaps.append(
        gap
    )

    print(
        "{:.2f}   "
        "{:.12f}   "
        "{:.12f}   "
        "{:+.9f}".format(
            r,
            a.energy_hartree,
            b.energy_hartree,
            gap,
        )
    )


print()
print("=" * 128)
print("ORDERING SUMMARY")
print("=" * 128)

print(
    "Minimum gap [meV]: "
    "{:+.9f}".format(
        min(gaps)
    )
)

print(
    "Maximum gap [meV]: "
    "{:+.9f}".format(
        max(gaps)
    )
)

print(
    "MINAO lower at all 11 points:",
    all(
        gap > 0.0
        for gap in gaps
    ),
)

print(
    "Ordering changes sign:",
    (
        min(gaps) <= 0.0
        <= max(gaps)
    ),
)

print()
print("=" * 128)
print("FeH FINE ORDERING REGRESSION COMPLETE")
print("=" * 128)
