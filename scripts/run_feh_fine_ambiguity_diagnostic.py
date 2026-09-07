#!/usr/bin/env python3

import numpy as np

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
)
from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.ground_state import (
    DiscoveredBranch,
    canonicalize_branch_at_minimum,
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
from diatomic_ea_v09.state_identity import (
    compare_states,
)
from diatomic_ea_v09.state_scout import (
    orthonormal_spin_density,
    run_guess,
)


BASIS = "def2-qzvpd"

BASE_SETTINGS = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def make_spec(
    r_A,
):
    return MoleculeSpec(
        atom="Fe",
        ligand="H",
        charge=0,
        spin=3,
        basis=BASIS,
        r_A=float(r_A),
        max_memory_mb=2000,
    )


def descriptor_from_point(
    point,
):
    return {
        "energy":
            point.energy_hartree,

        "s2":
            point.s2,

        "density":
            point.density_orth,

        "density_ao":
            point.density_ao,
    }


def descriptor_from_mf(
    mf,
):
    ss, mult = mf.spin_square()

    dm = np.asarray(
        mf.make_rdm1()
    )

    dm_orth = (
        orthonormal_spin_density(
            mf.mol,
            dm,
        )
    )

    return {
        "energy":
            float(mf.e_tot),

        "s2":
            float(ss),

        "multiplicity":
            float(mult),

        "density":
            dm_orth,

        "density_ao":
            dm,
    }


def compare(
    label_a,
    a,
    label_b,
    b,
):
    result = compare_states(
        energy_a_hartree=
            a["energy"],

        energy_b_hartree=
            b["energy"],

        s2_a=
            a["s2"],

        s2_b=
            b["s2"],

        dm_orth_a=
            a["density"],

        dm_orth_b=
            b["density"],

        spin_a=3,
        spin_b=3,
    )

    print(
        "{} vs {}: "
        "dE={:.9f} meV  "
        "dS2={:.9e}  "
        "dTot={:.9e}  "
        "dSpin={:.9e}  "
        "{}".format(
            label_a,
            label_b,

            result.delta_energy_meV,
            result.delta_s2,
            result.total_spectrum_max,
            result.spin_spectrum_max,
            result.relation.value,
        )
    )

    return result


print("=" * 132)
print("FeH FINE-STATE AMBIGUITY / DFT-GRID DIAGNOSTIC")
print("=" * 132)

print(
    "Target geometry: R = 1.570 A"
)

print(
    "Quartet only, PBE / def2-QZVPD"
)

print()


# =====================================================================
# 1. Reproduce local R=1.40 quartet groups
# =====================================================================

solutions = []

for guess in [
    "minao",
    "atom",
    "huckel",
    "hcore",
]:

    solution = run_guess(
        make_spec(1.40),
        BASE_SETTINGS,
        guess,
    )

    if solution is None:
        continue

    solution.mf = None

    solutions.append(
        solution
    )

    print(
        "{:>8s}: "
        "E={:.12f} "
        "<S2>={:.9f}".format(
            guess,
            solution.energy_hartree,
            solution.s2,
        )
    )


groups = build_candidate_groups(
    solutions
)


print()
print(
    "Local groups:",
    len(groups),
)


# We want the known huckel and minao groups only.
wanted = []

for group in groups:

    guesses = set(
        group.equivalent_guesses
    )

    if "huckel" in guesses:
        wanted.append(
            (
                "HUCKEL_BRANCH",
                group,
            )
        )

    elif (
        guesses == {"minao"}
        or "minao" in guesses
    ):
        wanted.append(
            (
                "MINAO_BRANCH",
                group,
            )
        )


if len(wanted) != 2:
    raise RuntimeError(
        "Expected exactly huckel and minao quartet groups."
    )


# =====================================================================
# 2. Coarse propagate to 1.60 and stability-canonicalize
# =====================================================================

canonical = {}


for label, group in wanted:

    representative = (
        group.representative
    )

    points = follow_scout_solution(
        branch_id=
            label + "_RAW",

        atom="Fe",
        ligand="H",

        seed=
            representative,

        settings=
            BASE_SETTINGS,

        r_values=[
            1.40,
            1.50,
            1.60,
        ],

        max_memory_mb=2000,
    )


    raw = DiscoveredBranch(
        branch_id=
            label,

        charge=0,
        spin=3,
        multiplicity=4,

        source_seed_r_A=1.40,

        source_scout_group_id=
            group.scout_group_id,

        source_guesses=
            group.equivalent_guesses,

        seed_solution=
            representative,

        raw_points=
            points,
    )


    result = canonicalize_branch_at_minimum(
        branch=
            raw,

        atom="Fe",
        ligand="H",

        basis=BASIS,

        settings=
            BASE_SETTINGS,

        coarse_r_values=[
            1.40,
            1.50,
            1.60,
        ],

        max_memory_mb=2000,

        max_stability_iterations=6,
    )


    print()
    print("-" * 132)

    print(
        label
    )

    print(
        "Canonicalization status:",
        result.status,
    )


    if result.stability_result is not None:

        s = result.stability_result

        print(
            "Stability: "
            "initial={} final={} "
            "reopts={} "
            "dE={:+.9f} meV".format(
                s.initially_stable,
                s.finally_stable,
                s.reoptimizations,
                s.total_delta_energy_meV,
            )
        )


    if not result.valid:
        raise RuntimeError(
            label
            + " failed canonicalization."
        )


    # Find canonical point at 1.60 A.
    point160 = [
        p
        for p in result.canonical_points
        if (
            p.converged
            and abs(
                p.r_A - 1.60
            ) < 1.0e-8
        )
    ]

    if len(point160) != 1:
        raise RuntimeError(
            label
            + ": no unique R=1.60 point."
        )


    # Turn R=1.60 canonical point into a seed by a direct reconstruction.
    mol160 = build_molecule(
        make_spec(1.60)
    )

    rec160 = run_uks(
        mol160,
        BASE_SETTINGS,

        dm0=
            point160[0].density_ao,
    )

    if not rec160.converged:
        raise RuntimeError(
            label
            + ": R=1.60 reconstruction failed."
        )


    from diatomic_ea_v09.ground_state import (
        scout_solution_from_mf,
    )

    seed160 = scout_solution_from_mf(
        mf=
            rec160.mf,

        template=
            representative,

        origin_guess=
            label + "_R160",
    )

    seed160.r_A = 1.60


    # Fine propagate only to R=1.57.
    fine = follow_scout_solution(
        branch_id=
            label + "_FINE",

        atom="Fe",
        ligand="H",

        seed=
            seed160,

        settings=
            BASE_SETTINGS,

        r_values=[
            1.57,
            1.60,
        ],

        max_memory_mb=2000,
    )


    point157 = [
        p
        for p in fine
        if (
            p.converged
            and abs(
                p.r_A - 1.57
            ) < 1.0e-8
        )
    ]


    if len(point157) != 1:
        raise RuntimeError(
            label
            + ": no R=1.57 point."
        )


    canonical[label] = (
        descriptor_from_point(
            point157[0]
        )
    )


print()
print("=" * 132)
print("BASELINE GRID-LEVEL 3 COMPARISON AT R=1.570 A")
print("=" * 132)

compare(
    "MINAO",
    canonical["MINAO_BRANCH"],
    "HUCKEL",
    canonical["HUCKEL_BRANCH"],
)


# =====================================================================
# 3. Rerun both R=1.57 densities at grids 3, 4, 5
# =====================================================================

grid_results = {}


for grid_level in [
    3,
    4,
    5,
]:

    print()
    print("=" * 132)
    print(
        "GRID LEVEL",
        grid_level,
    )
    print("=" * 132)


    settings = SCFSettings(
        xc="PBE",

        grid_level=
            grid_level,

        conv_tol=1.0e-9,
        max_cycle=200,
        threads=1,
        level_shift_helper=0.25,
    )


    grid_results[
        grid_level
    ] = {}


    for label in [
        "MINAO_BRANCH",
        "HUCKEL_BRANCH",
    ]:

        mol = build_molecule(
            make_spec(1.57)
        )


        result = run_uks(
            mol,
            settings,

            dm0=
                canonical[label][
                    "density_ao"
                ],
        )


        if not result.converged:

            print(
                label,
                "SCF FAILED"
            )

            continue


        desc = descriptor_from_mf(
            result.mf
        )

        grid_results[
            grid_level
        ][label] = desc


        print(
            "{}: "
            "E={:.12f} Eh "
            "<S2>={:.9f}".format(
                label,
                desc["energy"],
                desc["s2"],
            )
        )


    if (
        "MINAO_BRANCH"
        in grid_results[
            grid_level
        ]
        and
        "HUCKEL_BRANCH"
        in grid_results[
            grid_level
        ]
    ):

        print()

        compare(
            "MINAO",
            grid_results[
                grid_level
            ]["MINAO_BRANCH"],

            "HUCKEL",
            grid_results[
                grid_level
            ]["HUCKEL_BRANCH"],
        )


# =====================================================================
# 4. Track each branch itself across grid levels
# =====================================================================

print()
print("=" * 132)
print("SAME BRANCH ACROSS GRID LEVELS")
print("=" * 132)


for label in [
    "MINAO_BRANCH",
    "HUCKEL_BRANCH",
]:

    print()
    print(
        label
    )

    for a, b in [
        (3, 4),
        (4, 5),
        (3, 5),
    ]:

        if (
            label
            not in grid_results[a]
            or label
            not in grid_results[b]
        ):
            continue

        compare(
            f"GRID{a}",
            grid_results[a][label],
            f"GRID{b}",
            grid_results[b][label],
        )


# =====================================================================
# 5. Energy-gap convergence
# =====================================================================

print()
print("=" * 132)
print("ENERGY-GAP CONVERGENCE")
print("=" * 132)


for grid_level in [
    3,
    4,
    5,
]:

    data = grid_results[
        grid_level
    ]

    if (
        "MINAO_BRANCH" not in data
        or
        "HUCKEL_BRANCH" not in data
    ):
        continue

    gap_meV = (
        data["HUCKEL_BRANCH"][
            "energy"
        ]
        - data["MINAO_BRANCH"][
            "energy"
        ]
    ) * HARTREE_TO_EV * 1000.0

    print(
        "Grid {}: "
        "E(HUCKEL)-E(MINAO) = "
        "{:+.9f} meV".format(
            grid_level,
            gap_meV,
        )
    )


print()
print("=" * 132)
print("FeH FINE AMBIGUITY DIAGNOSTIC COMPLETE")
print("=" * 132)
