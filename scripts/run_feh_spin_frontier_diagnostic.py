#!/usr/bin/env python3

import csv
from pathlib import Path

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
)
from diatomic_ea_v09.discovery_policy import (
    DiscoveryPolicy,
    assess_spin_frontier,
    discovery_spin_values,
    make_scout_seed_grid,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.scf import (
    HARTREE_TO_EV,
)
from diatomic_ea_v09.state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


OUTDIR = Path(
    "pilot_runs/spin_frontier_feh"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


policy = DiscoveryPolicy(
    seed_spacing_A=0.20,
    seed_margin_A=0.20,

    initial_spin_sectors=5,
    max_spin_sectors=8,

    spin_tail_length=3,
)


COARSE_R = [
    round(
        1.0 + 0.1 * i,
        10,
    )
    for i in range(21)
]


SEEDS = make_scout_seed_grid(
    COARSE_R,
    policy=policy,
)


ALL_SPINS = discovery_spin_values(
    atom="Fe",
    ligand="H",
    charge=0,
    policy=policy,
)


if not ALL_SPINS:
    raise RuntimeError(
        "No physically allowed spins returned."
    )


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


print("=" * 132)
print("FeH NEUTRAL LOCAL SPIN-FRONTIER DIAGNOSTIC")
print("=" * 132)

print(
    "Seeds:",
    SEEDS,
)

print(
    "Available development spin sectors:",
    ALL_SPINS,
)

print(
    "Initial sectors:",
    policy.initial_spin_sectors,
)

print(
    "Maximum sectors:",
    policy.max_spin_sectors,
)

print()


# Cache results so extending the spin frontier does not rerun
# already completed seed/spin sectors.
sector_cache = {}

rows = []


def run_sector(
    seed_r,
    spin,
):
    key = (
        round(
            float(seed_r),
            10,
        ),
        int(spin),
    )

    if key in sector_cache:
        return sector_cache[key]


    solutions = []
    errors = []


    for guess in DEFAULT_GUESSES:

        spec = MoleculeSpec(
            atom="Fe",
            ligand="H",

            charge=0,
            spin=int(spin),

            basis="def2-qzvpd",

            r_A=float(
                seed_r
            ),

            max_memory_mb=2000,
        )


        try:

            solution = run_guess(
                spec,
                settings,
                guess,
            )

        except Exception as exc:

            errors.append(
                (
                    guess,
                    repr(exc),
                )
            )

            continue


        if solution is None:
            continue


        solution.mf = None

        solutions.append(
            solution
        )


    groups = (
        build_candidate_groups(
            solutions
        )
        if solutions
        else []
    )


    minimum = (
        None
        if not groups
        else min(
            group.representative
            .energy_hartree
            for group in groups
        )
    )


    result = {
        "seed_r":
            float(seed_r),

        "spin":
            int(spin),

        "n_solutions":
            len(solutions),

        "n_groups":
            len(groups),

        "minimum_hartree":
            minimum,

        "errors":
            errors,
    }


    sector_cache[key] = result

    return result


n_scan = min(
    policy.initial_spin_sectors,
    len(ALL_SPINS),
)


final_assessment = None


while True:

    scanned_spins = (
        ALL_SPINS[:n_scan]
    )


    print()
    print("#" * 132)

    print(
        "SCANNING SPINS:",
        scanned_spins,
    )

    print("#" * 132)


    minima_by_seed = {}


    for seed_r in SEEDS:

        minima_by_seed[
            seed_r
        ] = {}


        seed_results = []


        for spin in scanned_spins:

            result = run_sector(
                seed_r,
                spin,
            )

            seed_results.append(
                result
            )


            if (
                result[
                    "minimum_hartree"
                ]
                is not None
            ):

                minima_by_seed[
                    seed_r
                ][spin] = (
                    result[
                        "minimum_hartree"
                    ]
                )


        available = [
            result[
                "minimum_hartree"
            ]
            for result in seed_results
            if (
                result[
                    "minimum_hartree"
                ]
                is not None
            )
        ]


        reference = (
            min(available)
            if available
            else None
        )


        print()
        print(
            "R={:.2f} A".format(
                seed_r
            )
        )


        for result in seed_results:

            energy = (
                result[
                    "minimum_hartree"
                ]
            )


            relative = (
                None
                if (
                    energy is None
                    or reference is None
                )
                else (
                    energy - reference
                )
                * HARTREE_TO_EV
            )


            print(
                "  2S={:<2d}  "
                "groups={:<2d}  "
                "solutions={:<2d}  "
                "Emin={}  "
                "rel={} eV".format(
                    result["spin"],
                    result["n_groups"],
                    result["n_solutions"],

                    (
                        "-"
                        if energy is None
                        else
                        "{:.12f}".format(
                            energy
                        )
                    ),

                    (
                        "-"
                        if relative is None
                        else
                        "{:.6f}".format(
                            relative
                        )
                    ),
                )
            )


    assessment = assess_spin_frontier(
        scanned_spins=
            scanned_spins,

        minima_by_seed=
            minima_by_seed,

        all_allowed_spins=
            ALL_SPINS,

        tail_length=
            policy.spin_tail_length,
    )


    print()
    print("=" * 132)
    print("FRONTIER ASSESSMENT")
    print("=" * 132)

    print(
        "Scanned:",
        scanned_spins,
    )

    print(
        "Tail:",
        assessment.tail_spins,
    )

    print(
        "Closed:",
        assessment.closed,
    )

    print(
        "Physical spin space exhausted:",
        assessment.physical_spin_space_exhausted,
    )

    print(
        "Non-rising seeds:",
        assessment.failing_seed_r_values,
    )

    print(
        "Missing-tail-data seeds:",
        assessment.missing_seed_r_values,
    )


    if assessment.closed:

        final_assessment = (
            assessment
        )

        final_status = (
            "PASS_SPIN_FRONTIER"
        )

        break


    if n_scan >= len(
        ALL_SPINS
    ):

        final_assessment = (
            assessment
        )

        final_status = (
            "QC_FAIL_SPIN_FRONTIER"
        )

        break


    n_scan += 1

    print()
    print(
        "Frontier remains open; adding next spin sector:",
        ALL_SPINS[
            n_scan - 1
        ],
    )


# ----------------------------------------------------------------------
# Audit CSV
# ----------------------------------------------------------------------

for key in sorted(
    sector_cache
):

    result = (
        sector_cache[key]
    )

    rows.append({
        "seed_r_A":
            result["seed_r"],

        "spin_2S":
            result["spin"],

        "n_solutions":
            result["n_solutions"],

        "n_groups":
            result["n_groups"],

        "minimum_hartree":
            (
                ""
                if result[
                    "minimum_hartree"
                ]
                is None
                else result[
                    "minimum_hartree"
                ]
            ),

        "n_errors":
            len(
                result["errors"]
            ),
    })


with (
    OUTDIR
    / "spin_frontier.csv"
).open(
    "w",
    newline="",
) as fh:

    writer = csv.DictWriter(
        fh,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


print()
print("=" * 132)
print("FINAL SPIN-FRONTIER RESULT")
print("=" * 132)

print(
    "Status:",
    final_status,
)

print(
    "Final scanned spins:",
    ALL_SPINS[:n_scan],
)

print(
    "Scout seeds:",
    SEEDS,
)

print()
print("=" * 132)
print("FeH SPIN-FRONTIER DIAGNOSTIC COMPLETE")
print("=" * 132)
