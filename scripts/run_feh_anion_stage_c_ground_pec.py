#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.ground_state import (
    DiscoveredBranch,
    canonicalize_branch_at_minimum,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


OUTDIR = Path(
    "pilot_runs/feh_anion_checkpoint"
)

GRAPH_PATH = (
    OUTDIR / "seed_branch_graph.pkl"
)

CHECKPOINT_PATH = (
    OUTDIR / "stage_c_ground_pec.pkl"
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
    conv_tol=1e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def atomic_pickle(
    payload,
):
    tmp = Path(
        str(CHECKPOINT_PATH)
        + ".tmp"
    )

    with tmp.open("wb") as fh:
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
        CHECKPOINT_PATH,
    )


print("=" * 132)
print("FeH ANION — STAGE C: FULL GS-RELEVANT COARSE PEC")
print("=" * 132)
print(flush=True)


with GRAPH_PATH.open(
    "rb"
) as fh:

    graph = pickle.load(
        fh
    )


nodes = graph[
    "nodes"
]

TARGET_COMPONENT_ID = (
    "SEED_BRANCH_043"
)


components = [
    component
    for component
    in graph[
        "components"
    ]
    if (
        component[
            "component_id"
        ]
        == TARGET_COMPONENT_ID
    )
]


print(
    "Selected GS-relevant components:",
    len(components),
)

for component in components:

    print(
        component[
            "component_id"
        ],
        "nodes=",
        component[
            "n_members"
        ],
        "seeds=",
        component[
            "seed_r_values"
        ],
        "Emin=",
        component[
            "minimum_local_energy_hartree"
        ],
    )

print(flush=True)


if len(components) != 1:

    raise RuntimeError(
        "Expected exactly one explicitly selected "
        "FeH- GS component from Stage A¾."
    )


if int(
    components[0][
        "spin"
    ]
) != 4:

    raise RuntimeError(
        "Selected FeH- GS component does not "
        "have expected 2S=4."
    )


# =====================================================================
# Resume
# =====================================================================

if CHECKPOINT_PATH.exists():

    with CHECKPOINT_PATH.open(
        "rb"
    ) as fh:

        state = pickle.load(
            fh
        )

    raw_branches = dict(
        state.get(
            "raw_branches",
            {},
        )
    )

    canonical_branches = dict(
        state.get(
            "canonical_branches",
            {},
        )
    )

    print(
        "RESUMING Stage C"
    )

else:

    raw_branches = {}
    canonical_branches = {}

    print(
        "Starting new Stage C"
    )


# =====================================================================
# Full PEC for each quartet component.
#
# Representative = energetically lowest already-stabilized node
# belonging to that component.
# =====================================================================

for index, component in enumerate(
    components,
    start=1,
):

    cid = component[
        "component_id"
    ]


    print()
    print("#" * 132)

    print(
        "COMPONENT {}/{}  {}".format(
            index,
            len(components),
            cid,
        ),
        flush=True,
    )

    print("#" * 132)


    member_ids = (
        component[
            "members"
        ]
    )


    representative_node_id = min(
        member_ids,
        key=lambda nid:
            nodes[nid][
                "energy_hartree"
            ],
    )


    representative_node = (
        nodes[
            representative_node_id
        ]
    )

    seed = (
        representative_node[
            "group"
        ]
        .representative
    )


    print(
        "Representative node:",
        representative_node_id,
    )

    print(
        "Seed R [A]:",
        seed.r_A,
    )

    print(
        "Seed energy [Eh]:",
        "{:.12f}".format(
            seed.energy_hartree
        ),
    )


    # -----------------------------------------------------------------
    # Raw full coarse PEC
    # -----------------------------------------------------------------

    if cid not in raw_branches:

        branch_id = (
            "FEH_A_"
            + cid
            + "_S4"
        )


        print(
            "FULL_RAW_PEC_BEGIN",
            branch_id,
            flush=True,
        )


        points = follow_scout_solution(
            branch_id=
                branch_id,

            atom="Fe",
            ligand="H",

            seed=
                seed,

            settings=
                settings,

            r_values=
                COARSE_R,

            max_memory_mb=3000,
        )


        branch = DiscoveredBranch(
            branch_id=
                branch_id,

            charge=-1,

            spin=4,

            multiplicity=
                seed.multiplicity,

            source_seed_r_A=
                float(
                    seed.r_A
                ),

            source_scout_group_id=
                cid,

            source_guesses=
                list(
                    seed.equivalent_guesses
                ),

            seed_solution=
                seed,

            raw_points=
                points,
        )


        raw_branches[
            cid
        ] = branch


        atomic_pickle({
            "raw_branches":
                raw_branches,

            "canonical_branches":
                canonical_branches,
        })


        minimum = (
            branch.raw_minimum
        )


        print(
            "FULL_RAW_PEC_DONE "
            "conv={}/{} "
            "Rmin={:.3f} "
            "Emin={:.12f}".format(
                sum(
                    point.converged
                    for point in points
                ),
                len(points),
                minimum.r_A,
                minimum.energy_hartree,
            ),
            flush=True,
        )

    else:

        branch = (
            raw_branches[
                cid
            ]
        )

        print(
            "RAW PEC already checkpointed.",
            flush=True,
        )


    # -----------------------------------------------------------------
    # Existing canonicalization:
    # reconstruct raw minimum -> Stability -> canonical full PEC.
    # -----------------------------------------------------------------

    if cid not in canonical_branches:

        print(
            "CANONICALIZATION_BEGIN",
            cid,
            flush=True,
        )


        canonical = (
            canonicalize_branch_at_minimum(
                branch=
                    branch,

                atom="Fe",
                ligand="H",

                basis=
                    "def2-qzvpd",

                settings=
                    settings,

                coarse_r_values=
                    COARSE_R,

                max_memory_mb=3000,

                max_stability_iterations=6,
            )
        )


        canonical_branches[
            cid
        ] = canonical


        atomic_pickle({
            "raw_branches":
                raw_branches,

            "canonical_branches":
                canonical_branches,
        })


        print(
            "CANONICALIZATION_DONE "
            "status={} "
            "reconstruction={} "
            "stability={}".format(
                canonical.status,

                (
                    None
                    if canonical
                    .reconstruction_relation
                    is None
                    else canonical
                    .reconstruction_relation
                    .value
                ),

                (
                    None
                    if canonical
                    .stability_relation
                    is None
                    else canonical
                    .stability_relation
                    .value
                ),
            ),
            flush=True,
        )

    else:

        canonical = (
            canonical_branches[
                cid
            ]
        )

        print(
            "Canonical PEC already checkpointed.",
            flush=True,
        )


    if (
        canonical.valid
        and canonical.minimum
        is not None
    ):

        print(
            "CANONICAL MINIMUM "
            "R={:.3f} "
            "E={:.12f}".format(
                canonical.minimum.r_A,
                canonical.minimum
                .energy_hartree,
            )
        )


        stability = (
            canonical
            .stability_result
        )


        if stability is not None:

            print(
                "STABILITY "
                "initial={} "
                "final={} "
                "reopts={} "
                "dE={:+.6f} meV".format(
                    stability
                    .initially_stable,

                    stability
                    .finally_stable,

                    stability
                    .reoptimizations,

                    stability
                    .total_delta_energy_meV,
                )
            )


# =====================================================================
# Final comparison
# =====================================================================

print()
print("=" * 132)
print("STAGE C FeH- GS COMPONENT RESULT")
print("=" * 132)


valid = []


for component in components:

    cid = (
        component[
            "component_id"
        ]
    )

    canonical = (
        canonical_branches[
            cid
        ]
    )


    if (
        canonical.valid
        and canonical.minimum
        is not None
    ):

        valid.append(
            (
                cid,
                canonical,
            )
        )


        print(
            "{}  "
            "Rmin={:.3f}  "
            "Emin={:.12f}  "
            "stab_relation={}".format(
                cid,

                canonical.minimum
                .r_A,

                canonical.minimum
                .energy_hartree,

                (
                    None
                    if canonical
                    .stability_relation
                    is None
                    else canonical
                    .stability_relation
                    .value
                ),
            )
        )

    else:

        print(
            cid,
            "INVALID",
            canonical.status,
        )


if valid:

    winner_id, winner = min(
        valid,
        key=lambda item:
            item[1]
            .minimum
            .energy_hartree,
    )


    print()
    print(
        "SELECTED CANONICAL FeH- GS BRANCH:",
        winner_id,
    )

    print(
        "Rmin [A]:",
        winner.minimum.r_A,
    )

    print(
        "Emin [Eh]:",
        "{:.12f}".format(
            winner.minimum
            .energy_hartree
        ),
    )


    # ==========================================================
    # Stable-seed competitor diagnostic
    # ==========================================================

    print()
    print("=" * 132)
    print("STABLE-SEED SPIN COMPETITOR DIAGNOSTIC")
    print("=" * 132)


    minima_by_spin = {}


    for node in nodes.values():

        group = node[
            "group"
        ]

        spin = int(
            group.spin
        )

        energy = float(
            node[
                "energy_hartree"
            ]
        )

        current = (
            minima_by_spin.get(
                spin
            )
        )

        if (
            current is None
            or energy
            < current[
                "energy_hartree"
            ]
        ):
            minima_by_spin[
                spin
            ] = {
                "energy_hartree":
                    energy,

                "r_A":
                    float(
                        group
                        .representative
                        .r_A
                    ),

                "node_id":
                    next(
                        nid
                        for nid, candidate
                        in nodes.items()
                        if candidate is node
                    ),
            }


    for spin in sorted(
        minima_by_spin
    ):

        record = (
            minima_by_spin[
                spin
            ]
        )

        gap = (
            record[
                "energy_hartree"
            ]
            - winner.minimum
            .energy_hartree
        )

        print(
            "2S={}  "
            "Rseed={:.2f}  "
            "Eseed={:.12f}  "
            "gap_vs_selected_min={:+.9f} Eh  "
            "node={}".format(
                spin,

                record[
                    "r_A"
                ],

                record[
                    "energy_hartree"
                ],

                gap,

                record[
                    "node_id"
                ],
            )
        )


    # ==========================================================
    # Short-R diagnostic at R=1.20 A
    # ==========================================================

    print()
    print("=" * 132)
    print("SHORT-R DIAGNOSTIC AT R=1.20 A")
    print("=" * 132)


    canonical_points = list(
        getattr(
            winner,
            "canonical_points",
            [],
        )
    )


    selected_r12 = [
        point
        for point
        in canonical_points
        if (
            point.converged
            and abs(
                float(
                    point.r_A
                )
                - 1.20
            )
            < 1.0e-8
        )
    ]


    if selected_r12:

        s4_r12 = (
            selected_r12[0]
        )

        print(
            "Selected 2S=4 canonical branch: "
            "E(R=1.20)={:.12f} Eh".format(
                s4_r12
                .energy_hartree
            )
        )


        competitors_r12 = []


        for nid, node in nodes.items():

            group = node[
                "group"
            ]

            r_A = float(
                group
                .representative
                .r_A
            )

            if (
                abs(
                    r_A - 1.20
                )
                < 1.0e-8
                and int(
                    group.spin
                ) != 4
            ):
                competitors_r12.append(
                    (
                        int(
                            group.spin
                        ),

                        float(
                            node[
                                "energy_hartree"
                            ]
                        ),

                        nid,
                    )
                )


        for spin, energy, nid in sorted(
            competitors_r12,
            key=lambda item:
                (
                    item[0],
                    item[1],
                ),
        ):

            print(
                "Stable scout 2S={}  "
                "E={:.12f} Eh  "
                "delta_vs_S4={:+.9f} Eh  "
                "node={}".format(
                    spin,
                    energy,
                    energy
                    - s4_r12
                    .energy_hartree,
                    nid,
                )
            )

    else:

        print(
            "No converged canonical 2S=4 "
            "point available at R=1.20 A."
        )


print()
print(
    "Checkpoint:",
    CHECKPOINT_PATH,
)

print("=" * 132)
print("FeH ANION STAGE C COMPLETE")
print("=" * 132)
