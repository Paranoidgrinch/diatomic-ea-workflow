#!/usr/bin/env python3

import json
import os
import pickle
from pathlib import Path

from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.ground_state import (
    compare_solution_to_point,
    point_at_r,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)
from diatomic_ea_v09.state_identity import (
    StateRelation,
)


OUTDIR = Path(
    "pilot_runs/feh_neutral_checkpoint"
)

STABLE_SCOUT_PATH = (
    OUTDIR / "stable_scout.pkl"
)

WORK_PATH = (
    OUTDIR / "stage_a_three_quarter_work.pkl"
)

FINAL_PATH = (
    OUTDIR / "seed_branch_graph.pkl"
)

SUMMARY_PATH = (
    OUTDIR / "seed_branch_graph_summary.json"
)


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def node_id(
    seed_r,
    group,
):
    return (
        "R{:.2f}_S{}_{}".format(
            float(seed_r),
            int(group.spin),
            str(group.scout_group_id),
        )
    )


def atomic_pickle(
    path,
    payload,
):
    tmp = Path(
        str(path) + ".tmp"
    )

    with tmp.open("wb") as fh:

        pickle.dump(
            payload,
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        fh.flush()
        os.fsync(
            fh.fileno()
        )

    os.replace(
        tmp,
        path,
    )


print("=" * 132)
print("FeH NEUTRAL — STAGE A¾: STABLE SEED-BRANCH GRAPH")
print("=" * 132)
print(flush=True)


if not STABLE_SCOUT_PATH.exists():

    raise RuntimeError(
        "Missing Stage-A½ checkpoint: "
        + str(STABLE_SCOUT_PATH)
    )


with STABLE_SCOUT_PATH.open(
    "rb"
) as fh:

    stable = pickle.load(
        fh
    )


groups_by_seed = (
    stable[
        "stable_groups_by_seed"
    ]
)

seeds = sorted(
    float(r)
    for r in stable[
        "seed_r_values"
    ]
)


nodes = {}

for seed_r in seeds:

    seed_key = round(
        seed_r,
        10,
    )

    for group in (
        groups_by_seed[
            seed_key
        ]
    ):

        nid = node_id(
            seed_key,
            group,
        )

        nodes[nid] = {
            "seed_r_A":
                seed_key,

            "spin":
                int(group.spin),

            "group":
                group,

            "energy_hartree":
                float(
                    group.representative
                    .energy_hartree
                ),
        }


print(
    "Stable local nodes:",
    len(nodes),
)

print(
    "Seeds:",
    seeds,
)

print(flush=True)


# ====================================================================
# Build one-step directed propagation tasks.
#
# A task is:
#     one local stable root at Ri
#         ->
#     one neighboring scout geometry Rj
#
# The propagated state is compared against ALL stable local groups
# of the same spin at Rj.
# ====================================================================

tasks = []


for i, seed_r in enumerate(
    seeds
):

    neighbors = []

    if i > 0:
        neighbors.append(
            seeds[
                i - 1
            ]
        )

    if i + 1 < len(seeds):
        neighbors.append(
            seeds[
                i + 1
            ]
        )


    seed_key = round(
        seed_r,
        10,
    )


    for group in (
        groups_by_seed[
            seed_key
        ]
    ):

        source_id = node_id(
            seed_key,
            group,
        )


        for target_r in neighbors:

            tasks.append({
                "source_id":
                    source_id,

                "source_r_A":
                    seed_key,

                "target_r_A":
                    round(
                        float(target_r),
                        10,
                    ),
            })


print(
    "One-step propagation tasks:",
    len(tasks),
)


# ====================================================================
# Resume
# ====================================================================

if WORK_PATH.exists():

    with WORK_PATH.open(
        "rb"
    ) as fh:

        work = pickle.load(
            fh
        )

    completed_tasks = set(
        tuple(x)
        for x in work[
            "completed_tasks"
        ]
    )

    edge_records = list(
        work[
            "edge_records"
        ]
    )

    propagation_records = list(
        work[
            "propagation_records"
        ]
    )


    print(
        "RESUMING Stage A¾"
    )

    print(
        "Completed tasks:",
        len(
            completed_tasks
        ),
    )

else:

    completed_tasks = set()

    edge_records = []

    propagation_records = []

    print(
        "Starting new Stage A¾ checkpoint."
    )


print(
    "Remaining tasks:",
    len(tasks)
    - len(completed_tasks),
)

print(flush=True)


# ====================================================================
# Run every directed nearest-neighbor propagation once.
# ====================================================================

for task_index, task in enumerate(
    tasks,
    start=1,
):

    task_key = (
        task[
            "source_id"
        ],
        task[
            "target_r_A"
        ],
    )


    if task_key in completed_tasks:

        print(
            "SKIP {}/{} {} -> R={:.2f}".format(
                task_index,
                len(tasks),
                task[
                    "source_id"
                ],
                task[
                    "target_r_A"
                ],
            ),
            flush=True,
        )

        continue


    source = (
        nodes[
            task[
                "source_id"
            ]
        ]
    )

    source_group = (
        source[
            "group"
        ]
    )

    representative = (
        source_group
        .representative
    )


    print()
    print("-" * 132)

    print(
        "TASK {}/{}  {}  -> R={:.2f}".format(
            task_index,
            len(tasks),
            task[
                "source_id"
            ],
            task[
                "target_r_A"
            ],
        ),
        flush=True,
    )


    branch_id = (
        "GRAPH_"
        + task[
            "source_id"
        ]
        + "_TO_"
        + "R{:.2f}".format(
            task[
                "target_r_A"
            ]
        )
    )


    points = follow_scout_solution(
        branch_id=
            branch_id,

        atom="Fe",
        ligand="H",

        seed=
            representative,

        settings=
            settings,

        r_values=[
            task[
                "source_r_A"
            ],
            task[
                "target_r_A"
            ],
        ],

        max_memory_mb=3000,
    )


    propagated = point_at_r(
        points,
        task[
            "target_r_A"
        ],
    )


    if propagated is None:

        print(
            "  -> PROPAGATION_FAILED",
            flush=True,
        )

        propagation_records.append({
            "source_id":
                task[
                    "source_id"
                ],

            "source_r_A":
                task[
                    "source_r_A"
                ],

            "target_r_A":
                task[
                    "target_r_A"
                ],

            "spin":
                source[
                    "spin"
                ],

            "status":
                "PROPAGATION_FAILED",
        })


    else:

        target_groups = [
            group
            for group in groups_by_seed[
                task[
                    "target_r_A"
                ]
            ]
            if int(
                group.spin
            ) == int(
                source[
                    "spin"
                ]
            )
        ]


        relations = []


        for target_group in target_groups:

            target_id = node_id(
                task[
                    "target_r_A"
                ],
                target_group,
            )


            relation = (
                compare_solution_to_point(
                    target_group
                    .representative,

                    propagated,
                )
            )


            record = {
                "source_id":
                    task[
                        "source_id"
                    ],

                "target_id":
                    target_id,

                "source_r_A":
                    task[
                        "source_r_A"
                    ],

                "target_r_A":
                    task[
                        "target_r_A"
                    ],

                "spin":
                    source[
                        "spin"
                    ],

                "relation":
                    relation.value,

                "propagated_energy_hartree":
                    float(
                        propagated
                        .energy_hartree
                    ),

                "target_energy_hartree":
                    float(
                        target_group
                        .representative
                        .energy_hartree
                    ),
            }


            edge_records.append(
                record
            )

            relations.append(
                (
                    target_id,
                    relation.value,
                )
            )


        print(
            "  propagated E={:.12f}".format(
                propagated
                .energy_hartree
            ),
            flush=True,
        )


        for target_id, relation in (
            relations
        ):

            print(
                "  -> {:16s} {}".format(
                    relation,
                    target_id,
                ),
                flush=True,
            )


        propagation_records.append({
            "source_id":
                task[
                    "source_id"
                ],

            "source_r_A":
                task[
                    "source_r_A"
                ],

            "target_r_A":
                task[
                    "target_r_A"
                ],

            "spin":
                source[
                    "spin"
                ],

            "status":
                "PASS",

            "propagated_energy_hartree":
                float(
                    propagated
                    .energy_hartree
                ),

            "n_target_groups":
                len(
                    target_groups
                ),
        })


    completed_tasks.add(
        task_key
    )


    atomic_pickle(
        WORK_PATH,
        {
            "completed_tasks":
                list(
                    completed_tasks
                ),

            "edge_records":
                edge_records,

            "propagation_records":
                propagation_records,
        },
    )


    print(
        "  CHECKPOINT {}/{}".format(
            len(
                completed_tasks
            ),
            len(tasks),
        ),
        flush=True,
    )


# ====================================================================
# Reciprocal SAME_STATE graph.
#
# We merge ONLY if:
#
# A(Ri) -> B(Rj) = SAME_STATE
# AND
# B(Rj) -> A(Ri) = SAME_STATE
#
# One-sided SAME or any AMBIGUOUS relation remains unresolved.
# ====================================================================

same_directed = set()


for record in edge_records:

    if (
        record[
            "relation"
        ]
        == StateRelation
        .SAME_STATE
        .value
    ):

        same_directed.add(
            (
                record[
                    "source_id"
                ],
                record[
                    "target_id"
                ],
            )
        )


reciprocal_same = set()


for a, b in same_directed:

    if (
        b,
        a,
    ) in same_directed:

        reciprocal_same.add(
            tuple(
                sorted(
                    (
                        a,
                        b,
                    )
                )
            )
        )


# ====================================================================
# Union-find confirmed components.
# ====================================================================

parent = {
    nid: nid
    for nid in nodes
}


def find(
    x,
):
    while parent[x] != x:

        parent[x] = (
            parent[
                parent[x]
            ]
        )

        x = parent[x]

    return x


def union(
    a,
    b,
):
    ra = find(a)
    rb = find(b)

    if ra == rb:
        return

    if ra < rb:
        parent[rb] = ra
    else:
        parent[ra] = rb


for a, b in sorted(
    reciprocal_same
):

    union(
        a,
        b,
    )


components = {}


for nid in nodes:

    root = find(
        nid
    )

    components.setdefault(
        root,
        []
    ).append(
        nid
    )


component_list = []


for index, members in enumerate(
    sorted(
        components.values(),
        key=lambda xs: (
            min(
                nodes[x][
                    "spin"
                ]
                for x in xs
            ),
            min(
                nodes[x][
                    "seed_r_A"
                ]
                for x in xs
            ),
            xs,
        ),
    ),
    start=1,
):

    members = sorted(
        members,
        key=lambda nid: (
            nodes[nid][
                "seed_r_A"
            ],
            nodes[nid][
                "energy_hartree"
            ],
        ),
    )


    spins = sorted({
        nodes[nid][
            "spin"
        ]
        for nid in members
    })


    if len(spins) != 1:

        raise RuntimeError(
            "Graph component spans "
            "multiple spin sectors."
        )


    energies = [
        nodes[nid][
            "energy_hartree"
        ]
        for nid in members
    ]


    component_list.append({
        "component_id":
            "SEED_BRANCH_{:03d}".format(
                index
            ),

        "spin":
            spins[0],

        "members":
            members,

        "n_members":
            len(members),

        "seed_r_values":
            sorted({
                nodes[nid][
                    "seed_r_A"
                ]
                for nid in members
            }),

        "minimum_local_energy_hartree":
            min(
                energies
            ),
    })


# ====================================================================
# Ambiguous adjacency audit
# ====================================================================

ambiguous_directed = [
    record
    for record in edge_records
    if (
        record[
            "relation"
        ]
        == StateRelation
        .AMBIGUOUS
        .value
    )
]


failed_propagations = [
    record
    for record
    in propagation_records
    if (
        record[
            "status"
        ]
        != "PASS"
    )
]


print()
print("=" * 132)
print("STAGE A¾ GRAPH RESULT")
print("=" * 132)

print(
    "Stable local nodes:",
    len(nodes),
)

print(
    "Directed nearest-neighbor propagations:",
    len(
        propagation_records
    ),
)

print(
    "Propagation failures:",
    len(
        failed_propagations
    ),
)

print(
    "Directed SAME_STATE matches:",
    len(
        same_directed
    ),
)

print(
    "Reciprocal SAME_STATE links:",
    len(
        reciprocal_same
    ),
)

print(
    "Directed AMBIGUOUS matches:",
    len(
        ambiguous_directed
    ),
)

print(
    "Confirmed seed-branch components:",
    len(
        component_list
    ),
)


print()
print("COMPONENTS")
print("-" * 132)


by_spin = {}


for component in (
    component_list
):

    by_spin.setdefault(
        component[
            "spin"
        ],
        0,
    )

    by_spin[
        component[
            "spin"
        ]
    ] += 1


    print(
        "{}  2S={}  nodes={}  seeds={}  Emin={:.12f}".format(
            component[
                "component_id"
            ],

            component[
                "spin"
            ],

            component[
                "n_members"
            ],

            component[
                "seed_r_values"
            ],

            component[
                "minimum_local_energy_hartree"
            ],
        )
    )

    for member in (
        component[
            "members"
        ]
    ):

        print(
            "    "
            + member
        )


print()
print(
    "Components by spin:",
    by_spin,
)


# ====================================================================
# Save final graph.
# ====================================================================

final_payload = {
    "status":
        "PASS_SEED_BRANCH_GRAPH",

    "nodes":
        nodes,

    "edge_records":
        edge_records,

    "propagation_records":
        propagation_records,

    "reciprocal_same_links":
        sorted(
            reciprocal_same
        ),

    "components":
        component_list,

    "ambiguous_directed_edges":
        ambiguous_directed,

    "failed_propagations":
        failed_propagations,
}


atomic_pickle(
    FINAL_PATH,
    final_payload,
)


summary = {
    "status":
        final_payload[
            "status"
        ],

    "stable_local_nodes":
        len(nodes),

    "directed_propagations":
        len(
            propagation_records
        ),

    "propagation_failures":
        len(
            failed_propagations
        ),

    "directed_same_matches":
        len(
            same_directed
        ),

    "reciprocal_same_links":
        len(
            reciprocal_same
        ),

    "directed_ambiguous_matches":
        len(
            ambiguous_directed
        ),

    "confirmed_components":
        len(
            component_list
        ),

    "components_by_spin":
        by_spin,

    "components":
        component_list,
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
    "Graph checkpoint:",
    FINAL_PATH,
)

print(
    "Summary:",
    SUMMARY_PATH,
)

print("=" * 132)
print("FeH STAGE A¾ COMPLETE")
print("=" * 132)
