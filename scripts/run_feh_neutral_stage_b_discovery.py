#!/usr/bin/env python3

import json
import os
import pickle
from pathlib import Path

from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.ground_state import (
    DiscoveredBranch,
    SeedDiscoveryRecord,
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

SCOUT_PATH = (
    OUTDIR / "scout.pkl"
)

CHECKPOINT_PATH = (
    OUTDIR / "stage_b_discovery.pkl"
)

SUMMARY_PATH = (
    OUTDIR / "stage_b_summary.json"
)


# Development compute guard only.
# Re-running the script continues from the checkpoint.
MAX_NEW_BRANCHES_THIS_RUN = 20


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


def group_key(
    seed_r,
    group,
):
    return (
        round(
            float(seed_r),
            10,
        ),
        int(group.spin),
        str(
            group.scout_group_id
        ),
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


def write_summary(
    *,
    scout,
    branches,
    records,
    completed_keys,
    all_keys,
    status,
):
    by_spin = {}

    for branch in branches:
        by_spin.setdefault(
            str(branch.spin),
            0,
        )
        by_spin[
            str(branch.spin)
        ] += 1

    payload = {
        "status":
            status,

        "total_scout_groups":
            len(all_keys),

        "completed_scout_groups":
            len(completed_keys),

        "remaining_scout_groups":
            (
                len(all_keys)
                - len(completed_keys)
            ),

        "discovered_branches":
            len(branches),

        "branches_by_spin":
            by_spin,

        "discovery_records":
            len(records),

        "scanned_spins":
            scout.scanned_spin_values,

        "seed_r_values":
            scout.seed_r_values,
    }

    with SUMMARY_PATH.open(
        "w"
    ) as fh:
        json.dump(
            payload,
            fh,
            indent=2,
        )


print("=" * 132)
print("FeH NEUTRAL — STAGE B: RESUMABLE COARSE BRANCH DISCOVERY")
print("=" * 132)
print(flush=True)


if not SCOUT_PATH.exists():
    raise RuntimeError(
        f"Missing Stage-A checkpoint: {SCOUT_PATH}"
    )


with SCOUT_PATH.open(
    "rb"
) as fh:
    scout = pickle.load(
        fh
    )


print(
    "Loaded Stage-A checkpoint:",
    SCOUT_PATH,
    flush=True,
)

print(
    "Scout status:",
    scout.status,
    flush=True,
)

print(
    "Scout groups:",
    sum(
        len(groups)
        for groups
        in scout.groups_by_seed.values()
    ),
    flush=True,
)


if (
    scout.status
    != "PASS_SPIN_FRONTIER"
):
    raise RuntimeError(
        "Stage-A spin frontier did not PASS."
    )


# ====================================================================
# Resume existing Stage-B state or initialize it.
# ====================================================================

if CHECKPOINT_PATH.exists():

    with CHECKPOINT_PATH.open(
        "rb"
    ) as fh:
        state = pickle.load(
            fh
        )

    branches = state[
        "branches"
    ]

    records = state[
        "records"
    ]

    completed_keys = set(
        tuple(x)
        for x
        in state[
            "completed_keys"
        ]
    )

    branch_counter = int(
        state[
            "branch_counter"
        ]
    )

    print()
    print(
        "RESUMING existing Stage-B checkpoint"
    )

    print(
        "Existing branches:",
        len(branches),
    )

    print(
        "Completed scout groups:",
        len(completed_keys),
    )

else:

    branches = []
    records = []
    completed_keys = set()
    branch_counter = 0

    print()
    print(
        "Starting new Stage-B checkpoint."
    )


all_keys = []

for seed_r in scout.seed_r_values:

    seed_key = round(
        float(seed_r),
        10,
    )

    for group in (
        scout.groups_by_seed[
            seed_key
        ]
    ):
        all_keys.append(
            group_key(
                seed_key,
                group,
            )
        )


print(
    "Total Stage-A CandidateGroups:",
    len(all_keys),
)

print(
    "Already completed:",
    len(completed_keys),
)

print(
    "Remaining:",
    (
        len(all_keys)
        - len(completed_keys)
    ),
)

print(
    "New-branch compute guard this run:",
    MAX_NEW_BRANCHES_THIS_RUN,
)

print(flush=True)


new_branches_this_run = 0
stopped_by_guard = False


# ====================================================================
# Existing discovery algorithm, but with checkpointing.
# ====================================================================

for seed_index, seed_r in enumerate(
    scout.seed_r_values,
    start=1,
):

    seed_key = round(
        float(seed_r),
        10,
    )

    groups = (
        scout.groups_by_seed[
            seed_key
        ]
    )

    print()
    print("#" * 132)

    print(
        "SEED {}/{}  R={:.2f} A  groups={}".format(
            seed_index,
            len(
                scout.seed_r_values
            ),
            seed_key,
            len(groups),
        ),
        flush=True,
    )

    print("#" * 132)


    for group_index, group in enumerate(
        groups,
        start=1,
    ):

        key = group_key(
            seed_key,
            group,
        )


        if key in completed_keys:

            print(
                "SKIP completed "
                "R={:.2f} 2S={} {}"
                .format(
                    seed_key,
                    group.spin,
                    group.scout_group_id,
                ),
                flush=True,
            )

            continue


        representative = (
            group.representative
        )


        print()
        print(
            "GROUP {}/{} "
            "R={:.2f} "
            "2S={} "
            "{} "
            "E={:.12f}".format(
                group_index,
                len(groups),

                seed_key,
                representative.spin,

                group.scout_group_id,

                representative
                .energy_hartree,
            ),
            flush=True,
        )


        same_matches = []


        for existing in branches:

            existing_point = (
                point_at_r(
                    existing.raw_points,
                    seed_key,
                )
            )

            if existing_point is None:
                continue


            relation = (
                compare_solution_to_point(
                    representative,
                    existing_point,
                )
            )


            if (
                relation
                == StateRelation.SAME_STATE
            ):
                same_matches.append(
                    existing.branch_id
                )


        if same_matches:

            matched = (
                same_matches[0]
            )

            print(
                "  -> MATCH_EXISTING",
                matched,
                flush=True,
            )


            records.append(
                SeedDiscoveryRecord(
                    seed_r_A=
                        seed_key,

                    spin=
                        representative.spin,

                    scout_group_id=
                        group.scout_group_id,

                    guesses=
                        group.equivalent_guesses,

                    energy_hartree=
                        representative
                        .energy_hartree,

                    action=
                        "MATCH_EXISTING",

                    matched_branch_id=
                        matched,
                )
            )


            completed_keys.add(
                key
            )

            continue


        # ------------------------------------------------------------
        # A genuinely new branch would trigger a full 21-point PEC.
        # Guard BEFORE starting another expensive branch.
        # ------------------------------------------------------------

        if (
            new_branches_this_run
            >= MAX_NEW_BRANCHES_THIS_RUN
        ):

            print()
            print(
                "DEVELOPMENT BRANCH GUARD REACHED.",
                flush=True,
            )

            print(
                "No scientific truncation has been made; "
                "rerun Stage B to continue.",
                flush=True,
            )

            stopped_by_guard = True
            break


        branch_counter += 1

        branch_id = (
            "FEH_N_"
            f"DISC{branch_counter:03d}_"
            f"S{representative.spin}"
        )


        print(
            "  -> NEW_BRANCH_BEGIN",
            branch_id,
            flush=True,
        )

        print(
            "     propagating 21-point coarse PEC...",
            flush=True,
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

            r_values=
                COARSE_R,

            max_memory_mb=3000,
        )


        converged = sum(
            point.converged
            for point in points
        )


        branch = DiscoveredBranch(
            branch_id=
                branch_id,

            charge=0,

            spin=
                representative.spin,

            multiplicity=
                representative
                .multiplicity,

            source_seed_r_A=
                seed_key,

            source_scout_group_id=
                group.scout_group_id,

            source_guesses=
                group.equivalent_guesses,

            seed_solution=
                representative,

            raw_points=
                points,
        )


        branches.append(
            branch
        )


        records.append(
            SeedDiscoveryRecord(
                seed_r_A=
                    seed_key,

                spin=
                    representative.spin,

                scout_group_id=
                    group.scout_group_id,

                guesses=
                    group.equivalent_guesses,

                energy_hartree=
                    representative
                    .energy_hartree,

                action=
                    "NEW_BRANCH",

                matched_branch_id=None,
            )
        )


        completed_keys.add(
            key
        )

        new_branches_this_run += 1


        minimum = (
            branch.raw_minimum
        )


        print(
            "  -> NEW_BRANCH_DONE "
            "{} converged={}/{} "
            "Rmin={:.3f} "
            "Emin={:.12f}".format(
                branch_id,
                converged,
                len(points),

                minimum.r_A,
                minimum.energy_hartree,
            ),
            flush=True,
        )


        # ------------------------------------------------------------
        # Checkpoint immediately after every expensive completed branch.
        # ------------------------------------------------------------

        state = {
            "branches":
                branches,

            "records":
                records,

            "completed_keys":
                list(
                    completed_keys
                ),

            "branch_counter":
                branch_counter,
        }


        atomic_pickle(
            state
        )


        write_summary(
            scout=scout,
            branches=branches,
            records=records,
            completed_keys=
                completed_keys,
            all_keys=all_keys,
            status=
                "IN_PROGRESS",
        )


        print(
            "     CHECKPOINT SAVED "
            "branches={} "
            "completed={}/{}".format(
                len(branches),
                len(completed_keys),
                len(all_keys),
            ),
            flush=True,
        )


    if stopped_by_guard:
        break


    # Save inexpensive MATCH_EXISTING progress at end of every seed.
    state = {
        "branches":
            branches,

        "records":
            records,

        "completed_keys":
            list(
                completed_keys
            ),

        "branch_counter":
            branch_counter,
    }

    atomic_pickle(
        state
    )


# ====================================================================
# Final status
# ====================================================================

complete = (
    len(completed_keys)
    == len(all_keys)
)


if complete:

    status = (
        "PASS_DISCOVERY_COMPLETE"
    )

elif stopped_by_guard:

    status = (
        "PAUSED_BRANCH_GUARD"
    )

else:

    status = (
        "INCOMPLETE"
    )


write_summary(
    scout=scout,
    branches=branches,
    records=records,
    completed_keys=
        completed_keys,
    all_keys=all_keys,
    status=status,
)


# Always persist final in-memory state.
atomic_pickle({
    "branches":
        branches,

    "records":
        records,

    "completed_keys":
        list(
            completed_keys
        ),

    "branch_counter":
        branch_counter,
})


print()
print("=" * 132)
print("STAGE B RESULT")
print("=" * 132)

print(
    "Status:",
    status,
)

print(
    "Branches total:",
    len(branches),
)

print(
    "New branches this run:",
    new_branches_this_run,
)

print(
    "Completed scout groups:",
    "{}/{}".format(
        len(completed_keys),
        len(all_keys),
    ),
)

print(
    "Remaining scout groups:",
    (
        len(all_keys)
        - len(completed_keys)
    ),
)


by_spin = {}

for branch in branches:
    by_spin.setdefault(
        branch.spin,
        0,
    )
    by_spin[
        branch.spin
    ] += 1


print(
    "Branches by spin:",
    by_spin,
)


print(
    "Checkpoint:",
    CHECKPOINT_PATH,
)

print(
    "Checkpoint size [MB]: "
    "{:.2f}".format(
        CHECKPOINT_PATH
        .stat()
        .st_size
        / 1024**2
    )
)

print("=" * 132)
print("FeH STAGE B EXIT")
print("=" * 132)
