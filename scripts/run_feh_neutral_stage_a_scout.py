#!/usr/bin/env python3

import json
import os
import pickle
from pathlib import Path

from diatomic_ea_v09.adaptive_scout import (
    run_adaptive_local_scout,
)
from diatomic_ea_v09.discovery_policy import (
    DiscoveryPolicy,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


OUTDIR = Path(
    "pilot_runs/feh_neutral_checkpoint"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
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
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


policy = DiscoveryPolicy(
    seed_spacing_A=0.20,
    seed_margin_A=0.20,

    initial_spin_sectors=5,
    max_spin_sectors=8,

    spin_tail_length=3,
)


def progress(
    message,
):
    print(
        message,
        flush=True,
    )


print("=" * 132)
print("FeH NEUTRAL — STAGE A: ADAPTIVE SCOUT")
print("=" * 132)
print(flush=True)


result = run_adaptive_local_scout(
    atom="Fe",
    ligand="H",

    charge=0,

    coarse_r_values=
        COARSE_R,

    basis=
        "def2-qzvpd",

    settings=
        settings,

    policy=
        policy,

    max_memory_mb=3000,

    progress_callback=
        progress,
)


print()
print("=" * 132)
print("STAGE A RESULT")
print("=" * 132)

print(
    "Status:",
    result.status,
)

print(
    "Seeds:",
    result.seed_r_values,
)

print(
    "Scanned spins:",
    result.scanned_spin_values,
)

print(
    "Frontier closed:",
    result.frontier.closed,
)

print(
    "Sector records:",
    len(
        result.sector_records
    ),
)

print(
    "CandidateGroups:",
    sum(
        len(groups)
        for groups
        in result.groups_by_seed.values()
    ),
)


if result.status != "PASS_SPIN_FRONTIER":

    raise RuntimeError(
        "Stage A did not close the spin frontier."
    )


# ------------------------------------------------------------
# Atomic pickle checkpoint
# ------------------------------------------------------------

tmp_path = (
    OUTDIR
    / "scout.pkl.tmp"
)

final_path = (
    OUTDIR
    / "scout.pkl"
)


with tmp_path.open(
    "wb"
) as fh:

    pickle.dump(
        result,
        fh,
        protocol=
            pickle.HIGHEST_PROTOCOL,
    )

    fh.flush()
    os.fsync(
        fh.fileno()
    )


os.replace(
    tmp_path,
    final_path,
)


# ------------------------------------------------------------
# Human-readable summary
# ------------------------------------------------------------

summary = {
    "molecule":
        result.molecule,

    "charge":
        result.charge,

    "status":
        result.status,

    "seed_r_values":
        result.seed_r_values,

    "scanned_spin_values":
        result.scanned_spin_values,

    "frontier_closed":
        result.frontier.closed,

    "n_sector_records":
        len(
            result.sector_records
        ),

    "n_candidate_groups":
        sum(
            len(groups)
            for groups
            in result.groups_by_seed.values()
        ),

    "groups_per_seed": {
        str(seed):
            len(groups)

        for seed, groups
        in result.groups_by_seed.items()
    },
}


with (
    OUTDIR
    / "scout_summary.json"
).open(
    "w"
) as fh:

    json.dump(
        summary,
        fh,
        indent=2,
    )


print()
print(
    "CHECKPOINT:",
    final_path,
)

print(
    "CHECKPOINT SIZE [MB]:",
    final_path.stat().st_size
    / 1024**2,
)

print("=" * 132)
print("FeH STAGE A COMPLETE")
print("=" * 132)
