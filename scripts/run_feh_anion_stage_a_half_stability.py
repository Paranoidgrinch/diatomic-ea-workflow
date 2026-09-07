#!/usr/bin/env python3

import json
import os
import pickle
from pathlib import Path

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
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
from diatomic_ea_v09.state_identity import (
    compare_states,
)


OUTDIR = Path(
    "pilot_runs/feh_anion_checkpoint"
)

RAW_SCOUT_PATH = (
    OUTDIR / "scout.pkl"
)

WORK_PATH = (
    OUTDIR / "stage_a_half_work.pkl"
)

FINAL_PATH = (
    OUTDIR / "stable_scout.pkl"
)

SUMMARY_PATH = (
    OUTDIR / "stable_scout_summary.json"
)


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def raw_key(
    seed_r,
    group,
):
    return (
        round(
            float(seed_r),
            10,
        ),
        int(group.spin),
        str(group.scout_group_id),
    )


def key_text(
    key,
):
    r_A, spin, group_id = key

    return (
        "R{:.2f}_S{}_{}".format(
            r_A,
            spin,
            group_id,
        )
    )


def atomic_pickle(
    path,
    payload,
):
    tmp = Path(
        str(path)
        + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as fh:

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
        path,
    )


print("=" * 132)
print("FeH ANION — STAGE A½: LOCAL STABILITY CANONICALIZATION")
print("=" * 132)
print(flush=True)


if not RAW_SCOUT_PATH.exists():

    raise RuntimeError(
        "Missing Stage-A checkpoint: "
        + str(RAW_SCOUT_PATH)
    )


with RAW_SCOUT_PATH.open(
    "rb"
) as fh:

    raw_scout = pickle.load(
        fh
    )


print(
    "Loaded Stage-A checkpoint:",
    RAW_SCOUT_PATH,
)

print(
    "Raw scout status:",
    raw_scout.status,
)

print(
    "Raw CandidateGroups:",
    sum(
        len(groups)
        for groups
        in raw_scout.groups_by_seed.values()
    ),
)

print(flush=True)


if (
    raw_scout.status
    != "PASS_SPIN_FRONTIER"
):

    raise RuntimeError(
        "Stage A did not PASS spin frontier."
    )


# ====================================================================
# Resume or initialize
# ====================================================================

if WORK_PATH.exists():

    with WORK_PATH.open(
        "rb"
    ) as fh:

        work = pickle.load(
            fh
        )


    completed_keys = set(
        tuple(x)
        for x
        in work[
            "completed_keys"
        ]
    )

    stabilized_by_key = dict(
        work[
            "stabilized_by_key"
        ]
    )

    records = list(
        work[
            "records"
        ]
    )


    print(
        "RESUMING Stage A½"
    )

    print(
        "Completed raw groups:",
        len(completed_keys),
    )

else:

    completed_keys = set()

    stabilized_by_key = {}

    records = []


    print(
        "Starting new Stage A½ checkpoint."
    )


all_items = []

for seed_r in (
    raw_scout.seed_r_values
):

    seed_key = round(
        float(seed_r),
        10,
    )

    for group in (
        raw_scout
        .groups_by_seed[
            seed_key
        ]
    ):

        all_items.append(
            (
                seed_key,
                group,
            )
        )


print(
    "Total raw groups:",
    len(all_items),
)

print(
    "Already completed:",
    len(completed_keys),
)

print(
    "Remaining:",
    len(all_items)
    - len(completed_keys),
)

print(flush=True)


# ====================================================================
# Stability-canonicalize every RAW LOCAL CandidateGroup representative.
# ====================================================================

for item_index, (
    seed_r,
    group,
) in enumerate(
    all_items,
    start=1,
):

    key = raw_key(
        seed_r,
        group,
    )


    if key in completed_keys:

        print(
            "SKIP {}/{} {}".format(
                item_index,
                len(all_items),
                key_text(key),
            ),
            flush=True,
        )

        continue


    representative = (
        group.representative
    )


    print()
    print("-" * 132)

    print(
        "RAW {}/{}  "
        "R={:.2f}  "
        "2S={}  "
        "{}  "
        "E={:.12f}".format(
            item_index,
            len(all_items),

            seed_r,
            representative.spin,

            group.scout_group_id,

            representative
            .energy_hartree,
        ),
        flush=True,
    )


    # ----------------------------------------------------------------
    # Reconstruct SCF object from the persisted AO density.
    # ----------------------------------------------------------------

    spec = MoleculeSpec(
        atom="Fe",
        ligand="H",

        charge=-1,

        spin=
            representative.spin,

        basis=
            "def2-qzvpd",

        r_A=
            seed_r,

        max_memory_mb=3000,
    )


    mol = build_molecule(
        spec
    )


    reconstruction = run_uks(
        mol,
        settings,

        dm0=
            representative
            .density_ao,
    )


    if not reconstruction.converged:

        print(
            "  -> QC_FAIL_SCF_RECONSTRUCTION",
            flush=True,
        )

        stabilized_by_key[
            key
        ] = None


        records.append({
            "key":
                list(key),

            "status":
                "QC_FAIL_SCF_RECONSTRUCTION",

            "raw_energy_hartree":
                representative.energy_hartree,
        })


        completed_keys.add(
            key
        )


        atomic_pickle(
            WORK_PATH,
            {
                "completed_keys":
                    list(
                        completed_keys
                    ),

                "stabilized_by_key":
                    stabilized_by_key,

                "records":
                    records,
            },
        )

        continue


    reconstructed = (
        scout_solution_from_mf(
            mf=
                reconstruction.mf,

            template=
                representative,

            origin_guess=
                "seed_reconstruction",
        )
    )


    reconstruction_cmp = (
        compare_states(
            energy_a_hartree=
                representative
                .energy_hartree,

            energy_b_hartree=
                reconstructed
                .energy_hartree,

            s2_a=
                representative.s2,

            s2_b=
                reconstructed.s2,

            dm_orth_a=
                representative
                .density_orth,

            dm_orth_b=
                reconstructed
                .density_orth,

            spin_a=
                representative.spin,

            spin_b=
                reconstructed.spin,
        )
    )


    reconstruction_delta_meV = (
        (
            reconstructed.energy_hartree
            - representative.energy_hartree
        )
        * HARTREE_TO_EV
        * 1000.0
    )


    print(
        "  reconstruction: "
        "{}  dE={:+.6f} meV".format(
            reconstruction_cmp
            .relation
            .value,

            reconstruction_delta_meV,
        ),
        flush=True,
    )


    # ----------------------------------------------------------------
    # Internal orbital stability.
    # ----------------------------------------------------------------

    stability = (
        optimize_internal_stability(
            reconstruction.mf,
            settings,

            max_iterations=6,
        )
    )


    print(
        "  stability: "
        "initial={} "
        "final={} "
        "reopts={} "
        "dE={:+.6f} meV".format(
            stability.initially_stable,
            stability.finally_stable,
            stability.reoptimizations,
            stability.total_delta_energy_meV,
        ),
        flush=True,
    )


    if not stability.finally_stable:

        print(
            "  -> QC_FAIL_STABILITY",
            flush=True,
        )

        stabilized_by_key[
            key
        ] = None


        records.append({
            "key":
                list(key),

            "status":
                "QC_FAIL_STABILITY",

            "raw_energy_hartree":
                representative
                .energy_hartree,

            "reconstruction_relation":
                reconstruction_cmp
                .relation
                .value,

            "reconstruction_delta_meV":
                reconstruction_delta_meV,

            "initially_stable":
                stability.initially_stable,

            "finally_stable":
                stability.finally_stable,

            "stability_reoptimizations":
                stability.reoptimizations,

            "stability_delta_meV":
                stability.total_delta_energy_meV,
        })


        completed_keys.add(
            key
        )


        atomic_pickle(
            WORK_PATH,
            {
                "completed_keys":
                    list(
                        completed_keys
                    ),

                "stabilized_by_key":
                    stabilized_by_key,

                "records":
                    records,
            },
        )

        continue


    stabilized = (
        scout_solution_from_mf(
            mf=
                stability.final_mf,

            template=
                reconstructed,

            origin_guess=
                "stability_canonical",
        )
    )


    stability_cmp = (
        compare_states(
            energy_a_hartree=
                reconstructed
                .energy_hartree,

            energy_b_hartree=
                stabilized
                .energy_hartree,

            s2_a=
                reconstructed.s2,

            s2_b=
                stabilized.s2,

            dm_orth_a=
                reconstructed
                .density_orth,

            dm_orth_b=
                stabilized
                .density_orth,

            spin_a=
                reconstructed.spin,

            spin_b=
                stabilized.spin,
        )
    )


    # Preserve raw multiguess provenance.
    stabilized.equivalent_guesses = (
        list(
            group.equivalent_guesses
        )
    )

    stabilized.candidate_id = (
        key_text(
            key
        )
    )


    # We only need the density/descriptors downstream.
    # Do not persist the PySCF object.
    stabilized.mf = None


    stabilized_by_key[
        key
    ] = stabilized


    records.append({
        "key":
            list(key),

        "status":
            "PASS",

        "raw_energy_hartree":
            representative
            .energy_hartree,

        "reconstructed_energy_hartree":
            reconstructed
            .energy_hartree,

        "stabilized_energy_hartree":
            stabilized
            .energy_hartree,

        "reconstruction_relation":
            reconstruction_cmp
            .relation
            .value,

        "reconstruction_delta_meV":
            reconstruction_delta_meV,

        "initially_stable":
            stability.initially_stable,

        "finally_stable":
            stability.finally_stable,

        "stability_reoptimizations":
            stability.reoptimizations,

        "stability_delta_meV":
            stability.total_delta_energy_meV,

        "stability_relation":
            stability_cmp
            .relation
            .value,
    })


    print(
        "  final relation:",
        stability_cmp
        .relation
        .value,
        flush=True,
    )

    print(
        "  stabilized E={:.12f}".format(
            stabilized
            .energy_hartree
        ),
        flush=True,
    )


    completed_keys.add(
        key
    )


    atomic_pickle(
        WORK_PATH,
        {
            "completed_keys":
                list(
                    completed_keys
                ),

            "stabilized_by_key":
                stabilized_by_key,

            "records":
                records,
        },
    )


    print(
        "  CHECKPOINT {}/{}".format(
            len(completed_keys),
            len(all_items),
        ),
        flush=True,
    )


# ====================================================================
# Re-group all stable solutions locally at every seed.
# ====================================================================

stable_groups_by_seed = {}

raw_to_stable_group = {}


for seed_r in (
    raw_scout.seed_r_values
):

    seed_key = round(
        float(seed_r),
        10,
    )


    stable_solutions = []


    for group in (
        raw_scout
        .groups_by_seed[
            seed_key
        ]
    ):

        key = raw_key(
            seed_key,
            group,
        )

        solution = (
            stabilized_by_key
            .get(key)
        )


        if solution is not None:

            stable_solutions.append(
                solution
            )


    stable_groups = (
        build_candidate_groups(
            stable_solutions
        )
        if stable_solutions
        else []
    )


    stable_groups_by_seed[
        seed_key
    ] = stable_groups


    for stable_group in (
        stable_groups
    ):

        for member in (
            stable_group.members
        ):

            raw_to_stable_group[
                member.candidate_id
            ] = (
                stable_group
                .scout_group_id
            )


# ====================================================================
# Human-readable reduction statistics.
# ====================================================================

n_raw = (
    len(all_items)
)

n_pass = sum(
    record[
        "status"
    ] == "PASS"
    for record in records
)

n_reconstruction_fail = sum(
    record[
        "status"
    ]
    == "QC_FAIL_SCF_RECONSTRUCTION"
    for record in records
)

n_stability_fail = sum(
    record[
        "status"
    ]
    == "QC_FAIL_STABILITY"
    for record in records
)

n_stable_groups = sum(
    len(groups)
    for groups
    in stable_groups_by_seed.values()
)


initially_unstable = sum(
    (
        record[
            "status"
        ] == "PASS"
        and not record[
            "initially_stable"
        ]
    )
    for record in records
)


stability_changed_state = sum(
    (
        record[
            "status"
        ] == "PASS"
        and record[
            "stability_relation"
        ] != "SAME_STATE"
    )
    for record in records
)


print()
print("=" * 132)
print("STAGE A½ REDUCTION")
print("=" * 132)

print(
    "Raw CandidateGroups:",
    n_raw,
)

print(
    "Successfully stable raw representatives:",
    n_pass,
)

print(
    "SCF reconstruction failures:",
    n_reconstruction_fail,
)

print(
    "Final stability failures:",
    n_stability_fail,
)

print(
    "Initially unstable representatives:",
    initially_unstable,
)

print(
    "Stability relation != SAME_STATE:",
    stability_changed_state,
)

print(
    "Canonical local groups after Stability + dedup:",
    n_stable_groups,
)

print(
    "Reduction: {} -> {}".format(
        n_raw,
        n_stable_groups,
    )
)


print()
print("GROUP COUNTS BY SEED")
print("-" * 132)


counts_by_seed = {}


for seed_r in (
    raw_scout.seed_r_values
):

    seed_key = round(
        float(seed_r),
        10,
    )

    raw_groups = (
        raw_scout
        .groups_by_seed[
            seed_key
        ]
    )

    stable_groups = (
        stable_groups_by_seed[
            seed_key
        ]
    )


    by_spin_raw = {}
    by_spin_stable = {}


    for group in raw_groups:

        by_spin_raw[
            group.spin
        ] = (
            by_spin_raw
            .get(
                group.spin,
                0,
            )
            + 1
        )


    for group in stable_groups:

        by_spin_stable[
            group.spin
        ] = (
            by_spin_stable
            .get(
                group.spin,
                0,
            )
            + 1
        )


    counts_by_seed[
        str(seed_key)
    ] = {
        "raw":
            len(raw_groups),

        "stable":
            len(stable_groups),

        "raw_by_spin":
            by_spin_raw,

        "stable_by_spin":
            by_spin_stable,
    }


    print(
        "R={:.2f}: "
        "{} -> {}   "
        "raw={}   stable={}".format(
            seed_key,
            len(raw_groups),
            len(stable_groups),
            by_spin_raw,
            by_spin_stable,
        )
    )


# ====================================================================
# Final persistent Stage-A½ result.
# ====================================================================

final_payload = {
    "molecule":
        raw_scout.molecule,

    "charge":
        raw_scout.charge,

    "status":
        "PASS_STABLE_SCOUT_CANONICALIZATION",

    "seed_r_values":
        list(
            raw_scout.seed_r_values
        ),

    "scanned_spin_values":
        list(
            raw_scout
            .scanned_spin_values
        ),

    "stable_groups_by_seed":
        stable_groups_by_seed,

    "records":
        records,

    "raw_to_stable_group":
        raw_to_stable_group,
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

    "raw_candidate_groups":
        n_raw,

    "stable_raw_representatives":
        n_pass,

    "scf_reconstruction_failures":
        n_reconstruction_fail,

    "stability_failures":
        n_stability_fail,

    "initially_unstable":
        initially_unstable,

    "stability_relation_not_same":
        stability_changed_state,

    "canonical_local_groups":
        n_stable_groups,

    "counts_by_seed":
        counts_by_seed,

    "records":
        records,

    "raw_to_stable_group":
        raw_to_stable_group,
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
    "Stable scout checkpoint:",
    FINAL_PATH,
)

print(
    "Checkpoint size [MB]: "
    "{:.2f}".format(
        FINAL_PATH
        .stat()
        .st_size
        / 1024**2
    )
)

print(
    "Summary:",
    SUMMARY_PATH,
)

print("=" * 132)
print("FeH ANION STAGE A½ COMPLETE")
print("=" * 132)
