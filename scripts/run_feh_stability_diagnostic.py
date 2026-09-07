#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np

from diatomic_ea_v09.coarse_branch import (
    follow_scout_solution,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.molecule import (
    build_molecule,
)
from diatomic_ea_v09.scf import (
    run_uks,
)
from diatomic_ea_v09.stability import (
    optimize_internal_stability,
)
from diatomic_ea_v09.state_identity import (
    compare_states,
)
from diatomic_ea_v09.state_scout import (
    orthonormal_spin_density,
    run_guess,
)


REFDIR = Path(
    "pilot_runs/coarse_branch_feh"
)

R_TEST = 1.55


settings = SCFSettings(
    xc="PBE",
    grid_level=3,
    conv_tol=1.0e-9,
    max_cycle=200,
    threads=1,
    level_shift_helper=0.25,
)


def spec_at(r):

    return MoleculeSpec(
        atom="Fe",
        ligand="H",
        charge=0,
        spin=3,
        basis="def2-qzvpd",
        r_A=float(r),
        max_memory_mb=2000,
    )


def load_branch_mf(
    branch_id,
    r_target,
):
    archive = np.load(
        REFDIR
        / f"{branch_id}_densities.npz"
    )

    r_values = np.asarray(
        archive["r_A"],
        dtype=float,
    )

    idx = np.where(
        np.isclose(
            r_values,
            r_target,
            atol=1e-8,
            rtol=0.0,
        )
    )[0]

    if len(idx) != 1:
        raise RuntimeError(
            f"{branch_id}: density at "
            f"R={r_target} not uniquely found"
        )

    dm0 = np.array(
        archive["density_ao"][
            int(idx[0])
        ],
        copy=True,
    )

    mol = build_molecule(
        spec_at(r_target)
    )

    outcome = run_uks(
        mol,
        settings,
        dm0=dm0,
    )

    if not outcome.converged:
        raise RuntimeError(
            f"{branch_id}: reconstruction "
            "did not converge"
        )

    return outcome.mf


def build_parallel_branch_mf():

    seed = run_guess(
        spec_at(1.40),
        settings,
        "minao",
    )

    if seed is None:
        raise RuntimeError(
            "R=1.40 minao seed failed"
        )

    points = follow_scout_solution(
        branch_id=
            "FEH_N_PARALLEL_MINAO",
        atom="Fe",
        ligand="H",
        seed=seed,
        settings=settings,
        r_values=[
            1.40,
            1.45,
            1.50,
            1.54,
            1.55,
        ],
        max_memory_mb=2000,
    )

    matches = [
        p
        for p in points
        if (
            p.converged
            and abs(
                p.r_A - R_TEST
            ) < 1e-8
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Parallel branch did not reach "
            f"R={R_TEST}"
        )

    mol = build_molecule(
        spec_at(R_TEST)
    )

    outcome = run_uks(
        mol,
        settings,
        dm0=matches[0].density_ao,
    )

    if not outcome.converged:
        raise RuntimeError(
            "Parallel branch reconstruction "
            "failed"
        )

    return outcome.mf


def descriptor(mf):

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
        "mult":
            float(mult),
        "density":
            dm_orth,
    }


print("=" * 120)
print("FeH INTERNAL-STABILITY DIAGNOSTIC")
print("PBE / def2-QZVPD / neutral quartet / R=1.55 A")
print("=" * 120)


initial_mfs = {
    "G01":
        load_branch_mf(
            "FEH_N_S3_G01",
            R_TEST,
        ),

    "G02":
        load_branch_mf(
            "FEH_N_S3_G02",
            R_TEST,
        ),

    "PARALLEL_MINAO":
        build_parallel_branch_mf(),
}


results = {}


for label, mf in initial_mfs.items():

    before = descriptor(
        mf
    )

    print()
    print("#" * 120)
    print(label)
    print("#" * 120)

    print(
        "Initial: "
        "E={:.12f} Eh "
        "<S2>={:.8f}".format(
            before["energy"],
            before["s2"],
        )
    )

    result = (
        optimize_internal_stability(
            mf,
            settings,
            max_iterations=6,
        )
    )

    after = descriptor(
        result.final_mf
    )

    print(
        "Initially stable:",
        result.initially_stable,
    )

    print(
        "Finally stable:",
        result.finally_stable,
    )

    print(
        "Iterations:",
        result.iterations,
    )

    print(
        "Total stability Delta E [meV]: "
        "{:+.6f}".format(
            result.total_delta_energy_meV
        )
    )

    for step in result.history:

        print(
            "  iteration {:d}: "
            "stable={} "
            "E_before={:.12f} "
            "E_after={} "
            "dE_meV={} "
            "path={}".format(
                step.iteration,
                step.stable_internal,
                step.energy_before_hartree,
                (
                    "-"
                    if step.energy_after_hartree
                    is None
                    else "{:.12f}".format(
                        step.energy_after_hartree
                    )
                ),
                (
                    "-"
                    if step.delta_energy_meV
                    is None
                    else "{:+.6f}".format(
                        step.delta_energy_meV
                    )
                ),
                step.scf_path,
            )
        )

    comparison = compare_states(
        energy_a_hartree=
            before["energy"],
        energy_b_hartree=
            after["energy"],

        s2_a=
            before["s2"],
        s2_b=
            after["s2"],

        dm_orth_a=
            before["density"],
        dm_orth_b=
            after["density"],

        spin_a=3,
        spin_b=3,
    )

    print(
        "Initial -> stabilized identity: "
        "{}  "
        "dE={:.6f} meV "
        "dTot={:.3e} "
        "dSpin={:.3e}".format(
            comparison.relation.value,
            comparison.delta_energy_meV,
            comparison.total_spectrum_max,
            comparison.spin_spectrum_max,
        )
    )

    results[label] = {
        "before": before,
        "after": after,
        "stability": result,
    }


print()
print("=" * 120)
print("STABILIZED SOLUTIONS: PAIRWISE")
print("=" * 120)


labels = list(
    results.keys()
)

for i, a in enumerate(labels):

    for b in labels[i + 1:]:

        A = results[a]["after"]
        B = results[b]["after"]

        comparison = compare_states(
            energy_a_hartree=
                A["energy"],
            energy_b_hartree=
                B["energy"],

            s2_a=
                A["s2"],
            s2_b=
                B["s2"],

            dm_orth_a=
                A["density"],
            dm_orth_b=
                B["density"],

            spin_a=3,
            spin_b=3,
        )

        print(
            "{} vs {}: "
            "dE={:.6f} meV  "
            "dS2={:.6f}  "
            "dTot={:.3e}  "
            "dSpin={:.3e}  "
            "{}".format(
                a,
                b,
                comparison.delta_energy_meV,
                comparison.delta_s2,
                comparison.total_spectrum_max,
                comparison.spin_spectrum_max,
                comparison.relation.value,
            )
        )


print()
print("=" * 120)
print("STABILITY DIAGNOSTIC COMPLETE")
print("=" * 120)

