from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .model import (
    MoleculeSpec,
    SCFSettings,
)
from .molecule import (
    build_molecule,
)
from .scf import (
    HARTREE_TO_EV,
    run_uks,
)
from .stability import (
    optimize_internal_stability,
)
from .state_identity import (
    StateRelation,
    compare_states,
)
from .state_scout import (
    orthonormal_spin_density,
)


@dataclass
class FinePointQC:
    r_A: float

    status: str

    stored_energy_hartree: float | None
    reconstructed_energy_hartree: float | None
    stabilized_energy_hartree: float | None

    reconstruction_delta_meV: float | None
    reconstruction_relation: StateRelation | None

    initially_stable: bool | None
    finally_stable: bool | None

    stability_reoptimizations: int | None
    stability_delta_energy_meV: float | None
    stability_relation: StateRelation | None


@dataclass
class FinePECQCResult:
    points: List[FinePointQC]

    status: str

    n_points: int
    n_pass: int

    n_reconstruction_fail: int
    n_identity_fail: int
    n_initially_unstable: int
    n_finally_unstable: int

    n_stability_same_state: int
    n_stability_ambiguous: int
    n_stability_state_change: int

    max_abs_reconstruction_delta_meV: float | None
    most_negative_stability_delta_meV: float | None


def _descriptor(
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

        "density_orth":
            dm_orth,
    }


def qc_fine_point(
    *,
    point,

    atom: str,
    ligand: str,
    charge: int,
    spin: int,

    basis: str,
    settings: SCFSettings,

    max_memory_mb: int = 2000,
    max_stability_iterations: int = 6,
):
    """
    Pointwise QC for an already state-followed ground-state Fine PEC.

    This diagnostic does NOT silently repair the PEC.

    It asks:
      1. Can the stored point be reconstructed from its density?
      2. Is the reconstructed solution electronically the SAME_STATE?
      3. Is that solution internally stable?
      4. If stability reoptimization occurs, does it remain the
         same state or change root?
    """

    if (
        not point.converged
        or point.energy_hartree is None
        or point.density_ao is None
        or point.density_orth is None
    ):

        return FinePointQC(
            r_A=float(
                point.r_A
            ),

            status=
                "QC_FAIL_SCF_RECONSTRUCTION",

            stored_energy_hartree=
                point.energy_hartree,

            reconstructed_energy_hartree=None,
            stabilized_energy_hartree=None,

            reconstruction_delta_meV=None,
            reconstruction_relation=None,

            initially_stable=None,
            finally_stable=None,

            stability_reoptimizations=None,
            stability_delta_energy_meV=None,
            stability_relation=None,
        )


    spec = MoleculeSpec(
        atom=atom,
        ligand=ligand,

        charge=charge,
        spin=spin,

        basis=basis,

        r_A=float(
            point.r_A
        ),

        max_memory_mb=
            max_memory_mb,
    )

    mol = build_molecule(
        spec
    )

    reconstructed = run_uks(
        mol,
        settings,

        dm0=np.asarray(
            point.density_ao
        ),
    )

    if not reconstructed.converged:

        return FinePointQC(
            r_A=float(
                point.r_A
            ),

            status=
                "QC_FAIL_SCF_RECONSTRUCTION",

            stored_energy_hartree=
                point.energy_hartree,

            reconstructed_energy_hartree=None,
            stabilized_energy_hartree=None,

            reconstruction_delta_meV=None,
            reconstruction_relation=None,

            initially_stable=None,
            finally_stable=None,

            stability_reoptimizations=None,
            stability_delta_energy_meV=None,
            stability_relation=None,
        )


    reconstructed_desc = (
        _descriptor(
            reconstructed.mf
        )
    )

    reconstruction_cmp = (
        compare_states(
            energy_a_hartree=
                point.energy_hartree,

            energy_b_hartree=
                reconstructed_desc[
                    "energy"
                ],

            s2_a=
                point.s2,

            s2_b=
                reconstructed_desc[
                    "s2"
                ],

            dm_orth_a=
                point.density_orth,

            dm_orth_b=
                reconstructed_desc[
                    "density_orth"
                ],

            spin_a=
                spin,

            spin_b=
                spin,
        )
    )


    reconstruction_delta_meV = (
        reconstructed_desc[
            "energy"
        ]
        - point.energy_hartree
    ) * HARTREE_TO_EV * 1000.0


    if (
        reconstruction_cmp.relation
        != StateRelation.SAME_STATE
    ):

        return FinePointQC(
            r_A=float(
                point.r_A
            ),

            status=
                "QC_FAIL_STATE_IDENTITY",

            stored_energy_hartree=
                point.energy_hartree,

            reconstructed_energy_hartree=
                reconstructed_desc[
                    "energy"
                ],

            stabilized_energy_hartree=None,

            reconstruction_delta_meV=
                float(
                    reconstruction_delta_meV
                ),

            reconstruction_relation=
                reconstruction_cmp.relation,

            initially_stable=None,
            finally_stable=None,

            stability_reoptimizations=None,
            stability_delta_energy_meV=None,
            stability_relation=None,
        )


    stability = (
        optimize_internal_stability(
            reconstructed.mf,

            settings,

            max_iterations=
                max_stability_iterations,
        )
    )


    stabilized_desc = (
        _descriptor(
            stability.final_mf
        )
    )


    stability_cmp = (
        compare_states(
            energy_a_hartree=
                reconstructed_desc[
                    "energy"
                ],

            energy_b_hartree=
                stabilized_desc[
                    "energy"
                ],

            s2_a=
                reconstructed_desc[
                    "s2"
                ],

            s2_b=
                stabilized_desc[
                    "s2"
                ],

            dm_orth_a=
                reconstructed_desc[
                    "density_orth"
                ],

            dm_orth_b=
                stabilized_desc[
                    "density_orth"
                ],

            spin_a=
                spin,

            spin_b=
                spin,
        )
    )


    if not stability.finally_stable:

        status = (
            "QC_FAIL_STABILITY"
        )

    elif stability.initially_stable:

        status = "PASS"

    elif (
        stability_cmp.relation
        == StateRelation.SAME_STATE
    ):

        status = (
            "STABILITY_REOPT_SAME_STATE"
        )

    elif (
        stability_cmp.relation
        == StateRelation.AMBIGUOUS
    ):

        status = (
            "STABILITY_REOPT_AMBIGUOUS"
        )

    else:

        status = (
            "STABILITY_ROOT_CHANGE"
        )


    return FinePointQC(
        r_A=float(
            point.r_A
        ),

        status=status,

        stored_energy_hartree=
            float(
                point.energy_hartree
            ),

        reconstructed_energy_hartree=
            reconstructed_desc[
                "energy"
            ],

        stabilized_energy_hartree=
            stabilized_desc[
                "energy"
            ],

        reconstruction_delta_meV=
            float(
                reconstruction_delta_meV
            ),

        reconstruction_relation=
            reconstruction_cmp.relation,

        initially_stable=
            stability.initially_stable,

        finally_stable=
            stability.finally_stable,

        stability_reoptimizations=
            stability.reoptimizations,

        stability_delta_energy_meV=
            stability.total_delta_energy_meV,

        stability_relation=
            stability_cmp.relation,
    )


def qc_fine_pec(
    *,
    points: Sequence,

    atom: str,
    ligand: str,
    charge: int,
    spin: int,

    basis: str,
    settings: SCFSettings,

    max_memory_mb: int = 2000,
    max_stability_iterations: int = 6,
):
    records = []

    for point in sorted(
        points,
        key=lambda p:
            p.r_A,
    ):

        records.append(
            qc_fine_point(
                point=point,

                atom=atom,
                ligand=ligand,

                charge=charge,
                spin=spin,

                basis=basis,
                settings=settings,

                max_memory_mb=
                    max_memory_mb,

                max_stability_iterations=
                    max_stability_iterations,
            )
        )


    n_pass = sum(
        x.status == "PASS"
        for x in records
    )

    n_reconstruction_fail = sum(
        x.status
        == "QC_FAIL_SCF_RECONSTRUCTION"
        for x in records
    )

    n_identity_fail = sum(
        x.status
        == "QC_FAIL_STATE_IDENTITY"
        for x in records
    )

    n_initially_unstable = sum(
        x.initially_stable is False
        for x in records
    )

    n_finally_unstable = sum(
        x.finally_stable is False
        for x in records
    )

    n_stability_same = sum(
        x.status
        == "STABILITY_REOPT_SAME_STATE"
        for x in records
    )

    n_stability_ambiguous = sum(
        x.status
        == "STABILITY_REOPT_AMBIGUOUS"
        for x in records
    )

    n_stability_root_change = sum(
        x.status
        == "STABILITY_ROOT_CHANGE"
        for x in records
    )


    reconstruction_deltas = [
        abs(
            x.reconstruction_delta_meV
        )
        for x in records
        if x.reconstruction_delta_meV
        is not None
    ]


    stability_deltas = [
        x.stability_delta_energy_meV
        for x in records
        if x.stability_delta_energy_meV
        is not None
    ]


    clean = all(
        x.status == "PASS"
        for x in records
    )


    return FinePECQCResult(
        points=records,

        status=(
            "PASS"
            if clean
            else "QC_REVIEW_REQUIRED"
        ),

        n_points=
            len(records),

        n_pass=
            n_pass,

        n_reconstruction_fail=
            n_reconstruction_fail,

        n_identity_fail=
            n_identity_fail,

        n_initially_unstable=
            n_initially_unstable,

        n_finally_unstable=
            n_finally_unstable,

        n_stability_same_state=
            n_stability_same,

        n_stability_ambiguous=
            n_stability_ambiguous,

        n_stability_state_change=
            n_stability_root_change,

        max_abs_reconstruction_delta_meV=(
            None
            if not reconstruction_deltas
            else max(
                reconstruction_deltas
            )
        ),

        most_negative_stability_delta_meV=(
            None
            if not stability_deltas
            else min(
                stability_deltas
            )
        ),
    )
