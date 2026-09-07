import math
from dataclasses import replace
from typing import Dict, Iterable, List

import numpy as np

from .model import MoleculeSpec, SCFOutcome, SCFSettings
from .molecule import build_molecule
from .scf import HARTREE_TO_EV, run_uks


def project_density(
    old_mol,
    old_dm,
    new_mol,
):
    """
    Project a converged density matrix from one geometry onto the AO basis
    of the neighboring geometry.
    """
    from pyscf.scf import addons

    return addons.project_dm_nr2nr(
        old_mol,
        old_dm,
        new_mol,
    )


def make_point_record(
    *,
    state_id: str,
    seed_provenance: str,
    direction: str,
    parent_r_A,
    spec: MoleculeSpec,
    outcome: SCFOutcome,
    projected_guess: bool,
) -> Dict[str, object]:

    return {
        "molecule": spec.molecule,
        "atom": spec.atom,
        "ligand": spec.ligand,

        "charge": spec.charge,
        "spin": spec.spin,
        "multiplicity": spec.multiplicity,

        "functional": "",
        "basis": spec.basis,

        "state_id": state_id,
        "seed_provenance": seed_provenance,

        "direction": direction,
        "parent_r_A": parent_r_A,
        "r_A": float(spec.r_A),

        "energy_hartree": outcome.energy_hartree,
        "energy_eV": (
            outcome.energy_hartree * HARTREE_TO_EV
            if math.isfinite(outcome.energy_hartree)
            else math.nan
        ),

        "converged": outcome.converged,
        "projected_guess": projected_guess,

        "scf_path": outcome.scf_path,
        "final_solver": outcome.final_solver,

        "used_level_shift_helper":
            outcome.used_level_shift_helper,

        "used_newton_solver":
            outcome.used_newton_solver,

        "homo_eV": outcome.homo_eV,
        "lumo_eV": outcome.lumo_eV,
        "gap_eV": outcome.gap_eV,

        "positive_homo_warning":
            outcome.positive_homo_warning,

        "s2": outcome.s2,
        "observed_multiplicity":
            outcome.observed_multiplicity,

        "error": outcome.error,
    }


def follow_state(
    *,
    seed_spec: MoleculeSpec,
    seed_outcome: SCFOutcome,
    state_id: str,
    seed_provenance: str,
    r_values: Iterable[float],
    settings: SCFSettings,
) -> List[Dict[str, object]]:
    """
    Follow one already-defined electronic candidate away from a seed geometry.

    Important:
    this primitive transports a candidate state by density projection.
    It does NOT decide whether two candidates represent the same electronic
    state. Candidate discovery and identity/deduplication belong to a higher
    workflow layer.
    """
    if not seed_outcome.converged:
        raise RuntimeError(
            "Cannot follow a non-converged seed."
        )

    if seed_outcome.mf is None:
        raise RuntimeError(
            "Converged seed has no PySCF mean-field object."
        )

    seed_r = float(seed_spec.r_A)

    grid = sorted(
        {
            round(float(r), 10)
            for r in r_values
        }
    )

    if not any(
        abs(r - seed_r) < 1.0e-9
        for r in grid
    ):
        grid.append(seed_r)
        grid.sort()

    records = [
        make_point_record(
            state_id=state_id,
            seed_provenance=seed_provenance,
            direction="seed",
            parent_r_A=None,
            spec=seed_spec,
            outcome=seed_outcome,
            projected_guess=False,
        )
    ]

    seed_mol = seed_outcome.mf.mol
    seed_dm = np.array(
        seed_outcome.mf.make_rdm1(),
        copy=True,
    )

    directions = [
        (
            "down",
            sorted(
                [r for r in grid if r < seed_r],
                reverse=True,
            ),
        ),
        (
            "up",
            sorted(
                [r for r in grid if r > seed_r]
            ),
        ),
    ]

    for direction, values in directions:

        previous_mol = seed_mol
        previous_dm = np.array(
            seed_dm,
            copy=True,
        )
        previous_r = seed_r

        for r_A in values:

            spec = replace(
                seed_spec,
                r_A=float(r_A),
            )

            mol = build_molecule(spec)

            try:
                dm0 = project_density(
                    previous_mol,
                    previous_dm,
                    mol,
                )

                outcome = run_uks(
                    mol,
                    settings,
                    dm0=dm0,
                )

            except Exception as exc:
                records.append({
                    "molecule": spec.molecule,
                    "atom": spec.atom,
                    "ligand": spec.ligand,
                    "charge": spec.charge,
                    "spin": spec.spin,
                    "multiplicity": spec.multiplicity,
                    "functional": settings.xc,
                    "basis": spec.basis,
                    "state_id": state_id,
                    "seed_provenance": seed_provenance,
                    "direction": direction,
                    "parent_r_A": previous_r,
                    "r_A": float(r_A),
                    "converged": False,
                    "projected_guess": True,
                    "error": repr(exc),
                })
                break

            record = make_point_record(
                state_id=state_id,
                seed_provenance=seed_provenance,
                direction=direction,
                parent_r_A=previous_r,
                spec=spec,
                outcome=outcome,
                projected_guess=True,
            )

            record["functional"] = settings.xc
            records.append(record)

            if not outcome.converged:
                break

            previous_mol = outcome.mf.mol
            previous_dm = np.array(
                outcome.mf.make_rdm1(),
                copy=True,
            )
            previous_r = float(r_A)

    records[0]["functional"] = settings.xc

    return records
