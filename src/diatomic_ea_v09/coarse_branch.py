from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from pyscf.scf import addons

from .model import MoleculeSpec, SCFSettings
from .molecule import build_molecule
from .scf import run_uks
from .state_scout import (
    ScoutSolution,
    orthonormal_spin_density,
)


@dataclass
class BranchPoint:
    branch_id: str

    r_A: float
    parent_r_A: float | None

    spin: int
    multiplicity: int

    converged: bool
    energy_hartree: float | None

    s2: float | None
    observed_multiplicity: float | None

    homo_eV: float | None
    lumo_eV: float | None
    gap_eV: float | None

    scf_path: str

    density_ao: np.ndarray | None
    density_orth: np.ndarray | None

    point_source: str


def split_grid_around_seed(
    r_values: Sequence[float],
    seed_r_A: float,
):
    """
    Return deterministic propagation order.

    Lower-R side:
        nearest seed -> outward

    Higher-R side:
        nearest seed -> outward
    """
    unique = sorted(
        {
            round(float(r), 10)
            for r in r_values
        }
    )

    seed = round(
        float(seed_r_A),
        10,
    )

    lower = sorted(
        [
            r
            for r in unique
            if r < seed
        ],
        reverse=True,
    )

    upper = sorted(
        [
            r
            for r in unique
            if r > seed
        ]
    )

    return lower, upper


def _spec_at_r(
    *,
    atom: str,
    ligand: str,
    charge: int,
    spin: int,
    basis: str,
    r_A: float,
    max_memory_mb: int,
):
    return MoleculeSpec(
        atom=atom,
        ligand=ligand,
        charge=charge,
        spin=spin,
        basis=basis,
        r_A=float(r_A),
        max_memory_mb=max_memory_mb,
    )


def _project_density(
    old_mol,
    old_dm,
    new_mol,
):
    projected = addons.project_dm_nr2nr(
        old_mol,
        old_dm,
        new_mol,
    )

    return np.asarray(
        projected
    )


def _propagate_side(
    *,
    branch_id: str,
    atom: str,
    ligand: str,
    charge: int,
    spin: int,
    basis: str,
    settings: SCFSettings,
    max_memory_mb: int,
    start_r_A: float,
    start_dm,
    target_r_values,
) -> List[BranchPoint]:

    points = []

    previous_r = float(
        start_r_A
    )

    previous_spec = _spec_at_r(
        atom=atom,
        ligand=ligand,
        charge=charge,
        spin=spin,
        basis=basis,
        r_A=previous_r,
        max_memory_mb=max_memory_mb,
    )

    previous_mol = build_molecule(
        previous_spec
    )

    previous_dm = np.asarray(
        start_dm
    )

    for target_r in target_r_values:

        target_spec = _spec_at_r(
            atom=atom,
            ligand=ligand,
            charge=charge,
            spin=spin,
            basis=basis,
            r_A=target_r,
            max_memory_mb=max_memory_mb,
        )

        target_mol = build_molecule(
            target_spec
        )

        try:
            dm0 = _project_density(
                previous_mol,
                previous_dm,
                target_mol,
            )

            outcome = run_uks(
                target_mol,
                settings,
                dm0=dm0,
            )

        except Exception as exc:

            points.append(
                BranchPoint(
                    branch_id=branch_id,
                    r_A=float(target_r),
                    parent_r_A=float(
                        previous_r
                    ),
                    spin=int(spin),
                    multiplicity=
                        int(spin + 1),
                    converged=False,
                    energy_hartree=None,
                    s2=None,
                    observed_multiplicity=None,
                    homo_eV=None,
                    lumo_eV=None,
                    gap_eV=None,
                    scf_path=(
                        "ERROR:"
                        + repr(exc)
                    ),
                    density_ao=None,
                    density_orth=None,
                    point_source=
                        "state_follow",
                )
            )

            # State continuity is broken on this side.
            # Do not jump over a failed point.
            break

        if not outcome.converged:

            points.append(
                BranchPoint(
                    branch_id=branch_id,
                    r_A=float(target_r),
                    parent_r_A=float(
                        previous_r
                    ),
                    spin=int(spin),
                    multiplicity=
                        int(spin + 1),
                    converged=False,
                    energy_hartree=None,
                    s2=None,
                    observed_multiplicity=None,
                    homo_eV=None,
                    lumo_eV=None,
                    gap_eV=None,
                    scf_path=str(
                        outcome.scf_path
                    ),
                    density_ao=None,
                    density_orth=None,
                    point_source=
                        "state_follow",
                )
            )

            break

        dm = np.asarray(
            outcome.mf.make_rdm1()
        )

        dm_orth = (
            orthonormal_spin_density(
                target_mol,
                dm,
            )
        )

        points.append(
            BranchPoint(
                branch_id=branch_id,
                r_A=float(target_r),
                parent_r_A=float(
                    previous_r
                ),
                spin=int(spin),
                multiplicity=
                    int(spin + 1),
                converged=True,
                energy_hartree=float(
                    outcome.energy_hartree
                ),
                s2=float(
                    outcome.s2
                ),
                observed_multiplicity=float(
                    outcome.observed_multiplicity
                ),
                homo_eV=float(
                    outcome.homo_eV
                ),
                lumo_eV=float(
                    outcome.lumo_eV
                ),
                gap_eV=float(
                    outcome.gap_eV
                ),
                scf_path=str(
                    outcome.scf_path
                ),
                density_ao=np.array(
                    dm,
                    copy=True,
                ),
                density_orth=np.array(
                    dm_orth,
                    copy=True,
                ),
                point_source=
                    "state_follow",
            )
        )

        previous_r = float(
            target_r
        )

        previous_mol = target_mol

        previous_dm = np.array(
            dm,
            copy=True,
        )

    return points


def follow_scout_solution(
    *,
    branch_id: str,
    atom: str,
    ligand: str,
    seed: ScoutSolution,
    settings: SCFSettings,
    r_values: Sequence[float],
    max_memory_mb: int = 2000,
) -> List[BranchPoint]:
    """
    Follow one converged scout solution independently toward lower
    and higher R.

    The seed density is the only common origin of the two directions.
    """

    seed_r = float(
        seed.r_A
    )

    lower, upper = (
        split_grid_around_seed(
            r_values,
            seed_r,
        )
    )

    seed_point = BranchPoint(
        branch_id=branch_id,
        r_A=seed_r,
        parent_r_A=None,
        spin=int(seed.spin),
        multiplicity=
            int(seed.multiplicity),
        converged=True,
        energy_hartree=float(
            seed.energy_hartree
        ),
        s2=float(seed.s2),
        observed_multiplicity=float(
            seed.observed_multiplicity
        ),
        homo_eV=float(
            seed.homo_eV
        ),
        lumo_eV=float(
            seed.lumo_eV
        ),
        gap_eV=float(
            seed.gap_eV
        ),
        scf_path=str(
            seed.scf_path
        ),
        density_ao=np.array(
            seed.density_ao,
            copy=True,
        ),
        density_orth=np.array(
            seed.density_orth,
            copy=True,
        ),
        point_source=
            "scout_seed",
    )

    low_points = _propagate_side(
        branch_id=branch_id,
        atom=atom,
        ligand=ligand,
        charge=seed.charge,
        spin=seed.spin,
        basis=seed.basis,
        settings=settings,
        max_memory_mb=max_memory_mb,
        start_r_A=seed_r,
        start_dm=seed.density_ao,
        target_r_values=lower,
    )

    high_points = _propagate_side(
        branch_id=branch_id,
        atom=atom,
        ligand=ligand,
        charge=seed.charge,
        spin=seed.spin,
        basis=seed.basis,
        settings=settings,
        max_memory_mb=max_memory_mb,
        start_r_A=seed_r,
        start_dm=seed.density_ao,
        target_r_values=upper,
    )

    return sorted(
        low_points
        + [seed_point]
        + high_points,
        key=lambda p: p.r_A,
    )


def extend_branch_points(
    *,
    points,
    atom: str,
    ligand: str,
    charge: int,
    spin: int,
    basis: str,
    settings: SCFSettings,
    new_r_values,
    direction: str,
    max_memory_mb: int = 2000,
):
    """
    Incrementally extend an already propagated branch.

    Existing points are retained and are not recomputed.
    """

    good = sorted(
        [
            p for p in points
            if (
                p.converged
                and p.density_ao is not None
            )
        ],
        key=lambda p: p.r_A,
    )

    if not good:
        raise RuntimeError(
            "Cannot extend branch without a converged point."
        )

    if direction == "lower":
        boundary = good[0]

        targets = sorted(
            [
                float(r)
                for r in new_r_values
                if float(r) < boundary.r_A
            ],
            reverse=True,
        )

    elif direction == "upper":
        boundary = good[-1]

        targets = sorted(
            [
                float(r)
                for r in new_r_values
                if float(r) > boundary.r_A
            ]
        )

    else:
        raise ValueError(
            "direction must be 'lower' or 'upper'"
        )

    if not targets:
        return list(points)

    extra = _propagate_side(
        branch_id=boundary.branch_id,
        atom=atom,
        ligand=ligand,
        charge=charge,
        spin=spin,
        basis=basis,
        settings=settings,
        max_memory_mb=max_memory_mb,
        start_r_A=boundary.r_A,
        start_dm=boundary.density_ao,
        target_r_values=targets,
    )

    merged = {
        round(p.r_A, 10): p
        for p in points
    }

    for point in extra:
        merged[
            round(point.r_A, 10)
        ] = point

    return sorted(
        merged.values(),
        key=lambda p: p.r_A,
    )
