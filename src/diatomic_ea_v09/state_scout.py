from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np

from .model import MoleculeSpec, SCFSettings
from .molecule import build_molecule
from .scf import HARTREE_TO_EV, run_uks


DEFAULT_GUESSES = (
    "minao",
    "atom",
    "huckel",
    "hcore",
)


@dataclass
class ScoutSolution:
    molecule: str
    charge: int
    spin: int
    multiplicity: int
    r_A: float
    functional: str
    basis: str

    origin_guess: str
    equivalent_guesses: List[str]

    energy_hartree: float
    s2: float
    observed_multiplicity: float

    homo_eV: float
    lumo_eV: float
    gap_eV: float

    scf_path: str

    density_ao: np.ndarray
    density_orth: np.ndarray

    mf: Any = None
    candidate_id: str = ""


def orthonormal_spin_density(
    mol,
    dm,
) -> np.ndarray:
    """
    Spin-resolved density in the symmetric Löwdin-orthonormal AO basis.

    This representation is invariant to ordinary occupied-orbital rotations
    and is used only for comparing converged SCF solutions at the SAME
    geometry, basis, charge and spin.
    """
    s = mol.intor_symmetric("int1e_ovlp")

    eigval, eigvec = np.linalg.eigh(s)

    if np.any(eigval <= 0.0):
        raise RuntimeError(
            "AO overlap matrix is not positive definite."
        )

    s_half = (
        eigvec
        @ np.diag(np.sqrt(eigval))
        @ eigvec.T
    )

    dm = np.asarray(dm)

    if dm.ndim != 3 or dm.shape[0] != 2:
        raise RuntimeError(
            "Expected spin-resolved UKS density with shape (2, nao, nao)."
        )

    return np.stack(
        [
            s_half @ dm_spin @ s_half
            for dm_spin in dm
        ]
    )


def density_distance(
    a: ScoutSolution,
    b: ScoutSolution,
) -> float:
    return float(
        np.linalg.norm(
            a.density_orth
            - b.density_orth
        )
    )


def energy_distance_meV(
    a: ScoutSolution,
    b: ScoutSolution,
) -> float:
    return abs(
        float(a.energy_hartree)
        - float(b.energy_hartree)
    ) * HARTREE_TO_EV * 1000.0


def run_guess(
    spec: MoleculeSpec,
    settings: SCFSettings,
    guess: str,
) -> ScoutSolution | None:
    """
    Run one universal SCF starting guess at one fixed geometry.
    """
    from pyscf import dft

    mol = build_molecule(spec)

    guess_mf = dft.UKS(mol)

    guess_mf.xc = settings.xc
    guess_mf.grids.level = int(settings.grid_level)
    guess_mf.max_memory = int(spec.max_memory_mb)
    guess_mf.verbose = 0

    dm0 = guess_mf.get_init_guess(
        mol,
        key=str(guess),
    )

    outcome = run_uks(
        mol,
        settings,
        dm0=dm0,
    )

    if not outcome.converged:
        return None

    dm = np.array(
        outcome.mf.make_rdm1(),
        copy=True,
    )

    dm_orth = orthonormal_spin_density(
        outcome.mf.mol,
        dm,
    )

    return ScoutSolution(
        molecule=spec.molecule,
        charge=int(spec.charge),
        spin=int(spec.spin),
        multiplicity=int(spec.multiplicity),
        r_A=float(spec.r_A),
        functional=str(settings.xc),
        basis=str(spec.basis),

        origin_guess=str(guess),
        equivalent_guesses=[str(guess)],

        energy_hartree=float(
            outcome.energy_hartree
        ),
        s2=float(outcome.s2),
        observed_multiplicity=float(
            outcome.observed_multiplicity
        ),

        homo_eV=float(outcome.homo_eV),
        lumo_eV=float(outcome.lumo_eV),
        gap_eV=float(outcome.gap_eV),

        scf_path=str(outcome.scf_path),

        density_ao=dm,
        density_orth=dm_orth,

        mf=outcome.mf,
    )


def scout_spin_at_geometry(
    spec: MoleculeSpec,
    settings: SCFSettings,
    guesses: Iterable[str] = DEFAULT_GUESSES,
) -> List[ScoutSolution]:
    """
    Generate all converged multistart SCF solutions for one fixed
    molecule/charge/spin/geometry.
    """
    raw: List[ScoutSolution] = []

    for guess in guesses:
        solution = run_guess(
            spec,
            settings,
            guess,
        )

        if solution is not None:
            raw.append(solution)

    return raw


def deduplicate_same_spin(
    solutions: Iterable[ScoutSolution],
    *,
    energy_tol_meV: float = 1.0,
    density_tol: float = 0.01,
) -> List[ScoutSolution]:
    """
    Conservatively merge only solutions that are simultaneously close
    in energy and in their spin-resolved orthonormal density.

    IMPORTANT:
    These tolerances are DEVELOPMENT-PILOT values, not frozen production
    criteria.
    """
    ordered = sorted(
        list(solutions),
        key=lambda x: x.energy_hartree,
    )

    unique: List[ScoutSolution] = []

    for candidate in ordered:

        duplicate_of = None

        for existing in unique:

            if candidate.spin != existing.spin:
                continue

            dE = energy_distance_meV(
                candidate,
                existing,
            )

            dP = density_distance(
                candidate,
                existing,
            )

            if (
                dE <= float(energy_tol_meV)
                and dP <= float(density_tol)
            ):
                duplicate_of = existing
                break

        if duplicate_of is None:
            unique.append(candidate)

        else:
            guesses = set(
                duplicate_of.equivalent_guesses
            )

            guesses.update(
                candidate.equivalent_guesses
            )

            duplicate_of.equivalent_guesses = sorted(
                guesses
            )

    return unique


def select_lowest_candidates(
    solutions: Iterable[ScoutSolution],
    *,
    n_keep: int = 4,
) -> List[ScoutSolution]:
    """
    Select the globally lowest distinct candidates AFTER within-spin
    deduplication has been performed.
    """
    ordered = sorted(
        list(solutions),
        key=lambda x: x.energy_hartree,
    )

    return ordered[: int(n_keep)]


def assign_candidate_ids(
    candidates: Iterable[ScoutSolution],
) -> List[ScoutSolution]:
    """
    Assign deterministic discovery-stage candidate IDs.

    IDs are persistent labels. They do not encode a spectroscopic state
    assignment.
    """
    candidates = sorted(
        list(candidates),
        key=lambda x: x.energy_hartree,
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        charge_tag = (
            "N"
            if candidate.charge == 0
            else "A"
        )

        candidate.candidate_id = (
            f"{candidate.molecule.upper()}_"
            f"{charge_tag}_"
            f"S{candidate.spin}_"
            f"C{rank:02d}"
        )

    return candidates


def comparison_record(
    a: ScoutSolution,
    b: ScoutSolution,
) -> Dict[str, object]:
    return {
        "guess_a": a.origin_guess,
        "guess_b": b.origin_guess,
        "spin": a.spin,
        "delta_energy_meV":
            energy_distance_meV(a, b),
        "density_distance":
            density_distance(a, b),
        "delta_s2":
            float(b.s2 - a.s2),
    }
