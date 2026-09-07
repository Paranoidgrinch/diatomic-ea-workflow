from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np

from .scf import HARTREE_TO_EV


class StateRelation(str, Enum):
    SAME_STATE = "SAME_STATE"
    DISTINCT_STATE = "DISTINCT_STATE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class IdentityThresholds:

    # Conservative development values.
    # These are NOT production-frozen parameters.

    same_energy_meV: float = 1.0
    same_delta_s2: float = 1.0e-3
    same_total_spectrum_max: float = 1.0e-4
    same_spin_spectrum_max: float = 5.0e-4

    distinct_energy_meV: float = 5.0
    distinct_delta_s2: float = 1.0e-2
    distinct_total_spectrum_max: float = 1.0e-3
    distinct_spin_spectrum_max: float = 5.0e-3


@dataclass(frozen=True)
class StateComparison:

    delta_energy_meV: float
    delta_s2: float

    raw_density_distance: float

    total_spectrum_max: float
    total_spectrum_l2: float

    spin_spectrum_max: float
    spin_spectrum_l2: float

    relation: StateRelation


def density_invariant_spectra(
    dm_orth,
) -> Dict[str, np.ndarray]:
    """
    Return rotation-insensitive spectra derived from a spin-resolved
    orthonormal AO density.

    The separate alpha/beta density eigenvalue spectra are deliberately
    not used for state identity: for a single-determinant UKS solution
    they are largely fixed by orbital occupation.

    The total- and spin-density spectra contain the useful comparison
    information for the present purpose.
    """
    dm = np.asarray(
        dm_orth,
        dtype=float,
    )

    if dm.ndim != 3 or dm.shape[0] != 2:
        raise ValueError(
            "Expected spin-resolved density with shape (2, nao, nao)."
        )

    da = 0.5 * (
        dm[0] + dm[0].T
    )

    db = 0.5 * (
        dm[1] + dm[1].T
    )

    total = da + db
    spin = da - db

    return {
        "total": np.sort(
            np.linalg.eigvalsh(total)
        )[::-1],

        "spin": np.sort(
            np.linalg.eigvalsh(spin)
        )[::-1],
    }


def spectrum_distances(
    spectra_a,
    spectra_b,
):
    out = {}

    for key in (
        "total",
        "spin",
    ):

        delta = (
            np.asarray(spectra_a[key])
            - np.asarray(spectra_b[key])
        )

        out[key + "_max"] = float(
            np.max(
                np.abs(delta)
            )
        )

        out[key + "_l2"] = float(
            np.linalg.norm(delta)
        )

    return out


def classify_metrics(
    *,
    same_spin: bool,
    delta_energy_meV: float,
    delta_s2: float,
    total_spectrum_max: float,
    spin_spectrum_max: float,
    thresholds: IdentityThresholds = IdentityThresholds(),
) -> StateRelation:
    """
    Conservative three-way state-equivalence decision.

    Different spin sectors are always distinct.

    SAME_STATE requires all tested descriptors to agree.

    DISTINCT_STATE requires a meaningful energy separation together
    with at least one clearly different structural/spin descriptor.

    Everything between those limits remains AMBIGUOUS and must be
    retained for further PEC-level analysis.
    """

    if not same_spin:
        return StateRelation.DISTINCT_STATE

    dE = abs(
        float(delta_energy_meV)
    )

    dS2 = abs(
        float(delta_s2)
    )

    dtotal = abs(
        float(total_spectrum_max)
    )

    dspin = abs(
        float(spin_spectrum_max)
    )

    if (
        dE <= thresholds.same_energy_meV
        and dS2 <= thresholds.same_delta_s2
        and dtotal <= thresholds.same_total_spectrum_max
        and dspin <= thresholds.same_spin_spectrum_max
    ):
        return StateRelation.SAME_STATE

    structurally_distinct = (
        dS2 >= thresholds.distinct_delta_s2
        or dtotal >= thresholds.distinct_total_spectrum_max
        or dspin >= thresholds.distinct_spin_spectrum_max
    )

    if (
        dE >= thresholds.distinct_energy_meV
        and structurally_distinct
    ):
        return StateRelation.DISTINCT_STATE

    return StateRelation.AMBIGUOUS


def compare_states(
    *,
    energy_a_hartree: float,
    energy_b_hartree: float,
    s2_a: float,
    s2_b: float,
    dm_orth_a,
    dm_orth_b,
    spin_a: int,
    spin_b: int,
    thresholds: IdentityThresholds = IdentityThresholds(),
) -> StateComparison:

    dE_meV = abs(
        float(
            energy_b_hartree
            - energy_a_hartree
        )
    ) * HARTREE_TO_EV * 1000.0

    dS2 = abs(
        float(
            s2_b - s2_a
        )
    )

    raw_density_distance = float(
        np.linalg.norm(
            np.asarray(dm_orth_b)
            - np.asarray(dm_orth_a)
        )
    )

    spectra_a = density_invariant_spectra(
        dm_orth_a
    )

    spectra_b = density_invariant_spectra(
        dm_orth_b
    )

    distances = spectrum_distances(
        spectra_a,
        spectra_b,
    )

    relation = classify_metrics(
        same_spin=(
            int(spin_a)
            == int(spin_b)
        ),
        delta_energy_meV=dE_meV,
        delta_s2=dS2,
        total_spectrum_max=
            distances["total_max"],
        spin_spectrum_max=
            distances["spin_max"],
        thresholds=thresholds,
    )

    return StateComparison(
        delta_energy_meV=dE_meV,
        delta_s2=dS2,

        raw_density_distance=
            raw_density_distance,

        total_spectrum_max=
            distances["total_max"],

        total_spectrum_l2=
            distances["total_l2"],

        spin_spectrum_max=
            distances["spin_max"],

        spin_spectrum_l2=
            distances["spin_l2"],

        relation=relation,
    )
