from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyscf.data import nist


@dataclass(frozen=True)
class Isotopologue:
    atom_mass_u: float
    ligand_mass_u: float
    label: str = ""


@dataclass
class HarmonicFitResult:
    n_points: int

    r_low_A: float
    r_high_A: float

    discrete_min_r_A: float

    fitted_re_A: float
    fitted_energy_hartree: float

    curvature_hartree_per_A2: float

    reduced_mass_u: float

    harmonic_frequency_hz: float
    harmonic_wavenumber_cm1: float

    zpe_hartree: float
    zpe_eV: float

    fit_rms_microhartree: float
    fit_max_abs_microhartree: float


def reduced_mass_u(
    mass1_u: float,
    mass2_u: float,
) -> float:
    return (
        float(mass1_u)
        * float(mass2_u)
        / (
            float(mass1_u)
            + float(mass2_u)
        )
    )


def harmonic_from_curvature(
    curvature_hartree_per_A2: float,
    mass1_u: float,
    mass2_u: float,
):
    """
    Convert d2E/dR2 in Eh/A^2 into harmonic frequency and ZPE.

    V(R) ~= V(Re) + 1/2 k (R-Re)^2

    omega = sqrt(k / mu)
    """

    curvature = float(
        curvature_hartree_per_A2
    )

    if curvature <= 0.0:
        raise ValueError(
            "Harmonic curvature must be positive."
        )

    mu_u = reduced_mass_u(
        mass1_u,
        mass2_u,
    )

    mu_kg = (
        mu_u
        * nist.ATOMIC_MASS
    )

    # 1 A = 1e-10 m
    k_si = (
        curvature
        * nist.HARTREE2J
        / 1.0e-20
    )

    omega_rad_s = np.sqrt(
        k_si / mu_kg
    )

    frequency_hz = (
        omega_rad_s
        / (2.0 * np.pi)
    )

    wavenumber_cm1 = (
        frequency_hz
        / nist.LIGHT_SPEED_SI
        * 1.0e-2
    )

    zpe_joule = (
        0.5
        * nist.PLANCK
        * frequency_hz
    )

    zpe_hartree = (
        zpe_joule
        / nist.HARTREE2J
    )

    zpe_eV = (
        zpe_hartree
        * nist.HARTREE2EV
    )

    return {
        "reduced_mass_u":
            float(mu_u),

        "frequency_hz":
            float(frequency_hz),

        "wavenumber_cm1":
            float(wavenumber_cm1),

        "zpe_hartree":
            float(zpe_hartree),

        "zpe_eV":
            float(zpe_eV),
    }


def harmonic_quadratic_fit(
    r_A,
    energy_hartree,
    *,
    n_points: int,
    isotopologue: Isotopologue,
) -> HarmonicFitResult:
    """
    Symmetric local quadratic fit around the discrete PEC minimum.

    n_points must be odd.

    This routine assumes the supplied fine PEC is already electronically
    validated.
    """

    if (
        int(n_points) < 3
        or int(n_points) % 2 != 1
    ):
        raise ValueError(
            "n_points must be odd and >= 3."
        )

    r = np.asarray(
        r_A,
        dtype=float,
    )

    e = np.asarray(
        energy_hartree,
        dtype=float,
    )

    if (
        r.ndim != 1
        or e.ndim != 1
        or len(r) != len(e)
    ):
        raise ValueError(
            "r_A and energy_hartree must be equal-length 1D arrays."
        )

    order = np.argsort(
        r
    )

    r = r[order]
    e = e[order]

    min_index = int(
        np.argmin(e)
    )

    half = (
        int(n_points) // 2
    )

    lo = (
        min_index - half
    )

    hi = (
        min_index + half + 1
    )

    if (
        lo < 0
        or hi > len(r)
    ):
        raise ValueError(
            "Not enough symmetric points around the discrete minimum."
        )

    r_fit = np.array(
        r[lo:hi],
        copy=True,
    )

    e_fit = np.array(
        e[lo:hi],
        copy=True,
    )

    # Center x to avoid unnecessary numerical conditioning problems.
    center = float(
        np.mean(r_fit)
    )

    x = (
        r_fit - center
    )

    a, b, c = np.polyfit(
        x,
        e_fit,
        2,
    )

    curvature = (
        2.0 * a
    )

    if curvature <= 0.0:
        raise ValueError(
            "Quadratic fit has non-positive curvature."
        )

    x_min = (
        -b
        / (2.0 * a)
    )

    fitted_re = (
        center + x_min
    )

    if not (
        r_fit[0]
        <= fitted_re
        <= r_fit[-1]
    ):
        raise ValueError(
            "Quadratic minimum lies outside fitted interval."
        )

    fitted_energy = (
        a * x_min**2
        + b * x_min
        + c
    )

    predicted = (
        a * x**2
        + b * x
        + c
    )

    residual = (
        e_fit - predicted
    )

    vib = harmonic_from_curvature(
        curvature,
        isotopologue.atom_mass_u,
        isotopologue.ligand_mass_u,
    )

    return HarmonicFitResult(
        n_points=
            int(n_points),

        r_low_A=
            float(r_fit[0]),

        r_high_A=
            float(r_fit[-1]),

        discrete_min_r_A=
            float(r[min_index]),

        fitted_re_A=
            float(fitted_re),

        fitted_energy_hartree=
            float(fitted_energy),

        curvature_hartree_per_A2=
            float(curvature),

        reduced_mass_u=
            vib["reduced_mass_u"],

        harmonic_frequency_hz=
            vib["frequency_hz"],

        harmonic_wavenumber_cm1=
            vib["wavenumber_cm1"],

        zpe_hartree=
            vib["zpe_hartree"],

        zpe_eV=
            vib["zpe_eV"],

        fit_rms_microhartree=
            float(
                np.sqrt(
                    np.mean(
                        residual**2
                    )
                )
                * 1.0e6
            ),

        fit_max_abs_microhartree=
            float(
                np.max(
                    np.abs(
                        residual
                    )
                )
                * 1.0e6
            ),
    )


def zpe_corrected_ea_eV(
    *,
    neutral_energy_hartree: float,
    anion_energy_hartree: float,
    neutral_zpe_hartree: float,
    anion_zpe_hartree: float,
) -> float:
    """
    EA_0 = EA_el + ZPE_neutral - ZPE_anion
    """

    ea0_hartree = (
        float(
            neutral_energy_hartree
        )
        - float(
            anion_energy_hartree
        )
        + float(
            neutral_zpe_hartree
        )
        - float(
            anion_zpe_hartree
        )
    )

    return (
        ea0_hartree
        * nist.HARTREE2EV
    )
