import math
import traceback
from typing import Optional, Tuple

import numpy as np

from .model import SCFOutcome, SCFSettings


HARTREE_TO_EV = 27.211386245988


def functional_to_xc(name: str) -> str:
    """
    Small transparent alias layer.

    Arbitrary LibXC/PySCF strings remain allowed; the aliases merely make
    common workflow names explicit.
    """
    aliases = {
        "PBE": "PBE",
        "PBE0": "PBE0",
        "B3LYP": "B3LYP",
        "TPSSH": "TPSSh",
    }

    key = str(name).strip()

    return aliases.get(
        key.upper(),
        key,
    )


def _make_uks(
    mol,
    settings: SCFSettings,
    level_shift: float = 0.0,
):
    from pyscf import dft

    mf = dft.UKS(mol)

    mf.xc = functional_to_xc(settings.xc)
    mf.conv_tol = float(settings.conv_tol)
    mf.max_cycle = int(settings.max_cycle)
    mf.grids.level = int(settings.grid_level)
    mf.max_memory = int(mol.max_memory)

    mf.direct_scf = True
    mf.init_guess = "minao"
    mf.verbose = 0

    if level_shift:
        mf.level_shift = float(level_shift)

    return mf


def _kernel(mf, dm0=None) -> float:
    if dm0 is None:
        return float(mf.kernel())

    return float(mf.kernel(dm0))


def _safe_density(mf):
    try:
        return np.array(
            mf.make_rdm1(),
            copy=True,
        )
    except Exception:
        return None


def _frontier_orbitals(mf) -> Tuple[float, float]:
    occupied = []
    virtual = []

    energies = mf.mo_energy
    occupations = mf.mo_occ

    if isinstance(energies, np.ndarray) and energies.ndim == 1:
        energies = [energies]
        occupations = [occupations]

    for eps, occ in zip(energies, occupations):
        eps = np.asarray(eps, dtype=float)
        occ = np.asarray(occ, dtype=float)

        occupied.extend(
            eps[occ > 1.0e-8].tolist()
        )

        virtual.extend(
            eps[occ <= 1.0e-8].tolist()
        )

    homo = (
        float(max(occupied))
        if occupied
        else math.nan
    )

    lumo = (
        float(min(virtual))
        if virtual
        else math.nan
    )

    return homo, lumo


def _diagnostics(mf):
    homo_h = math.nan
    lumo_h = math.nan

    try:
        homo_h, lumo_h = _frontier_orbitals(mf)
    except Exception:
        pass

    homo_eV = (
        homo_h * HARTREE_TO_EV
        if math.isfinite(homo_h)
        else math.nan
    )

    lumo_eV = (
        lumo_h * HARTREE_TO_EV
        if math.isfinite(lumo_h)
        else math.nan
    )

    gap_eV = (
        (lumo_h - homo_h) * HARTREE_TO_EV
        if math.isfinite(homo_h)
        and math.isfinite(lumo_h)
        else math.nan
    )

    try:
        s2, observed_multiplicity = mf.spin_square()
        s2 = float(s2)
        observed_multiplicity = float(observed_multiplicity)
    except Exception:
        s2 = math.nan
        observed_multiplicity = math.nan

    return {
        "homo_hartree": homo_h,
        "lumo_hartree": lumo_h,
        "homo_eV": homo_eV,
        "lumo_eV": lumo_eV,
        "gap_eV": gap_eV,
        "positive_homo_warning": (
            bool(math.isfinite(homo_h) and homo_h > 0.0)
        ),
        "s2": s2,
        "observed_multiplicity": observed_multiplicity,
    }


def run_uks(
    mol,
    settings: SCFSettings,
    dm0=None,
) -> SCFOutcome:
    """
    Universal UKS execution policy.

    The level-shift calculation is a convergence helper only. Its energy is
    never accepted as the final reported energy. A normal unshifted SCF is
    attempted afterwards.

    Newton is treated as an alternative solver for the same SCF equations,
    not as a different electronic-structure method.
    """
    from pyscf import lib

    lib.num_threads(int(settings.threads))

    path = []
    errors = []

    used_level_shift = False
    used_newton = False

    final_mf = None
    final_energy = math.nan
    final_solver = "none"

    latest_dm = (
        np.array(dm0, copy=True)
        if dm0 is not None
        else None
    )

    # ------------------------------------------------------------
    # 1. Standard UKS
    # ------------------------------------------------------------

    try:
        mf = _make_uks(
            mol,
            settings,
            level_shift=0.0,
        )

        energy = _kernel(
            mf,
            latest_dm,
        )

        path.append("standard")

        final_mf = mf
        final_energy = energy

        if bool(getattr(mf, "converged", False)):
            final_solver = "standard"

        else:
            dm = _safe_density(mf)

            if dm is not None:
                latest_dm = dm

    except Exception:
        errors.append(
            "standard: "
            + traceback.format_exc(limit=5).replace("\n", " | ")
        )

    # ------------------------------------------------------------
    # 2. Level shift as helper, followed by mandatory unshifted SCF
    # ------------------------------------------------------------

    if final_solver == "none":

        try:
            shifted = _make_uks(
                mol,
                settings,
                level_shift=settings.level_shift_helper,
            )

            _kernel(
                shifted,
                latest_dm,
            )

            path.append("level_shift_helper")
            used_level_shift = True

            dm = _safe_density(shifted)

            if dm is not None:
                latest_dm = dm

        except Exception:
            errors.append(
                "level_shift_helper: "
                + traceback.format_exc(limit=5).replace("\n", " | ")
            )

        try:
            polished = _make_uks(
                mol,
                settings,
                level_shift=0.0,
            )

            energy = _kernel(
                polished,
                latest_dm,
            )

            path.append("standard_after_shift")

            final_mf = polished
            final_energy = energy

            if bool(
                getattr(
                    polished,
                    "converged",
                    False,
                )
            ):
                final_solver = "standard_after_shift"

            else:
                dm = _safe_density(polished)

                if dm is not None:
                    latest_dm = dm

        except Exception:
            errors.append(
                "standard_after_shift: "
                + traceback.format_exc(limit=5).replace("\n", " | ")
            )

    # ------------------------------------------------------------
    # 3. Newton solver if ordinary SCF still did not converge
    # ------------------------------------------------------------

    if final_solver == "none":

        try:
            base = _make_uks(
                mol,
                settings,
                level_shift=0.0,
            )

            newton = base.newton()

            newton.max_cycle = int(settings.max_cycle)
            newton.conv_tol = float(settings.conv_tol)
            newton.verbose = 0

            energy = _kernel(
                newton,
                latest_dm,
            )

            path.append("newton")
            used_newton = True

            final_mf = newton
            final_energy = energy

            if bool(
                getattr(
                    newton,
                    "converged",
                    False,
                )
            ):
                final_solver = "newton"

                dm = _safe_density(newton)

                # Try once to return to ordinary UKS. Failure here does
                # not invalidate a converged Newton solution because the
                # physical SCF equations are unchanged.
                if dm is not None:
                    try:
                        polished = _make_uks(
                            mol,
                            settings,
                            level_shift=0.0,
                        )

                        polished_energy = _kernel(
                            polished,
                            dm,
                        )

                        path.append("standard_after_newton")

                        if bool(
                            getattr(
                                polished,
                                "converged",
                                False,
                            )
                        ):
                            final_mf = polished
                            final_energy = polished_energy
                            final_solver = "standard_after_newton"

                    except Exception:
                        errors.append(
                            "standard_after_newton: "
                            + traceback.format_exc(limit=5).replace(
                                "\n",
                                " | ",
                            )
                        )

        except Exception:
            errors.append(
                "newton: "
                + traceback.format_exc(limit=5).replace("\n", " | ")
            )

    converged = (
        final_mf is not None
        and bool(
            getattr(
                final_mf,
                "converged",
                False,
            )
        )
        and final_solver != "none"
    )

    if final_mf is not None:
        diag = _diagnostics(final_mf)
    else:
        diag = {
            "homo_hartree": math.nan,
            "lumo_hartree": math.nan,
            "homo_eV": math.nan,
            "lumo_eV": math.nan,
            "gap_eV": math.nan,
            "positive_homo_warning": False,
            "s2": math.nan,
            "observed_multiplicity": math.nan,
        }

    return SCFOutcome(
        mf=final_mf,
        energy_hartree=float(final_energy),
        converged=bool(converged),
        scf_path=" -> ".join(path),
        final_solver=final_solver,
        used_level_shift_helper=used_level_shift,
        used_newton_solver=used_newton,
        homo_hartree=diag["homo_hartree"],
        lumo_hartree=diag["lumo_hartree"],
        homo_eV=diag["homo_eV"],
        lumo_eV=diag["lumo_eV"],
        gap_eV=diag["gap_eV"],
        positive_homo_warning=diag["positive_homo_warning"],
        s2=diag["s2"],
        observed_multiplicity=diag[
            "observed_multiplicity"
        ],
        error=" || ".join(errors),
    )
