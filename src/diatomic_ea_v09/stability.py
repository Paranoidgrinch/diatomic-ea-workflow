from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .model import SCFSettings
from .scf import HARTREE_TO_EV, run_uks


@dataclass
class StabilityIteration:
    iteration: int
    energy_before_hartree: float
    stable_internal: bool
    energy_after_hartree: float | None
    delta_energy_meV: float | None
    scf_path: str | None


@dataclass
class StabilityResult:
    initial_mf: Any
    final_mf: Any

    initially_stable: bool
    finally_stable: bool

    iterations: int
    reoptimizations: int

    initial_energy_hartree: float
    final_energy_hartree: float
    total_delta_energy_meV: float

    history: List[StabilityIteration]


def optimize_internal_stability(
    mf,
    settings: SCFSettings,
    *,
    max_iterations: int = 6,
) -> StabilityResult:
    """
    Bounded internal UKS/UHF ground-state stability optimization.

    max_iterations is the maximum number of orbital-rotation
    reoptimizations.

    A final stability CHECK is always performed after the final allowed
    reoptimization. Thus max_iterations=6 permits:

        check -> reopt
        ...
        check -> reopt   [sixth reoptimization]
        final check

    This routine is appropriate for ground-state canonicalization.

    It must not be used to define whether a deliberately state-specific
    excited-state solution is physically distinct.
    """

    initial_mf = mf
    current_mf = mf

    initial_energy = float(
        current_mf.e_tot
    )

    history = []

    initially_stable = None
    finally_stable = False

    reoptimizations = 0

    # max_iterations reoptimizations plus one final check
    for check_index in range(
        1,
        int(max_iterations) + 2,
    ):

        result = current_mf.stability(
            internal=True,
            external=False,
            return_status=True,
        )

        mo_internal = result[0]

        stable_internal = bool(
            result[2]
        )

        if initially_stable is None:
            initially_stable = (
                stable_internal
            )

        energy_before = float(
            current_mf.e_tot
        )

        if stable_internal:

            history.append(
                StabilityIteration(
                    iteration=check_index,
                    energy_before_hartree=
                        energy_before,
                    stable_internal=True,
                    energy_after_hartree=
                        energy_before,
                    delta_energy_meV=0.0,
                    scf_path=(
                        "stable_check"
                    ),
                )
            )

            finally_stable = True
            break

        # We have already used all permitted reoptimizations.
        # Record the final failed check, but do not perform a seventh.
        if (
            reoptimizations
            >= int(max_iterations)
        ):

            history.append(
                StabilityIteration(
                    iteration=check_index,
                    energy_before_hartree=
                        energy_before,
                    stable_internal=False,
                    energy_after_hartree=None,
                    delta_energy_meV=None,
                    scf_path=(
                        "reoptimization_limit_reached"
                    ),
                )
            )

            break

        dm0 = current_mf.make_rdm1(
            mo_coeff=mo_internal,
            mo_occ=current_mf.mo_occ,
        )

        outcome = run_uks(
            current_mf.mol,
            settings,
            dm0=np.asarray(dm0),
        )

        if not outcome.converged:

            history.append(
                StabilityIteration(
                    iteration=check_index,
                    energy_before_hartree=
                        energy_before,
                    stable_internal=False,
                    energy_after_hartree=None,
                    delta_energy_meV=None,
                    scf_path=str(
                        outcome.scf_path
                    ),
                )
            )

            break

        energy_after = float(
            outcome.energy_hartree
        )

        delta_meV = (
            energy_after
            - energy_before
        ) * HARTREE_TO_EV * 1000.0

        history.append(
            StabilityIteration(
                iteration=check_index,
                energy_before_hartree=
                    energy_before,
                stable_internal=False,
                energy_after_hartree=
                    energy_after,
                delta_energy_meV=
                    float(delta_meV),
                scf_path=str(
                    outcome.scf_path
                ),
            )
        )

        current_mf = outcome.mf

        reoptimizations += 1

    final_energy = float(
        current_mf.e_tot
    )

    return StabilityResult(
        initial_mf=initial_mf,
        final_mf=current_mf,

        initially_stable=bool(
            initially_stable
        ),

        finally_stable=bool(
            finally_stable
        ),

        iterations=len(history),
        reoptimizations=
            int(reoptimizations),

        initial_energy_hartree=
            initial_energy,

        final_energy_hartree=
            final_energy,

        total_delta_energy_meV=float(
            (
                final_energy
                - initial_energy
            )
            * HARTREE_TO_EV
            * 1000.0
        ),

        history=history,
    )
