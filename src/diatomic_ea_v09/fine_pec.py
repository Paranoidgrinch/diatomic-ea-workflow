from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .coarse_branch import (
    BranchPoint,
    follow_scout_solution,
)
from .scf import HARTREE_TO_EV


@dataclass(frozen=True)
class FineGridPolicy:
    half_width_A: float = 0.20
    step_A: float = 0.01
    quadratic_half_points: int = 3


@dataclass
class FinePECSummary:
    points: list

    discrete_min_r_A: float
    discrete_min_energy_hartree: float

    fitted_min_r_A: float | None
    fitted_min_energy_hartree: float | None

    curvature_hartree_per_A2: float | None
    quadratic_points_used: int

    minimum_at_boundary: bool

    # Exact interval actually used by local_quadratic_fit().
    # Defaults preserve compatibility with development checkpoints
    # pickled before these fields existed.
    quadratic_r_low_A: float | None = None
    quadratic_r_high_A: float | None = None


def make_fine_grid(
    center_r_A: float,
    *,
    half_width_A: float = 0.20,
    step_A: float = 0.01,
):
    n = int(
        round(
            half_width_A
            / step_A
        )
    )

    return [
        round(
            center_r_A
            + i * step_A,
            10,
        )
        for i in range(
            -n,
            n + 1,
        )
    ]


def converged_fine_points(
    points: Sequence[BranchPoint],
):
    return sorted(
        [
            p
            for p in points
            if (
                p.converged
                and p.energy_hartree
                is not None
            )
        ],
        key=lambda p: p.r_A,
    )


def local_quadratic_fit(
    points,
    *,
    half_points: int = 3,
):
    """
    Development-only local quadratic diagnostic.

    Uses 2*half_points+1 points centered on the discrete minimum,
    when available.

    Final vibrational/ZPE fit policy is decided later.
    """

    good = converged_fine_points(
        points
    )

    if len(good) < 3:
        return None

    minimum_index = min(
        range(len(good)),
        key=lambda i:
            good[i].energy_hartree,
    )

    lo = max(
        0,
        minimum_index - half_points,
    )

    hi = min(
        len(good),
        minimum_index
        + half_points
        + 1,
    )

    subset = good[
        lo:hi
    ]

    if len(subset) < 3:
        return None

    r = np.array(
        [
            p.r_A
            for p in subset
        ],
        dtype=float,
    )

    e = np.array(
        [
            p.energy_hartree
            for p in subset
        ],
        dtype=float,
    )

    # Centering improves numerical conditioning.
    r0 = float(
        np.mean(r)
    )

    x = r - r0

    a, b, c = np.polyfit(
        x,
        e,
        2,
    )

    if a <= 0.0:
        return {
            "valid": False,
            "n_points":
                len(subset),
        }

    x_min = (
        -b / (2.0 * a)
    )

    r_min = (
        r0 + x_min
    )

    e_min = (
        a * x_min**2
        + b * x_min
        + c
    )

    return {
        "valid": True,

        "r_min_A":
            float(r_min),

        "energy_min_hartree":
            float(e_min),

        # E = a*x^2 + ...
        # second derivative = 2a
        "curvature_hartree_per_A2":
            float(2.0 * a),

        "n_points":
            len(subset),

        "r_low_A":
            float(min(r)),

        "r_high_A":
            float(max(r)),
    }


def run_fine_pec(
    *,
    branch_id: str,

    atom: str,
    ligand: str,

    seed,

    settings,

    center_r_A: float,

    policy: FineGridPolicy =
        FineGridPolicy(),

    max_memory_mb: int = 2000,
):
    """
    State-follow one already canonicalized electronic state through
    a dense local PEC window.

    The seed must represent the canonical state at center_r_A.
    """

    if (
        abs(
            float(seed.r_A)
            - float(center_r_A)
        )
        > 1.0e-8
    ):
        raise ValueError(
            "Fine PEC seed geometry must equal "
            "the requested center geometry."
        )

    grid = make_fine_grid(
        center_r_A,

        half_width_A=
            policy.half_width_A,

        step_A=
            policy.step_A,
    )

    points = follow_scout_solution(
        branch_id=branch_id,

        atom=atom,
        ligand=ligand,

        seed=seed,

        settings=settings,

        r_values=grid,

        max_memory_mb=
            max_memory_mb,
    )

    good = converged_fine_points(
        points
    )

    if not good:
        raise RuntimeError(
            "Fine PEC contains no converged points."
        )

    minimum = min(
        good,
        key=lambda p:
            p.energy_hartree,
    )

    boundary = (
        abs(
            minimum.r_A
            - good[0].r_A
        ) < 1.0e-8
        or
        abs(
            minimum.r_A
            - good[-1].r_A
        ) < 1.0e-8
    )

    fit = local_quadratic_fit(
        good,
        half_points=
            policy.quadratic_half_points,
    )

    return FinePECSummary(
        points=points,

        discrete_min_r_A=
            minimum.r_A,

        discrete_min_energy_hartree=
            minimum.energy_hartree,

        fitted_min_r_A=(
            None
            if (
                fit is None
                or not fit.get(
                    "valid",
                    False,
                )
            )
            else fit["r_min_A"]
        ),

        fitted_min_energy_hartree=(
            None
            if (
                fit is None
                or not fit.get(
                    "valid",
                    False,
                )
            )
            else fit[
                "energy_min_hartree"
            ]
        ),

        curvature_hartree_per_A2=(
            None
            if (
                fit is None
                or not fit.get(
                    "valid",
                    False,
                )
            )
            else fit[
                "curvature_hartree_per_A2"
            ]
        ),

        quadratic_points_used=(
            0
            if fit is None
            else int(
                fit.get(
                    "n_points",
                    0,
                )
            )
        ),

        minimum_at_boundary=
            bool(boundary),

        quadratic_r_low_A=(
            None
            if fit is None
            else fit.get(
                "r_low_A"
            )
        ),

        quadratic_r_high_A=(
            None
            if fit is None
            else fit.get(
                "r_high_A"
            )
        ),
    )
