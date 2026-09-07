from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .molecule import allowed_spins


@dataclass(frozen=True)
class DiscoveryPolicy:
    seed_spacing_A: float = 0.20
    seed_margin_A: float = 0.20

    initial_spin_sectors: int = 5
    max_spin_sectors: int = 8

    spin_tail_length: int = 3


@dataclass
class SpinFrontierAssessment:
    closed: bool
    physical_spin_space_exhausted: bool

    tail_spins: List[int]

    failing_seed_r_values: List[float]
    missing_seed_r_values: List[float]


def make_scout_seed_grid(
    coarse_r_values: Sequence[float],
    *,
    policy: DiscoveryPolicy = DiscoveryPolicy(),
) -> List[float]:
    """
    Construct universal scout geometries from an existing uniform
    coarse grid.

    Scout geometries are guaranteed to be actual coarse-grid points.
    """

    grid = sorted(
        {
            round(float(r), 10)
            for r in coarse_r_values
        }
    )

    if len(grid) < 3:
        raise ValueError(
            "Coarse grid requires at least three points."
        )

    steps = [
        grid[i + 1] - grid[i]
        for i in range(len(grid) - 1)
    ]

    coarse_step = steps[0]

    if coarse_step <= 0.0:
        raise ValueError(
            "Invalid coarse-grid spacing."
        )

    if any(
        abs(step - coarse_step) > 1.0e-8
        for step in steps
    ):
        raise ValueError(
            "Scout-grid construction currently requires "
            "a uniform coarse grid."
        )


    seed_stride = round(
        policy.seed_spacing_A
        / coarse_step
    )

    margin_steps = round(
        policy.seed_margin_A
        / coarse_step
    )


    if seed_stride < 1:
        raise ValueError(
            "Seed spacing is smaller than coarse-grid spacing."
        )

    if (
        abs(
            seed_stride * coarse_step
            - policy.seed_spacing_A
        )
        > 1.0e-8
    ):
        raise ValueError(
            "Seed spacing must be an integer multiple "
            "of coarse-grid spacing."
        )

    if (
        abs(
            margin_steps * coarse_step
            - policy.seed_margin_A
        )
        > 1.0e-8
    ):
        raise ValueError(
            "Seed margin must be an integer multiple "
            "of coarse-grid spacing."
        )


    start = margin_steps
    stop = len(grid) - margin_steps

    seeds = [
        grid[index]
        for index in range(
            start,
            stop,
            seed_stride,
        )
    ]

    if not seeds:
        raise ValueError(
            "Discovery policy produced no scout seeds."
        )

    return seeds


def discovery_spin_values(
    *,
    atom: str,
    ligand: str,
    charge: int,
    policy: DiscoveryPolicy = DiscoveryPolicy(),
) -> List[int]:
    """
    Return up to max_spin_sectors physically allowed 2S values.

    A generous spin_max is supplied to allowed_spins; that routine
    remains responsible for electron-count/parity validity.
    """

    spin_ceiling = (
        2 * policy.max_spin_sectors
        + 1
    )

    values = allowed_spins(
        atom,
        ligand,
        charge,
        spin_ceiling,
    )

    return list(
        values[
            : policy.max_spin_sectors
        ]
    )


def assess_spin_frontier(
    *,
    scanned_spins: Sequence[int],

    minima_by_seed:
        Dict[
            float,
            Dict[int, float],
        ],

    all_allowed_spins:
        Sequence[int],

    tail_length: int = 3,
) -> SpinFrontierAssessment:
    """
    Threshold-free high-spin stopping criterion.

    The frontier is closed when either:

    A) every physically allowed spin sector has been scanned, or

    B) at every seed the minima of the highest `tail_length`
       scanned spin sectors rise strictly with increasing spin.

    Missing data in any required tail sector prevents closure.
    """

    scanned = sorted(
        set(
            int(x)
            for x in scanned_spins
        )
    )

    allowed = sorted(
        set(
            int(x)
            for x in all_allowed_spins
        )
    )


    if not scanned:
        return SpinFrontierAssessment(
            closed=False,
            physical_spin_space_exhausted=False,
            tail_spins=[],
            failing_seed_r_values=[],
            missing_seed_r_values=
                sorted(
                    float(r)
                    for r
                    in minima_by_seed
                ),
        )


    exhausted = (
        scanned == allowed
    )


    if exhausted:

        return SpinFrontierAssessment(
            closed=True,
            physical_spin_space_exhausted=True,

            tail_spins=
                scanned[
                    -min(
                        len(scanned),
                        int(tail_length),
                    ):
                ],

            failing_seed_r_values=[],
            missing_seed_r_values=[],
        )


    if len(scanned) < int(tail_length):

        return SpinFrontierAssessment(
            closed=False,
            physical_spin_space_exhausted=False,
            tail_spins=scanned,
            failing_seed_r_values=[],
            missing_seed_r_values=
                sorted(
                    float(r)
                    for r
                    in minima_by_seed
                ),
        )


    tail = scanned[
        -int(tail_length):
    ]

    failing = []
    missing = []


    for seed_r, spin_map in sorted(
        minima_by_seed.items()
    ):

        if any(
            spin not in spin_map
            for spin in tail
        ):

            missing.append(
                float(seed_r)
            )

            continue


        energies = [
            float(
                spin_map[spin]
            )
            for spin in tail
        ]


        rising = all(
            energies[i + 1]
            > energies[i]
            for i in range(
                len(energies) - 1
            )
        )


        if not rising:

            failing.append(
                float(seed_r)
            )


    closed = (
        not failing
        and not missing
    )


    return SpinFrontierAssessment(
        closed=closed,
        physical_spin_space_exhausted=False,
        tail_spins=tail,
        failing_seed_r_values=failing,
        missing_seed_r_values=missing,
    )
