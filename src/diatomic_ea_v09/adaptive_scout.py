from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .candidate_pool import (
    CandidateGroup,
    build_candidate_groups,
)
from .discovery_policy import (
    DiscoveryPolicy,
    SpinFrontierAssessment,
    assess_spin_frontier,
    make_scout_seed_grid,
)
from .model import (
    MoleculeSpec,
    SCFSettings,
)
from .molecule import (
    allowed_spins,
)
from .state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


@dataclass
class AdaptiveScoutSectorRecord:
    seed_r_A: float
    spin: int

    n_raw_solutions: int
    n_candidate_groups: int

    minimum_energy_hartree: float | None

    failed_guesses: List[str]


@dataclass
class AdaptiveScoutResult:
    molecule: str
    charge: int

    status: str

    seed_r_values: List[float]

    physical_spin_values: List[int]
    development_spin_values: List[int]
    scanned_spin_values: List[int]

    frontier: SpinFrontierAssessment

    groups_by_seed: Dict[
        float,
        List[CandidateGroup],
    ]

    sector_records: List[
        AdaptiveScoutSectorRecord
    ]

    @property
    def spin_max(self) -> int:
        if not self.scanned_spin_values:
            raise ValueError(
                "Adaptive scout has no scanned spin sectors."
            )

        return max(
            self.scanned_spin_values
        )


def full_physical_spin_values(
    *,
    atom: str,
    ligand: str,
    charge: int,
) -> List[int]:
    """
    Obtain the full physically allowed 2S list.

    The deliberately generous ceiling is only an API bound.
    Electron number/parity in allowed_spins determines the
    actual physical list.
    """

    values = allowed_spins(
        atom,
        ligand,
        charge,
        999,
    )

    return list(values)


def development_spin_values(
    *,
    atom: str,
    ligand: str,
    charge: int,
    policy: DiscoveryPolicy,
) -> tuple[
    List[int],
    List[int],
]:
    """
    Return:
        full physical spin list,
        spin sectors allowed for development search.

    The development list is capped independently of whether
    the physical spin space is larger.
    """

    physical = full_physical_spin_values(
        atom=atom,
        ligand=ligand,
        charge=charge,
    )

    development = physical[
        : policy.max_spin_sectors
    ]

    return (
        physical,
        development,
    )


def scout_spin_with_failures(
    *,
    spec: MoleculeSpec,
    settings: SCFSettings,
    guesses: Sequence[str],
):
    """
    Run all requested initial guesses for one fixed (R, spin) sector.

    A PySCF RuntimeError raised by one individual initial guess does
    not invalidate the entire spin sector. The failing guess is
    recorded and the remaining guesses are still attempted.

    Other exception types remain fatal so programming/API errors are
    not silently hidden.
    """

    solutions = []
    failed_guesses = []

    for guess in guesses:

        try:
            solution = run_guess(
                spec,
                settings,
                guess,
            )

        except RuntimeError as exc:

            failed_guesses.append(
                "{}: {}: {}".format(
                    guess,
                    type(exc).__name__,
                    str(exc),
                )
            )

            continue


        if solution is not None:
            solutions.append(
                solution
            )


    return (
        solutions,
        failed_guesses,
    )


def run_adaptive_local_scout(
    *,
    atom: str,
    ligand: str,

    charge: int,

    coarse_r_values:
        Sequence[float],

    basis: str,

    settings: SCFSettings,

    policy:
        DiscoveryPolicy =
        DiscoveryPolicy(),

    max_memory_mb: int = 2000,

    guesses:
        Sequence[str] =
        DEFAULT_GUESSES,

    progress_callback=None,
) -> AdaptiveScoutResult:
    """
    Adaptive local state discovery.

    Critically, the resulting CandidateGroups are RETAINED and
    are intended to be passed directly into branch discovery.

    Already calculated seed/spin sectors are never repeated while
    expanding the spin frontier.
    """

    molecule = (
        atom + ligand
    ).upper()

    seeds = make_scout_seed_grid(
        coarse_r_values,
        policy=policy,
    )

    (
        physical_spins,
        available_spins,
    ) = development_spin_values(
        atom=atom,
        ligand=ligand,
        charge=charge,
        policy=policy,
    )


    if not available_spins:

        raise RuntimeError(
            "No physically allowed spin sectors."
        )


    n_scan = min(
        policy.initial_spin_sectors,
        len(available_spins),
    )


    # Cache at the finest useful level:
    # one fixed geometry + one spin sector.
    sector_cache: Dict[
        Tuple[float, int],
        List[CandidateGroup],
    ] = {}


    sector_records: Dict[
        Tuple[float, int],
        AdaptiveScoutSectorRecord,
    ] = {}


    final_frontier = None
    final_status = None


    while True:

        scanned_spins = (
            available_spins[:n_scan]
        )

        minima_by_seed = {}


        for seed_r in seeds:

            seed_key = round(
                float(seed_r),
                10,
            )

            minima_by_seed[
                seed_key
            ] = {}


            for spin in scanned_spins:

                key = (
                    seed_key,
                    int(spin),
                )


                if key not in sector_cache:

                    spec = MoleculeSpec(
                        atom=atom,
                        ligand=ligand,

                        charge=charge,
                        spin=int(spin),

                        basis=basis,

                        r_A=
                            seed_key,

                        max_memory_mb=
                            max_memory_mb,
                    )


                    (
                        solutions,
                        failed_guesses,
                    ) = scout_spin_with_failures(
                        spec=spec,
                        settings=settings,
                        guesses=guesses,
                    )


                    # The density and scalar descriptors are what
                    # subsequent branch following requires.
                    #
                    # Drop the PySCF mf object to keep the adaptive
                    # cache much smaller in memory.
                    for solution in solutions:
                        solution.mf = None


                    groups = (
                        build_candidate_groups(
                            solutions
                        )
                        if solutions
                        else []
                    )


                    sector_cache[
                        key
                    ] = groups


                    minimum = (
                        None
                        if not groups
                        else min(
                            group.representative
                            .energy_hartree
                            for group in groups
                        )
                    )


                    sector_records[
                        key
                    ] = (
                        AdaptiveScoutSectorRecord(
                            seed_r_A=
                                seed_key,

                            spin=
                                int(spin),

                            n_raw_solutions=
                                len(solutions),

                            n_candidate_groups=
                                len(groups),

                            minimum_energy_hartree=
                                minimum,

                            failed_guesses=
                                list(
                                    failed_guesses
                                ),
                        )
                    )


                    if progress_callback is not None:

                        progress_callback(
                            "SCOUT "
                            "R={:.2f} "
                            "2S={} "
                            "solutions={} "
                            "groups={} "
                            "failed_guesses={}".format(
                                seed_key,
                                int(spin),
                                len(solutions),
                                len(groups),
                                len(failed_guesses),
                            )
                        )


                groups = (
                    sector_cache[key]
                )


                if groups:

                    minima_by_seed[
                        seed_key
                    ][int(spin)] = min(
                        group.representative
                        .energy_hartree
                        for group in groups
                    )


        if progress_callback is not None:

            progress_callback(
                "FRONTIER_CHECK scanned_spins={}".format(
                    list(scanned_spins)
                )
            )


        frontier = assess_spin_frontier(
            scanned_spins=
                scanned_spins,

            minima_by_seed=
                minima_by_seed,

            # IMPORTANT:
            # full physical list, not the development-truncated list.
            all_allowed_spins=
                physical_spins,

            tail_length=
                policy.spin_tail_length,
        )


        if progress_callback is not None:

            progress_callback(
                "FRONTIER_RESULT "
                "closed={} "
                "tail={} "
                "failing={} "
                "missing={}".format(
                    frontier.closed,
                    frontier.tail_spins,
                    frontier.failing_seed_r_values,
                    frontier.missing_seed_r_values,
                )
            )


        if frontier.closed:

            final_frontier = frontier
            final_status = (
                "PASS_SPIN_FRONTIER"
            )

            break


        if n_scan >= len(
            available_spins
        ):

            final_frontier = frontier
            final_status = (
                "QC_FAIL_SPIN_FRONTIER"
            )

            break


        n_scan += 1


    final_spins = (
        available_spins[:n_scan]
    )


    groups_by_seed = {}


    for seed_r in seeds:

        seed_key = round(
            float(seed_r),
            10,
        )

        combined = []


        for spin in final_spins:

            combined.extend(
                sector_cache.get(
                    (
                        seed_key,
                        int(spin),
                    ),
                    [],
                )
            )


        groups_by_seed[
            seed_key
        ] = sorted(
            combined,
            key=lambda group:
                (
                    int(group.spin),
                    float(
                        group.representative
                        .energy_hartree
                    ),
                ),
        )


    return AdaptiveScoutResult(
        molecule=molecule,
        charge=charge,

        status=
            final_status,

        seed_r_values=[
            float(r)
            for r in seeds
        ],

        physical_spin_values=
            physical_spins,

        development_spin_values=
            available_spins,

        scanned_spin_values=
            list(final_spins),

        frontier=
            final_frontier,

        groups_by_seed=
            groups_by_seed,

        sector_records=[
            sector_records[key]
            for key in sorted(
                sector_records
            )
        ],
    )
