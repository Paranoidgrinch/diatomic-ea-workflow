from dataclasses import dataclass, replace
from typing import List

from .adaptive_grid import (
    ExpansionPolicy,
    adaptive_expand_branch,
    minimum_guard_status,
)
from .ground_state import (
    CanonicalBranch,
    branch_minimum,
    canonicalize_branch_at_minimum,
)


@dataclass
class FinalizationCycle:
    cycle: int

    expanded: bool
    expansion_status: str
    expansion_rounds: int

    pre_stability_min_r_A: float
    pre_stability_min_energy_hartree: float

    initially_stable: bool | None
    finally_stable: bool | None
    stability_reoptimizations: int | None
    stability_delta_energy_meV: float | None

    post_stability_min_r_A: float | None
    post_stability_min_energy_hartree: float | None

    minimum_shift_A: float | None


@dataclass
class FinalizedBranch:
    source: CanonicalBranch

    points: list

    valid: bool
    status: str

    cycles: int
    history: List[FinalizationCycle]

    @property
    def minimum(self):
        if not self.points:
            return None

        return branch_minimum(
            self.points
        )


def finalize_ground_branch(
    *,
    branch: CanonicalBranch,

    atom: str,
    ligand: str,
    basis: str,

    settings,

    max_memory_mb: int = 2000,

    expansion_policy: ExpansionPolicy =
        ExpansionPolicy(),

    max_cycles: int = 6,

    max_stability_iterations: int = 6,
):
    """
    Close the coarse ground-state loop:

      canonical branch
          -> adaptive R expansion
          -> stability at the resulting minimum
          -> repropagate stabilized state
          -> repeat if the minimum moved

    PASS requires:
    - successful expansion
    - final internal stability
    - an interior coarse-grid minimum
    - stability performed at the same grid point that remains
      the minimum after repropagation
    """

    if not branch.valid:

        return FinalizedBranch(
            source=branch,
            points=[],
            valid=False,
            status=branch.status,
            cycles=0,
            history=[],
        )

    current_points = list(
        branch.canonical_points
    )

    history = []

    for cycle in range(
        1,
        int(max_cycles) + 1,
    ):

        # ------------------------------------------------------
        # 1. Expand until coarse minimum is interior
        # ------------------------------------------------------

        expansion = adaptive_expand_branch(
            points=current_points,

            atom=atom,
            ligand=ligand,

            charge=
                branch.source_branch.charge,

            spin=
                branch.source_branch.spin,

            basis=basis,

            settings=settings,

            max_memory_mb=
                max_memory_mb,

            policy=
                expansion_policy,
        )

        if expansion.status != "PASS":

            return FinalizedBranch(
                source=branch,
                points=
                    expansion.points,
                valid=False,
                status=
                    expansion.status,
                cycles=cycle,
                history=history,
            )

        expanded_points = list(
            expansion.points
        )

        pre_min = branch_minimum(
            expanded_points
        )


        # ------------------------------------------------------
        # 2. Re-run universal ground-state stability at the
        #    current expanded coarse minimum.
        #
        #    canonicalize_branch_at_minimum already:
        #      - reconstructs from the branch density
        #      - runs stability
        #      - creates stabilized seed
        #      - repropagates whole supplied grid
        # ------------------------------------------------------

        temporary_source = replace(
            branch.source_branch,
            raw_points=
                expanded_points,
        )

        r_grid = sorted(
            {
                round(
                    p.r_A,
                    10,
                )
                for p
                in expanded_points
            }
        )

        recanonicalized = (
            canonicalize_branch_at_minimum(
                branch=
                    temporary_source,

                atom=atom,
                ligand=ligand,

                basis=basis,

                settings=settings,

                coarse_r_values=
                    r_grid,

                max_memory_mb=
                    max_memory_mb,

                max_stability_iterations=
                    max_stability_iterations,
            )
        )

        if not recanonicalized.valid:

            stability = (
                recanonicalized
                .stability_result
            )

            history.append(
                FinalizationCycle(
                    cycle=cycle,

                    expanded=
                        expansion.expanded,

                    expansion_status=
                        expansion.status,

                    expansion_rounds=
                        expansion.rounds,

                    pre_stability_min_r_A=
                        pre_min.r_A,

                    pre_stability_min_energy_hartree=
                        pre_min.energy_hartree,

                    initially_stable=(
                        None
                        if stability is None
                        else
                        stability.initially_stable
                    ),

                    finally_stable=(
                        None
                        if stability is None
                        else
                        stability.finally_stable
                    ),

                    stability_reoptimizations=(
                        None
                        if stability is None
                        else
                        stability.reoptimizations
                    ),

                    stability_delta_energy_meV=(
                        None
                        if stability is None
                        else
                        stability.total_delta_energy_meV
                    ),

                    post_stability_min_r_A=None,
                    post_stability_min_energy_hartree=None,
                    minimum_shift_A=None,
                )
            )

            return FinalizedBranch(
                source=branch,
                points=[],
                valid=False,
                status=
                    recanonicalized.status,
                cycles=cycle,
                history=history,
            )


        post_points = list(
            recanonicalized
            .canonical_points
        )

        post_min = branch_minimum(
            post_points
        )

        stability = (
            recanonicalized
            .stability_result
        )

        shift_A = (
            post_min.r_A
            - pre_min.r_A
        )


        history.append(
            FinalizationCycle(
                cycle=cycle,

                expanded=
                    expansion.expanded,

                expansion_status=
                    expansion.status,

                expansion_rounds=
                    expansion.rounds,

                pre_stability_min_r_A=
                    pre_min.r_A,

                pre_stability_min_energy_hartree=
                    pre_min.energy_hartree,

                initially_stable=
                    stability.initially_stable,

                finally_stable=
                    stability.finally_stable,

                stability_reoptimizations=
                    stability.reoptimizations,

                stability_delta_energy_meV=
                    stability.total_delta_energy_meV,

                post_stability_min_r_A=
                    post_min.r_A,

                post_stability_min_energy_hartree=
                    post_min.energy_hartree,

                minimum_shift_A=
                    shift_A,
            )
        )


        # ------------------------------------------------------
        # 3. Is the post-stability minimum still safely interior?
        # ------------------------------------------------------

        guard = minimum_guard_status(
            post_points,
            guard_points=
                expansion_policy.guard_points,
        )

        interior = (
            not guard["need_lower"]
            and not guard["need_upper"]
        )


        # ------------------------------------------------------
        # 4. Converged finalization condition
        #
        # Stability was performed at pre_min.
        # If repropagation leaves the minimum at exactly that
        # coarse-grid point, the minimum itself has now been
        # explicitly stability-checked.
        # ------------------------------------------------------

        same_minimum_point = (
            abs(
                post_min.r_A
                - pre_min.r_A
            )
            < 1.0e-8
        )

        if (
            stability.finally_stable
            and interior
            and same_minimum_point
        ):

            return FinalizedBranch(
                source=branch,
                points=
                    post_points,
                valid=True,
                status="PASS",
                cycles=cycle,
                history=history,
            )


        # Minimum moved after stability or became a new boundary.
        # Feed the stabilized PEC into the next universal cycle.
        current_points = (
            post_points
        )


    return FinalizedBranch(
        source=branch,
        points=current_points,
        valid=False,
        status=
            "QC_FAIL_STABILITY",
        cycles=
            int(max_cycles),
        history=history,
    )
