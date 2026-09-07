from dataclasses import dataclass

from .coarse_branch import (
    extend_branch_points,
)


@dataclass(frozen=True)
class ExpansionPolicy:
    step_A: float = 0.10
    block_width_A: float = 0.50

    lower_limit_A: float = 0.60
    upper_limit_A: float = 6.00

    guard_points: int = 2
    max_rounds: int = 20


@dataclass
class ExpansionResult:
    points: list

    status: str
    rounds: int

    lower_extensions: int
    upper_extensions: int

    expanded: bool

    minimum_r_A: float | None
    minimum_energy_hartree: float | None


def converged_points(
    points,
):
    return sorted(
        [
            p for p in points
            if (
                p.converged
                and p.energy_hartree is not None
            )
        ],
        key=lambda p: p.r_A,
    )


def minimum_guard_status(
    points,
    *,
    guard_points: int = 2,
):
    """
    Determine whether the minimum has enough converged
    coarse-grid points on both sides.
    """

    good = converged_points(
        points
    )

    if not good:
        return {
            "minimum": None,
            "need_lower": False,
            "need_upper": False,
            "left_points": 0,
            "right_points": 0,
        }

    min_index = min(
        range(len(good)),
        key=lambda i:
            good[i].energy_hartree,
    )

    minimum = good[
        min_index
    ]

    left = min_index
    right = (
        len(good)
        - 1
        - min_index
    )

    return {
        "minimum": minimum,

        "need_lower":
            left < int(guard_points),

        "need_upper":
            right < int(guard_points),

        "left_points":
            left,

        "right_points":
            right,
    }


def _block_values(
    *,
    boundary: float,
    direction: str,
    policy: ExpansionPolicy,
):
    n = max(
        1,
        int(
            round(
                policy.block_width_A
                / policy.step_A
            )
        ),
    )

    values = []

    for i in range(
        1,
        n + 1,
    ):

        if direction == "lower":

            value = (
                boundary
                - i * policy.step_A
            )

            if (
                value
                >= policy.lower_limit_A
                - 1.0e-10
            ):
                values.append(
                    round(value, 10)
                )

        else:

            value = (
                boundary
                + i * policy.step_A
            )

            if (
                value
                <= policy.upper_limit_A
                + 1.0e-10
            ):
                values.append(
                    round(value, 10)
                )

    return values


def adaptive_expand_branch(
    *,
    points,
    atom: str,
    ligand: str,
    charge: int,
    spin: int,
    basis: str,
    settings,
    max_memory_mb: int = 2000,
    policy: ExpansionPolicy =
        ExpansionPolicy(),
):
    """
    Expand a coarse state-followed PEC until its minimum has
    sufficient coarse-grid support on both sides.

    If a required physical/grid limit is reached first, return
    QC_FAIL_NO_INTERIOR_MINIMUM.

    If propagation cannot advance, return QC_FAIL_SCF.
    """

    current = list(
        points
    )

    lower_extensions = 0
    upper_extensions = 0

    for round_number in range(
        policy.max_rounds + 1
    ):

        status = minimum_guard_status(
            current,
            guard_points=
                policy.guard_points,
        )

        minimum = status[
            "minimum"
        ]

        if minimum is None:

            return ExpansionResult(
                points=current,
                status="QC_FAIL_SCF",
                rounds=round_number,
                lower_extensions=
                    lower_extensions,
                upper_extensions=
                    upper_extensions,
                expanded=(
                    lower_extensions > 0
                    or upper_extensions > 0
                ),
                minimum_r_A=None,
                minimum_energy_hartree=None,
            )

        if (
            not status["need_lower"]
            and not status["need_upper"]
        ):

            return ExpansionResult(
                points=current,
                status="PASS",
                rounds=round_number,
                lower_extensions=
                    lower_extensions,
                upper_extensions=
                    upper_extensions,
                expanded=(
                    lower_extensions > 0
                    or upper_extensions > 0
                ),
                minimum_r_A=
                    minimum.r_A,
                minimum_energy_hartree=
                    minimum.energy_hartree,
            )

        good_before = (
            converged_points(
                current
            )
        )

        old_min_r = good_before[
            0
        ].r_A

        old_max_r = good_before[
            -1
        ].r_A

        progressed = False


        # ------------------------------------------------------
        # Lower-R extension
        # ------------------------------------------------------

        if status["need_lower"]:

            new_values = _block_values(
                boundary=old_min_r,
                direction="lower",
                policy=policy,
            )

            if new_values:

                current = (
                    extend_branch_points(
                        points=current,
                        atom=atom,
                        ligand=ligand,
                        charge=charge,
                        spin=spin,
                        basis=basis,
                        settings=settings,
                        new_r_values=
                            new_values,
                        direction="lower",
                        max_memory_mb=
                            max_memory_mb,
                    )
                )

                good_after = (
                    converged_points(
                        current
                    )
                )

                if (
                    good_after
                    and good_after[0].r_A
                    < old_min_r - 1.0e-8
                ):
                    lower_extensions += 1
                    progressed = True


        # ------------------------------------------------------
        # Upper-R extension
        # ------------------------------------------------------

        status = minimum_guard_status(
            current,
            guard_points=
                policy.guard_points,
        )

        if status["need_upper"]:

            good_now = (
                converged_points(
                    current
                )
            )

            current_max = (
                good_now[-1].r_A
            )

            new_values = _block_values(
                boundary=current_max,
                direction="upper",
                policy=policy,
            )

            if new_values:

                current = (
                    extend_branch_points(
                        points=current,
                        atom=atom,
                        ligand=ligand,
                        charge=charge,
                        spin=spin,
                        basis=basis,
                        settings=settings,
                        new_r_values=
                            new_values,
                        direction="upper",
                        max_memory_mb=
                            max_memory_mb,
                    )
                )

                good_after = (
                    converged_points(
                        current
                    )
                )

                if (
                    good_after
                    and good_after[-1].r_A
                    > current_max + 1.0e-8
                ):
                    upper_extensions += 1
                    progressed = True


        if not progressed:

            final = (
                minimum_guard_status(
                    current,
                    guard_points=
                        policy.guard_points,
                )
            )

            minimum = final[
                "minimum"
            ]

            good = (
                converged_points(
                    current
                )
            )

            # If we physically reached a configured R limit,
            # this is a no-interior-minimum condition.
            hit_lower = (
                good
                and good[0].r_A
                <= policy.lower_limit_A
                + 1.0e-8
            )

            hit_upper = (
                good
                and good[-1].r_A
                >= policy.upper_limit_A
                - 1.0e-8
            )

            fail_status = (
                "QC_FAIL_NO_INTERIOR_MINIMUM"
                if (
                    hit_lower
                    or hit_upper
                )
                else
                "QC_FAIL_SCF"
            )

            return ExpansionResult(
                points=current,
                status=fail_status,
                rounds=round_number,
                lower_extensions=
                    lower_extensions,
                upper_extensions=
                    upper_extensions,
                expanded=(
                    lower_extensions > 0
                    or upper_extensions > 0
                ),
                minimum_r_A=(
                    None
                    if minimum is None
                    else minimum.r_A
                ),
                minimum_energy_hartree=(
                    None
                    if minimum is None
                    else minimum.energy_hartree
                ),
            )


    final = minimum_guard_status(
        current,
        guard_points=
            policy.guard_points,
    )

    minimum = final[
        "minimum"
    ]

    return ExpansionResult(
        points=current,
        status=
            "QC_FAIL_NO_INTERIOR_MINIMUM",
        rounds=
            policy.max_rounds,
        lower_extensions=
            lower_extensions,
        upper_extensions=
            upper_extensions,
        expanded=(
            lower_extensions > 0
            or upper_extensions > 0
        ),
        minimum_r_A=(
            None
            if minimum is None
            else minimum.r_A
        ),
        minimum_energy_hartree=(
            None
            if minimum is None
            else minimum.energy_hartree
        ),
    )
