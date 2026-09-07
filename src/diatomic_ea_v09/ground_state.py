from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .candidate_pool import (
    CandidateGroup,
    build_candidate_groups,
)
from .coarse_branch import (
    BranchPoint,
    follow_scout_solution,
)
from .model import (
    MoleculeSpec,
    SCFSettings,
)
from .molecule import (
    allowed_spins,
    build_molecule,
)
from .scf import (
    HARTREE_TO_EV,
    run_uks,
)
from .stability import (
    StabilityResult,
    optimize_internal_stability,
)
from .state_identity import (
    StateRelation,
    compare_states,
)
from .state_scout import (
    DEFAULT_GUESSES,
    ScoutSolution,
    orthonormal_spin_density,
    run_guess,
)


@dataclass
class SeedDiscoveryRecord:
    seed_r_A: float
    spin: int
    scout_group_id: str
    guesses: List[str]
    energy_hartree: float
    action: str
    matched_branch_id: str | None


@dataclass
class DiscoveredBranch:
    branch_id: str

    charge: int
    spin: int
    multiplicity: int

    source_seed_r_A: float
    source_scout_group_id: str
    source_guesses: List[str]

    seed_solution: ScoutSolution
    raw_points: List[BranchPoint]

    @property
    def raw_minimum(self) -> BranchPoint:
        return branch_minimum(
            self.raw_points
        )


@dataclass
class CanonicalBranch:
    source_branch: DiscoveredBranch

    reconstruction_relation: StateRelation | None
    stability_result: StabilityResult | None
    stability_relation: StateRelation | None

    canonical_points: List[BranchPoint]

    valid: bool
    status: str

    @property
    def branch_id(self):
        return self.source_branch.branch_id

    @property
    def minimum(self):
        if not self.canonical_points:
            return None

        return branch_minimum(
            self.canonical_points
        )


@dataclass
class ChargeGroundStateResult:
    molecule: str
    charge: int

    discovered_branches: List[DiscoveredBranch]
    canonical_branches: List[CanonicalBranch]

    discovery_records: List[SeedDiscoveryRecord]

    ground_state_branch: CanonicalBranch | None


def branch_minimum(
    points: Iterable[BranchPoint],
) -> BranchPoint:

    good = [
        p
        for p in points
        if (
            p.converged
            and p.energy_hartree is not None
        )
    ]

    if not good:
        raise RuntimeError(
            "Branch contains no converged points."
        )

    return min(
        good,
        key=lambda p:
            p.energy_hartree,
    )


def point_at_r(
    points: Iterable[BranchPoint],
    r_A: float,
) -> BranchPoint | None:

    for point in points:

        if (
            point.converged
            and abs(
                point.r_A - float(r_A)
            ) < 1.0e-8
        ):
            return point

    return None


def scout_solution_from_mf(
    *,
    mf,
    template: ScoutSolution,
    origin_guess: str,
) -> ScoutSolution:

    ss, observed_mult = (
        mf.spin_square()
    )

    dm = np.asarray(
        mf.make_rdm1()
    )

    dm_orth = (
        orthonormal_spin_density(
            mf.mol,
            dm,
        )
    )

    mo_energy = np.asarray(
        mf.mo_energy,
        dtype=float,
    )

    mo_occ = np.asarray(
        mf.mo_occ,
        dtype=float,
    )

    occupied = mo_energy[
        mo_occ > 1.0e-8
    ]

    virtual = mo_energy[
        mo_occ <= 1.0e-8
    ]

    homo_eV = (
        float(np.max(occupied))
        * HARTREE_TO_EV
        if occupied.size
        else float("nan")
    )

    lumo_eV = (
        float(np.min(virtual))
        * HARTREE_TO_EV
        if virtual.size
        else float("nan")
    )

    gap_eV = (
        lumo_eV - homo_eV
        if (
            np.isfinite(homo_eV)
            and np.isfinite(lumo_eV)
        )
        else float("nan")
    )

    return ScoutSolution(
        molecule=
            template.molecule,

        charge=
            template.charge,

        spin=
            template.spin,

        multiplicity=
            template.multiplicity,

        r_A=float(
            template.r_A
        ),

        functional=
            template.functional,

        basis=
            template.basis,

        origin_guess=
            origin_guess,

        equivalent_guesses=[
            origin_guess
        ],

        energy_hartree=float(
            mf.e_tot
        ),

        s2=float(ss),

        observed_multiplicity=float(
            observed_mult
        ),

        homo_eV=
            homo_eV,

        lumo_eV=
            lumo_eV,

        gap_eV=
            gap_eV,

        scf_path=
            "stability_canonical",

        density_ao=np.array(
            dm,
            copy=True,
        ),

        density_orth=np.array(
            dm_orth,
            copy=True,
        ),

        mf=mf,
    )


def compare_solution_to_point(
    solution: ScoutSolution,
    point: BranchPoint,
) -> StateRelation:

    comparison = compare_states(
        energy_a_hartree=
            solution.energy_hartree,

        energy_b_hartree=
            point.energy_hartree,

        s2_a=
            solution.s2,

        s2_b=
            point.s2,

        dm_orth_a=
            solution.density_orth,

        dm_orth_b=
            point.density_orth,

        spin_a=
            solution.spin,

        spin_b=
            point.spin,
    )

    return comparison.relation


def local_scout_groups(
    *,
    atom: str,
    ligand: str,
    charge: int,
    r_A: float,
    spin_max: int,
    basis: str,
    settings: SCFSettings,
    max_memory_mb: int,
    guesses: Sequence[str] = DEFAULT_GUESSES,
) -> List[CandidateGroup]:

    spins = allowed_spins(
        atom,
        ligand,
        charge,
        spin_max,
    )

    solutions = []

    for spin in spins:

        for guess in guesses:

            spec = MoleculeSpec(
                atom=atom,
                ligand=ligand,
                charge=charge,
                spin=spin,
                basis=basis,
                r_A=float(r_A),
                max_memory_mb=
                    max_memory_mb,
            )

            try:

                solution = run_guess(
                    spec,
                    settings,
                    guess,
                )

            except Exception:

                # Detailed GUESS_UNAVAILABLE / SCF error
                # separation is added in the QC layer.
                continue

            if solution is None:
                continue

            solution.mf = None

            solutions.append(
                solution
            )

    return build_candidate_groups(
        solutions
    )


def discover_charge_branches(
    *,
    atom: str,
    ligand: str,
    charge: int,
    seed_r_values: Sequence[float],
    coarse_r_values: Sequence[float],
    spin_max: int,
    basis: str,
    settings: SCFSettings,
    max_memory_mb: int = 2000,
    guesses: Sequence[str] = DEFAULT_GUESSES,
    precomputed_groups_by_seed=None,
) -> tuple[
    List[DiscoveredBranch],
    List[SeedDiscoveryRecord],
]:

    molecule = (
        atom + ligand
    ).upper()

    charge_tag = (
        "N"
        if charge == 0
        else "A"
    )

    branches = []
    records = []

    branch_counter = 0

    prepared_groups = None

    if precomputed_groups_by_seed is not None:

        prepared_groups = {
            round(
                float(seed_r),
                10,
            ): list(groups)

            for seed_r, groups
            in precomputed_groups_by_seed.items()
        }


    for seed_r in seed_r_values:

        seed_key = round(
            float(seed_r),
            10,
        )


        if prepared_groups is None:

            groups = local_scout_groups(
                atom=atom,
                ligand=ligand,
                charge=charge,
                r_A=seed_r,
                spin_max=spin_max,
                basis=basis,
                settings=settings,
                max_memory_mb=
                    max_memory_mb,
                guesses=guesses,
            )

        else:

            if seed_key not in prepared_groups:

                raise ValueError(
                    "Missing precomputed scout groups "
                    f"for seed R={seed_key:.10f} A"
                )

            # Keep spin_max semantics identical to the original path.
            groups = [
                group
                for group
                in prepared_groups[
                    seed_key
                ]
                if int(group.spin)
                <= int(spin_max)
            ]


        for group in groups:

            representative = (
                group.representative
            )

            same_matches = []

            for existing in branches:

                existing_point = (
                    point_at_r(
                        existing.raw_points,
                        seed_r,
                    )
                )

                if existing_point is None:
                    continue

                relation = (
                    compare_solution_to_point(
                        representative,
                        existing_point,
                    )
                )

                if (
                    relation
                    == StateRelation.SAME_STATE
                ):
                    same_matches.append(
                        existing.branch_id
                    )

            if len(same_matches) >= 1:

                records.append(
                    SeedDiscoveryRecord(
                        seed_r_A=
                            float(seed_r),

                        spin=
                            representative.spin,

                        scout_group_id=
                            group.scout_group_id,

                        guesses=
                            group.equivalent_guesses,

                        energy_hartree=
                            representative.energy_hartree,

                        action=
                            "MATCH_EXISTING",

                        matched_branch_id=
                            same_matches[0],
                    )
                )

                continue

            branch_counter += 1

            branch_id = (
                f"{molecule}_{charge_tag}_"
                f"DISC{branch_counter:03d}_"
                f"S{representative.spin}"
            )

            points = (
                follow_scout_solution(
                    branch_id=branch_id,
                    atom=atom,
                    ligand=ligand,
                    seed=representative,
                    settings=settings,
                    r_values=
                        coarse_r_values,
                    max_memory_mb=
                        max_memory_mb,
                )
            )

            branch = DiscoveredBranch(
                branch_id=
                    branch_id,

                charge=
                    charge,

                spin=
                    representative.spin,

                multiplicity=
                    representative.multiplicity,

                source_seed_r_A=
                    float(seed_r),

                source_scout_group_id=
                    group.scout_group_id,

                source_guesses=
                    group.equivalent_guesses,

                seed_solution=
                    representative,

                raw_points=
                    points,
            )

            branches.append(
                branch
            )

            records.append(
                SeedDiscoveryRecord(
                    seed_r_A=
                        float(seed_r),

                    spin=
                        representative.spin,

                    scout_group_id=
                        group.scout_group_id,

                    guesses=
                        group.equivalent_guesses,

                    energy_hartree=
                        representative.energy_hartree,

                    action=
                        "NEW_BRANCH",

                    matched_branch_id=None,
                )
            )

    return (
        branches,
        records,
    )


def canonicalize_branch_at_minimum(
    *,
    branch: DiscoveredBranch,
    atom: str,
    ligand: str,
    basis: str,
    settings: SCFSettings,
    coarse_r_values: Sequence[float],
    max_memory_mb: int = 2000,
    max_stability_iterations: int = 6,
) -> CanonicalBranch:

    raw_min = (
        branch.raw_minimum
    )

    spec = MoleculeSpec(
        atom=atom,
        ligand=ligand,
        charge=branch.charge,
        spin=branch.spin,
        basis=basis,
        r_A=raw_min.r_A,
        max_memory_mb=
            max_memory_mb,
    )

    mol = build_molecule(
        spec
    )

    reconstruction = run_uks(
        mol,
        settings,
        dm0=raw_min.density_ao,
    )

    if not reconstruction.converged:

        return CanonicalBranch(
            source_branch=
                branch,

            reconstruction_relation=None,

            stability_result=None,

            stability_relation=None,

            canonical_points=[],

            valid=False,

            status=
                "QC_FAIL_SCF_RECONSTRUCTION",
        )

    reconstructed = (
        scout_solution_from_mf(
            mf=reconstruction.mf,
            template=ScoutSolution(
                molecule=
                    branch.seed_solution.molecule,

                charge=
                    branch.charge,

                spin=
                    branch.spin,

                multiplicity=
                    branch.multiplicity,

                r_A=
                    raw_min.r_A,

                functional=
                    settings.xc,

                basis=
                    basis,

                origin_guess=
                    "branch_reconstruction",

                equivalent_guesses=[
                    "branch_reconstruction"
                ],

                energy_hartree=
                    reconstruction.energy_hartree,

                s2=
                    reconstruction.s2,

                observed_multiplicity=
                    reconstruction.observed_multiplicity,

                homo_eV=
                    reconstruction.homo_eV,

                lumo_eV=
                    reconstruction.lumo_eV,

                gap_eV=
                    reconstruction.gap_eV,

                scf_path=
                    reconstruction.scf_path,

                density_ao=
                    np.asarray(
                        reconstruction.mf.make_rdm1()
                    ),

                density_orth=
                    orthonormal_spin_density(
                        reconstruction.mf.mol,
                        reconstruction.mf.make_rdm1(),
                    ),

                mf=
                    reconstruction.mf,
            ),
            origin_guess=
                "branch_reconstruction",
        )
    )

    reconstruction_cmp = (
        compare_states(
            energy_a_hartree=
                raw_min.energy_hartree,

            energy_b_hartree=
                reconstructed.energy_hartree,

            s2_a=
                raw_min.s2,

            s2_b=
                reconstructed.s2,

            dm_orth_a=
                raw_min.density_orth,

            dm_orth_b=
                reconstructed.density_orth,

            spin_a=
                raw_min.spin,

            spin_b=
                reconstructed.spin,
        )
    )

    stability = (
        optimize_internal_stability(
            reconstruction.mf,
            settings,
            max_iterations=
                max_stability_iterations,
        )
    )

    if not stability.finally_stable:

        return CanonicalBranch(
            source_branch=
                branch,

            reconstruction_relation=
                reconstruction_cmp.relation,

            stability_result=
                stability,

            stability_relation=None,

            canonical_points=[],

            valid=False,

            status=
                "QC_FAIL_STABILITY",
        )

    stabilized = (
        scout_solution_from_mf(
            mf=
                stability.final_mf,

            template=
                reconstructed,

            origin_guess=
                "stability_canonical",
        )
    )

    stability_cmp = (
        compare_states(
            energy_a_hartree=
                reconstructed.energy_hartree,

            energy_b_hartree=
                stabilized.energy_hartree,

            s2_a=
                reconstructed.s2,

            s2_b=
                stabilized.s2,

            dm_orth_a=
                reconstructed.density_orth,

            dm_orth_b=
                stabilized.density_orth,

            spin_a=
                reconstructed.spin,

            spin_b=
                stabilized.spin,
        )
    )

    canonical_points = (
        follow_scout_solution(
            branch_id=(
                branch.branch_id
                + "_CANON"
            ),

            atom=atom,
            ligand=ligand,

            seed=
                stabilized,

            settings=
                settings,

            r_values=
                coarse_r_values,

            max_memory_mb=
                max_memory_mb,
        )
    )

    return CanonicalBranch(
        source_branch=
            branch,

        reconstruction_relation=
            reconstruction_cmp.relation,

        stability_result=
            stability,

        stability_relation=
            stability_cmp.relation,

        canonical_points=
            canonical_points,

        valid=True,

        status="PASS",
    )


def canonicalize_all_branches(
    *,
    branches: Sequence[DiscoveredBranch],
    atom: str,
    ligand: str,
    basis: str,
    settings: SCFSettings,
    coarse_r_values: Sequence[float],
    max_memory_mb: int = 2000,
    max_stability_iterations: int = 6,
) -> List[CanonicalBranch]:

    output = []

    for branch in branches:

        output.append(
            canonicalize_branch_at_minimum(
                branch=branch,
                atom=atom,
                ligand=ligand,
                basis=basis,
                settings=settings,
                coarse_r_values=
                    coarse_r_values,
                max_memory_mb=
                    max_memory_mb,
                max_stability_iterations=
                    max_stability_iterations,
            )
        )

    return output


def select_ground_state(
    canonical_branches:
        Sequence[CanonicalBranch],
) -> CanonicalBranch | None:

    valid = [
        branch
        for branch
        in canonical_branches
        if (
            branch.valid
            and branch.minimum
            is not None
        )
    ]

    if not valid:
        return None

    return min(
        valid,
        key=lambda branch:
            branch.minimum.energy_hartree,
    )


def run_charge_ground_state_discovery(
    *,
    atom: str,
    ligand: str,
    charge: int,
    seed_r_values: Sequence[float],
    coarse_r_values: Sequence[float],
    spin_max: int,
    basis: str,
    settings: SCFSettings,
    max_memory_mb: int = 2000,
    precomputed_groups_by_seed=None,
) -> ChargeGroundStateResult:

    branches, records = (
        discover_charge_branches(
            atom=atom,
            ligand=ligand,
            charge=charge,
            seed_r_values=
                seed_r_values,
            coarse_r_values=
                coarse_r_values,
            spin_max=
                spin_max,
            basis=basis,
            settings=settings,
            max_memory_mb=
                max_memory_mb,

            precomputed_groups_by_seed=
                precomputed_groups_by_seed,
        )
    )

    canonical = (
        canonicalize_all_branches(
            branches=branches,
            atom=atom,
            ligand=ligand,
            basis=basis,
            settings=settings,
            coarse_r_values=
                coarse_r_values,
            max_memory_mb=
                max_memory_mb,
        )
    )

    ground = select_ground_state(
        canonical
    )

    return ChargeGroundStateResult(
        molecule=(
            atom + ligand
        ).upper(),

        charge=
            charge,

        discovered_branches=
            branches,

        canonical_branches=
            canonical,

        discovery_records=
            records,

        ground_state_branch=
            ground,
    )
