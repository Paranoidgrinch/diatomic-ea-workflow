from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .adaptive_grid import (
    ExpansionPolicy,
)
from .adaptive_scout import (
    AdaptiveScoutResult,
    run_adaptive_local_scout,
)
from .charge_pipeline import (
    ChargeGroundStatePipelineResult,
    run_charge_ground_state_pipeline,
)
from .discovery_policy import (
    DiscoveryPolicy,
)
from .fine_pec import (
    FineGridPolicy,
)
from .model import (
    SCFSettings,
)


@dataclass
class AdaptiveChargePipelineResult:
    status: str

    scout: AdaptiveScoutResult

    ground_state: ChargeGroundStatePipelineResult | None


def run_adaptive_charge_ground_state_pipeline(
    *,
    atom: str,
    ligand: str,

    charge: int,

    coarse_r_values:
        Sequence[float],

    basis: str,

    settings: SCFSettings,

    discovery_policy:
        DiscoveryPolicy =
        DiscoveryPolicy(),

    expansion_policy:
        ExpansionPolicy =
        ExpansionPolicy(),

    fine_policy:
        FineGridPolicy =
        FineGridPolicy(),

    perform_pointwise_qc: bool = True,

    max_memory_mb: int = 2000,
) -> AdaptiveChargePipelineResult:
    """
    Integrated adaptive ground-state workflow for one charge state.

    The expensive local scouts are calculated ONCE.

    Their CandidateGroups are passed directly into the normal
    branch-discovery pipeline.
    """

    scout = run_adaptive_local_scout(
        atom=atom,
        ligand=ligand,

        charge=charge,

        coarse_r_values=
            coarse_r_values,

        basis=basis,

        settings=settings,

        policy=
            discovery_policy,

        max_memory_mb=
            max_memory_mb,
    )


    if (
        scout.status
        != "PASS_SPIN_FRONTIER"
    ):

        return AdaptiveChargePipelineResult(
            status=
                "QC_FAIL_SPIN_FRONTIER",

            scout=
                scout,

            ground_state=None,
        )


    ground_state = (
        run_charge_ground_state_pipeline(
            atom=atom,
            ligand=ligand,

            charge=charge,

            seed_r_values=
                scout.seed_r_values,

            coarse_r_values=
                coarse_r_values,

            spin_max=
                scout.spin_max,

            basis=basis,

            settings=settings,

            expansion_policy=
                expansion_policy,

            fine_policy=
                fine_policy,

            perform_pointwise_qc=
                perform_pointwise_qc,

            max_memory_mb=
                max_memory_mb,

            precomputed_groups_by_seed=
                scout.groups_by_seed,
        )
    )


    return AdaptiveChargePipelineResult(
        status=
            ground_state.status,

        scout=
            scout,

        ground_state=
            ground_state,
    )
