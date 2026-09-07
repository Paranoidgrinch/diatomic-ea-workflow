#!/usr/bin/env python3

import json
import pickle
from pathlib import Path

from pyscf.data import nist

from diatomic_ea_v09.fine_candidates import (
    evaluate_local_well_qc,
    fine_candidate_effective_status,
)
from diatomic_ea_v09.vibrational import (
    Isotopologue,
    harmonic_quadratic_fit,
    zpe_corrected_ea_eV,
)


NEUTRAL_PATH = Path(
    "pilot_runs/feh_neutral_checkpoint/stage_d_fine.pkl"
)

ANION_PATH = Path(
    "pilot_runs/feh_anion_checkpoint/stage_d_fine.pkl"
)

OUTPUT_PATH = Path(
    "pilot_runs/feh_harmonic_zpe_ea0.json"
)

NEUTRAL_ID = "SEED_BRANCH_016"
ANION_ID = "SEED_BRANCH_043"

FIT_WIDTHS = (
    5,
    7,
    9,
    11,
)


FE56_H1 = Isotopologue(
    atom_mass_u=55.93493633,
    ligand_mass_u=1.00782503223,
    label="56Fe1H",
)


def load_outcome(
    path,
    candidate_id,
):
    with path.open("rb") as fh:
        payload = pickle.load(fh)

    return payload[
        "outcomes"
    ][candidate_id]


def validate_outcome(
    outcome,
    label,
):
    effective = (
        fine_candidate_effective_status(
            outcome
        )
    )

    if effective not in (
        "PASS",
        "PASS_LOCAL_WELL_FULL_BRANCH_REVIEW",
    ):
        raise RuntimeError(
            "{} Fine candidate is not valid: {}".format(
                label,
                effective,
            )
        )

    if (
        outcome.fine is None
        or outcome.qc is None
    ):
        raise RuntimeError(
            label
            + " Fine/QC data missing."
        )

    local = evaluate_local_well_qc(
        fine=outcome.fine,
        qc=outcome.qc,
    )

    if local.status != "PASS":
        raise RuntimeError(
            "{} local-well QC failed: {}".format(
                label,
                local.status,
            )
        )

    points = sorted(
        [
            point
            for point in outcome.fine.points
            if (
                point.converged
                and point.energy_hartree
                is not None
            )
        ],
        key=lambda point:
            point.r_A,
    )

    return (
        effective,
        local,
        points,
    )


def assert_fit_points_pass_qc(
    *,
    outcome,
    fit,
    label,
):
    """
    A sensitivity fit is accepted only if every Fine point actually
    used by that n-point quadratic fit has strict pointwise QC=PASS.
    """

    selected = [
        point
        for point in outcome.qc.points
        if (
            fit.r_low_A - 1.0e-10
            <= float(point.r_A)
            <= fit.r_high_A + 1.0e-10
        )
    ]

    if len(selected) != fit.n_points:
        raise RuntimeError(
            "{} {}-point fit: expected {} QC records in "
            "{:.3f}--{:.3f} A, found {}".format(
                label,
                fit.n_points,
                fit.n_points,
                fit.r_low_A,
                fit.r_high_A,
                len(selected),
            )
        )

    failed = [
        (
            float(point.r_A),
            point.status,
        )
        for point in selected
        if point.status != "PASS"
    ]

    if failed:
        raise RuntimeError(
            "{} {}-point fit contains non-PASS QC points: {}".format(
                label,
                fit.n_points,
                failed,
            )
        )

    return len(selected)


neutral = load_outcome(
    NEUTRAL_PATH,
    NEUTRAL_ID,
)

anion = load_outcome(
    ANION_PATH,
    ANION_ID,
)


(
    neutral_status,
    neutral_local,
    neutral_points,
) = validate_outcome(
    neutral,
    "Neutral",
)

(
    anion_status,
    anion_local,
    anion_points,
) = validate_outcome(
    anion,
    "Anion",
)


neutral_r = [
    point.r_A
    for point in neutral_points
]

neutral_e = [
    point.energy_hartree
    for point in neutral_points
]

anion_r = [
    point.r_A
    for point in anion_points
]

anion_e = [
    point.energy_hartree
    for point in anion_points
]


print("=" * 132)
print("FeH HARMONIC FIT / ZPE / EA0 SENSITIVITY")
print("=" * 132)

print(
    "Isotopologue:",
    FE56_H1.label,
)

print(
    "m(56Fe) [u]:",
    FE56_H1.atom_mass_u,
)

print(
    "m(1H) [u]:",
    FE56_H1.ligand_mass_u,
)

print()

print(
    "Neutral candidate:",
    NEUTRAL_ID,
)

print(
    "Neutral effective status:",
    neutral_status,
)

print(
    "Neutral baseline local-well QC:",
    neutral_local.status,
    "{}--{} A".format(
        neutral_local.r_low_A,
        neutral_local.r_high_A,
    ),
)

print()

print(
    "Anion candidate:",
    ANION_ID,
)

print(
    "Anion effective status:",
    anion_status,
)

print(
    "Anion baseline local-well QC:",
    anion_local.status,
    "{}--{} A".format(
        anion_local.r_low_A,
        anion_local.r_high_A,
    ),
)


records = []


for n_points in FIT_WIDTHS:

    neutral_fit = (
        harmonic_quadratic_fit(
            neutral_r,
            neutral_e,
            n_points=n_points,
            isotopologue=FE56_H1,
        )
    )

    anion_fit = (
        harmonic_quadratic_fit(
            anion_r,
            anion_e,
            n_points=n_points,
            isotopologue=FE56_H1,
        )
    )


    neutral_qc_points = (
        assert_fit_points_pass_qc(
            outcome=neutral,
            fit=neutral_fit,
            label="Neutral",
        )
    )

    anion_qc_points = (
        assert_fit_points_pass_qc(
            outcome=anion,
            fit=anion_fit,
            label="Anion",
        )
    )


    ea_el_eV = (
        (
            neutral_fit
            .fitted_energy_hartree

            - anion_fit
            .fitted_energy_hartree
        )
        * nist.HARTREE2EV
    )


    delta_zpe_eV = (
        neutral_fit.zpe_eV
        - anion_fit.zpe_eV
    )


    ea0_eV = (
        zpe_corrected_ea_eV(
            neutral_energy_hartree=
                neutral_fit
                .fitted_energy_hartree,

            anion_energy_hartree=
                anion_fit
                .fitted_energy_hartree,

            neutral_zpe_hartree=
                neutral_fit
                .zpe_hartree,

            anion_zpe_hartree=
                anion_fit
                .zpe_hartree,
        )
    )


    record = {
        "n_points":
            n_points,

        "neutral": {
            "r_low_A":
                neutral_fit.r_low_A,

            "r_high_A":
                neutral_fit.r_high_A,

            "re_A":
                neutral_fit.fitted_re_A,

            "energy_hartree":
                neutral_fit
                .fitted_energy_hartree,

            "curvature_hartree_per_A2":
                neutral_fit
                .curvature_hartree_per_A2,

            "wavenumber_cm1":
                neutral_fit
                .harmonic_wavenumber_cm1,

            "zpe_eV":
                neutral_fit.zpe_eV,

            "fit_rms_microhartree":
                neutral_fit
                .fit_rms_microhartree,

            "fit_max_abs_microhartree":
                neutral_fit
                .fit_max_abs_microhartree,

            "qc_points_passed":
                neutral_qc_points,
        },

        "anion": {
            "r_low_A":
                anion_fit.r_low_A,

            "r_high_A":
                anion_fit.r_high_A,

            "re_A":
                anion_fit.fitted_re_A,

            "energy_hartree":
                anion_fit
                .fitted_energy_hartree,

            "curvature_hartree_per_A2":
                anion_fit
                .curvature_hartree_per_A2,

            "wavenumber_cm1":
                anion_fit
                .harmonic_wavenumber_cm1,

            "zpe_eV":
                anion_fit.zpe_eV,

            "fit_rms_microhartree":
                anion_fit
                .fit_rms_microhartree,

            "fit_max_abs_microhartree":
                anion_fit
                .fit_max_abs_microhartree,

            "qc_points_passed":
                anion_qc_points,
        },

        "ea_el_eV":
            ea_el_eV,

        "delta_zpe_eV":
            delta_zpe_eV,

        "ea0_eV":
            ea0_eV,
    }


    records.append(
        record
    )


    print()
    print("#" * 132)

    print(
        "{}-POINT QUADRATIC FIT".format(
            n_points
        )
    )

    print("#" * 132)

    print(
        "Neutral: "
        "R={:.3f}--{:.3f} A  "
        "Re={:.8f} A  "
        "omega_e={:.3f} cm^-1  "
        "ZPE={:.8f} eV  "
        "RMS={:.6f} microEh  "
        "QC={}/{}".format(
            neutral_fit.r_low_A,
            neutral_fit.r_high_A,
            neutral_fit.fitted_re_A,
            neutral_fit
            .harmonic_wavenumber_cm1,
            neutral_fit.zpe_eV,
            neutral_fit
            .fit_rms_microhartree,
            neutral_qc_points,
            n_points,
        )
    )

    print(
        "Anion:   "
        "R={:.3f}--{:.3f} A  "
        "Re={:.8f} A  "
        "omega_e={:.3f} cm^-1  "
        "ZPE={:.8f} eV  "
        "RMS={:.6f} microEh  "
        "QC={}/{}".format(
            anion_fit.r_low_A,
            anion_fit.r_high_A,
            anion_fit.fitted_re_A,
            anion_fit
            .harmonic_wavenumber_cm1,
            anion_fit.zpe_eV,
            anion_fit
            .fit_rms_microhartree,
            anion_qc_points,
            n_points,
        )
    )

    print(
        "EA_el = {:.8f} eV".format(
            ea_el_eV
        )
    )

    print(
        "dZPE  = {:+.8f} eV".format(
            delta_zpe_eV
        )
    )

    print(
        "EA_0  = {:.8f} eV".format(
            ea0_eV
        )
    )


baseline = next(
    record
    for record in records
    if record["n_points"] == 7
)


ea0_values = [
    record["ea0_eV"]
    for record in records
]


summary = {
    "isotopologue": {
        "label":
            FE56_H1.label,

        "fe56_mass_u":
            FE56_H1.atom_mass_u,

        "h1_mass_u":
            FE56_H1.ligand_mass_u,
    },

    "neutral_candidate":
        NEUTRAL_ID,

    "anion_candidate":
        ANION_ID,

    "neutral_effective_status":
        neutral_status,

    "anion_effective_status":
        anion_status,

    "fits":
        records,

    "working_baseline":
        baseline,

    "ea0_fit_width_span_eV":
        max(ea0_values)
        - min(ea0_values),
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w"
) as fh:

    json.dump(
        summary,
        fh,
        indent=2,
    )


print()
print("=" * 132)
print("7-POINT WORKING BASELINE")
print("=" * 132)

print(
    "EA_el [eV]:       {:.8f}".format(
        baseline["ea_el_eV"]
    )
)

print(
    "Neutral omega_e:  {:.3f} cm^-1".format(
        baseline[
            "neutral"
        ][
            "wavenumber_cm1"
        ]
    )
)

print(
    "Anion omega_e:    {:.3f} cm^-1".format(
        baseline[
            "anion"
        ][
            "wavenumber_cm1"
        ]
    )
)

print(
    "Neutral ZPE [eV]: {:.8f}".format(
        baseline[
            "neutral"
        ][
            "zpe_eV"
        ]
    )
)

print(
    "Anion ZPE [eV]:   {:.8f}".format(
        baseline[
            "anion"
        ][
            "zpe_eV"
        ]
    )
)

print(
    "Delta ZPE [eV]:   {:+.8f}".format(
        baseline[
            "delta_zpe_eV"
        ]
    )
)

print(
    "EA_0 [eV]:        {:.8f}".format(
        baseline[
            "ea0_eV"
        ]
    )
)

print(
    "5/7/9/11 EA0 span [eV]: {:.8f}".format(
        summary[
            "ea0_fit_width_span_eV"
        ]
    )
)

print()

print(
    "Output:",
    OUTPUT_PATH,
)

print("=" * 132)
