#!/usr/bin/env python3

import csv
from pathlib import Path

from diatomic_ea_v09.vibrational import (
    Isotopologue,
    harmonic_quadratic_fit,
    zpe_corrected_ea_eV,
)
from pyscf.data import nist


BASEDIR = Path(
    "pilot_runs/fine_pec_mgh"
)


# Explicit isotopologue:
# NIST:
#   24Mg = 23.985041697 u
#   1H   =  1.00782503223 u
MG24H1 = Isotopologue(
    atom_mass_u=
        23.985041697,

    ligand_mass_u=
        1.00782503223,

    label="24Mg1H",
)


def load_pec(
    filename,
):
    path = (
        BASEDIR
        / filename
    )

    r = []
    e = []

    with path.open() as fh:

        reader = csv.DictReader(
            fh
        )

        for row in reader:

            r.append(
                float(
                    row["r_A"]
                )
            )

            e.append(
                float(
                    row[
                        "energy_hartree"
                    ]
                )
            )

    return r, e


neutral_r, neutral_e = (
    load_pec(
        "MGH_NEUTRAL_GS.csv"
    )
)

anion_r, anion_e = (
    load_pec(
        "MGH_ANION_GS.csv"
    )
)


print("=" * 128)
print("MgH HARMONIC FIT / ZPE SENSITIVITY")
print("=" * 128)

print(
    "Isotopologue:",
    MG24H1.label,
)

print(
    "m(24Mg) [u]:",
    MG24H1.atom_mass_u,
)

print(
    "m(1H) [u]:",
    MG24H1.ligand_mass_u,
)

print()


FIT_WIDTHS = [
    5,
    7,
    9,
    11,
]


results = {}


for n_points in FIT_WIDTHS:

    neutral = (
        harmonic_quadratic_fit(
            neutral_r,
            neutral_e,

            n_points=
                n_points,

            isotopologue=
                MG24H1,
        )
    )

    anion = (
        harmonic_quadratic_fit(
            anion_r,
            anion_e,

            n_points=
                n_points,

            isotopologue=
                MG24H1,
        )
    )

    ea_el = (
        neutral.fitted_energy_hartree
        - anion.fitted_energy_hartree
    ) * nist.HARTREE2EV

    delta_zpe = (
        neutral.zpe_eV
        - anion.zpe_eV
    )

    ea0 = (
        zpe_corrected_ea_eV(
            neutral_energy_hartree=
                neutral.fitted_energy_hartree,

            anion_energy_hartree=
                anion.fitted_energy_hartree,

            neutral_zpe_hartree=
                neutral.zpe_hartree,

            anion_zpe_hartree=
                anion.zpe_hartree,
        )
    )

    results[
        n_points
    ] = (
        neutral,
        anion,
        ea_el,
        delta_zpe,
        ea0,
    )


    print("#" * 128)
    print(
        f"{n_points}-POINT QUADRATIC FIT"
    )
    print("#" * 128)

    print(
        "Neutral:"
    )

    print(
        "  interval [A]: "
        "{:.3f} - {:.3f}".format(
            neutral.r_low_A,
            neutral.r_high_A,
        )
    )

    print(
        "  Re [A]: "
        "{:.8f}".format(
            neutral.fitted_re_A
        )
    )

    print(
        "  curvature [Eh/A^2]: "
        "{:.9f}".format(
            neutral.curvature_hartree_per_A2
        )
    )

    print(
        "  omega_e [cm^-1]: "
        "{:.3f}".format(
            neutral.harmonic_wavenumber_cm1
        )
    )

    print(
        "  ZPE [eV]: "
        "{:.8f}".format(
            neutral.zpe_eV
        )
    )

    print(
        "  fit RMS [microEh]: "
        "{:.6f}".format(
            neutral.fit_rms_microhartree
        )
    )

    print()


    print(
        "Anion:"
    )

    print(
        "  interval [A]: "
        "{:.3f} - {:.3f}".format(
            anion.r_low_A,
            anion.r_high_A,
        )
    )

    print(
        "  Re [A]: "
        "{:.8f}".format(
            anion.fitted_re_A
        )
    )

    print(
        "  curvature [Eh/A^2]: "
        "{:.9f}".format(
            anion.curvature_hartree_per_A2
        )
    )

    print(
        "  omega_e [cm^-1]: "
        "{:.3f}".format(
            anion.harmonic_wavenumber_cm1
        )
    )

    print(
        "  ZPE [eV]: "
        "{:.8f}".format(
            anion.zpe_eV
        )
    )

    print(
        "  fit RMS [microEh]: "
        "{:.6f}".format(
            anion.fit_rms_microhartree
        )
    )

    print()

    print(
        "EA_el [eV]: "
        "{:.8f}".format(
            ea_el
        )
    )

    print(
        "ZPE neutral - anion [eV]: "
        "{:+.8f}".format(
            delta_zpe
        )
    )

    print(
        "EA_0 [eV]: "
        "{:.8f}".format(
            ea0
        )
    )

    print()


print("=" * 128)
print("FIT-WIDTH COMPARISON")
print("=" * 128)

print(
    " n   Re_N[A]    Re_A[A]    "
    "wn_N[cm-1]  wn_A[cm-1]  "
    "EA_el[eV]   dZPE[eV]   EA0[eV]"
)

for n_points in FIT_WIDTHS:

    (
        neutral,
        anion,
        ea_el,
        delta_zpe,
        ea0,
    ) = results[n_points]

    print(
        "{:2d}  "
        "{:10.7f}  "
        "{:10.7f}  "
        "{:10.2f}  "
        "{:10.2f}  "
        "{:10.7f}  "
        "{:+10.7f}  "
        "{:10.7f}".format(
            n_points,

            neutral.fitted_re_A,
            anion.fitted_re_A,

            neutral.harmonic_wavenumber_cm1,
            anion.harmonic_wavenumber_cm1,

            ea_el,
            delta_zpe,
            ea0,
        )
    )


print()
print("=" * 128)
print("HARMONIC MgH ANALYSIS COMPLETE")
print("=" * 128)
