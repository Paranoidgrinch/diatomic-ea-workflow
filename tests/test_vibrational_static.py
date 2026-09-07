import unittest

import numpy as np

from diatomic_ea_v09.vibrational import (
    Isotopologue,
    harmonic_quadratic_fit,
    reduced_mass_u,
)


class VibrationalStaticTests(
    unittest.TestCase
):

    def test_reduced_mass(self):

        mu = reduced_mass_u(
            1.0,
            1.0,
        )

        self.assertAlmostEqual(
            mu,
            0.5,
        )


    def test_exact_parabola(self):

        r = np.arange(
            1.70,
            1.81,
            0.01,
        )

        re = 1.755

        curvature = 0.25

        e = (
            -10.0
            + 0.5
            * curvature
            * (r - re)**2
        )

        result = harmonic_quadratic_fit(
            r,
            e,

            n_points=7,

            isotopologue=
                Isotopologue(
                    atom_mass_u=24.0,
                    ligand_mass_u=1.0,
                ),
        )

        self.assertAlmostEqual(
            result.fitted_re_A,
            re,
            places=10,
        )

        self.assertAlmostEqual(
            result.curvature_hartree_per_A2,
            curvature,
            places=10,
        )

        self.assertGreater(
            result.harmonic_wavenumber_cm1,
            0.0,
        )

        self.assertGreater(
            result.zpe_eV,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
