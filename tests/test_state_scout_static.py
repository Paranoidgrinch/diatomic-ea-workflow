import unittest

import numpy as np

from diatomic_ea_v09.state_scout import (
    ScoutSolution,
    deduplicate_same_spin,
)


def fake_solution(
    *,
    energy,
    spin,
    density_value,
    guess,
):
    d = np.zeros((2, 2, 2))
    d[0, 0, 0] = density_value

    return ScoutSolution(
        molecule="XX",
        charge=0,
        spin=spin,
        multiplicity=spin + 1,
        r_A=1.0,
        functional="PBE",
        basis="test",
        origin_guess=guess,
        equivalent_guesses=[guess],
        energy_hartree=energy,
        s2=0.0,
        observed_multiplicity=spin + 1,
        homo_eV=0.0,
        lumo_eV=0.0,
        gap_eV=0.0,
        scf_path="standard",
        density_ao=d.copy(),
        density_orth=d.copy(),
        mf=None,
    )


class StateScoutStaticTests(unittest.TestCase):

    def test_obvious_duplicate_is_merged(self):

        a = fake_solution(
            energy=-10.0,
            spin=1,
            density_value=1.000000,
            guess="minao",
        )

        b = fake_solution(
            energy=-10.000000001,
            spin=1,
            density_value=1.000001,
            guess="hcore",
        )

        result = deduplicate_same_spin(
            [a, b],
            energy_tol_meV=1.0,
            density_tol=0.01,
        )

        self.assertEqual(len(result), 1)

        self.assertEqual(
            set(result[0].equivalent_guesses),
            {"minao", "hcore"},
        )

    def test_close_energy_but_different_density_is_kept(self):

        a = fake_solution(
            energy=-10.0,
            spin=1,
            density_value=1.0,
            guess="minao",
        )

        b = fake_solution(
            energy=-10.0000001,
            spin=1,
            density_value=1.3,
            guess="atom",
        )

        result = deduplicate_same_spin(
            [a, b],
            energy_tol_meV=1.0,
            density_tol=0.01,
        )

        self.assertEqual(len(result), 2)

    def test_different_spins_are_never_deduplicated(self):

        a = fake_solution(
            energy=-10.0,
            spin=1,
            density_value=1.0,
            guess="minao",
        )

        b = fake_solution(
            energy=-10.0,
            spin=3,
            density_value=1.0,
            guess="minao",
        )

        result = deduplicate_same_spin(
            [a, b],
            energy_tol_meV=1.0,
            density_tol=0.01,
        )

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
