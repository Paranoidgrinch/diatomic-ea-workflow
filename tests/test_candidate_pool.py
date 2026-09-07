import unittest

import numpy as np

from diatomic_ea_v09.candidate_pool import (
    build_candidate_groups,
)
from diatomic_ea_v09.state_scout import (
    ScoutSolution,
)


def make_solution(
    *,
    energy,
    spin,
    total_shift,
    spin_shift,
    guess,
):

    # Construct simple spin-resolved orthonormal
    # densities with controllable total/spin spectra.
    da = np.array([
        [1.0 + total_shift + spin_shift, 0.0],
        [0.0, 0.0],
    ])

    db = np.array([
        [0.5 + total_shift - spin_shift, 0.0],
        [0.0, 0.0],
    ])

    dm = np.stack(
        [da, db]
    )

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
        s2=2.0,
        observed_multiplicity=
            spin + 1,

        homo_eV=-1.0,
        lumo_eV=1.0,
        gap_eV=2.0,

        scf_path="standard",

        density_ao=dm.copy(),
        density_orth=dm.copy(),

        mf=None,
    )


class CandidatePoolTests(unittest.TestCase):

    def test_same_states_form_one_group(self):

        a = make_solution(
            energy=-10.0,
            spin=2,
            total_shift=0.0,
            spin_shift=0.0,
            guess="minao",
        )

        b = make_solution(
            energy=-10.000000001,
            spin=2,
            total_shift=1.0e-6,
            spin_shift=1.0e-6,
            guess="hcore",
        )

        groups = build_candidate_groups(
            [a, b]
        )

        self.assertEqual(
            len(groups),
            1,
        )

        self.assertEqual(
            set(
                groups[0].equivalent_guesses
            ),
            {"minao", "hcore"},
        )

    def test_distinct_state_stays_separate(self):

        a = make_solution(
            energy=-10.0,
            spin=2,
            total_shift=0.0,
            spin_shift=0.0,
            guess="minao",
        )

        b = make_solution(
            energy=-9.99,
            spin=2,
            total_shift=0.02,
            spin_shift=0.02,
            guess="atom",
        )

        groups = build_candidate_groups(
            [a, b]
        )

        self.assertEqual(
            len(groups),
            2,
        )

    def test_different_spins_stay_separate(self):

        a = make_solution(
            energy=-10.0,
            spin=0,
            total_shift=0.0,
            spin_shift=0.0,
            guess="minao",
        )

        b = make_solution(
            energy=-10.0,
            spin=2,
            total_shift=0.0,
            spin_shift=0.0,
            guess="minao",
        )

        groups = build_candidate_groups(
            [a, b]
        )

        self.assertEqual(
            len(groups),
            2,
        )


if __name__ == "__main__":
    unittest.main()
