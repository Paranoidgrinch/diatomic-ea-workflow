import unittest

from diatomic_ea_v09.state_identity import (
    StateRelation,
    classify_metrics,
)


class StateIdentityTests(unittest.TestCase):

    def test_bn_like_equivalent_states(self):

        relation = classify_metrics(
            same_spin=True,
            delta_energy_meV=0.000996,
            delta_s2=5.0e-7,
            total_spectrum_max=3.2e-7,
            spin_spectrum_max=7.0e-6,
        )

        self.assertEqual(
            relation,
            StateRelation.SAME_STATE,
        )

    def test_feh_atom_huckel_like_same_state(self):

        relation = classify_metrics(
            same_spin=True,
            delta_energy_meV=0.043796,
            delta_s2=3.1e-5,
            total_spectrum_max=1.7e-5,
            spin_spectrum_max=9.0e-5,
        )

        self.assertEqual(
            relation,
            StateRelation.SAME_STATE,
        )

    def test_feh_minao_other_like_distinct_state(self):

        relation = classify_metrics(
            same_spin=True,
            delta_energy_meV=23.76,
            delta_s2=0.0146,
            total_spectrum_max=0.0077,
            spin_spectrum_max=0.037,
        )

        self.assertEqual(
            relation,
            StateRelation.DISTINCT_STATE,
        )

    def test_borderline_case_is_ambiguous(self):

        relation = classify_metrics(
            same_spin=True,
            delta_energy_meV=2.0,
            delta_s2=0.003,
            total_spectrum_max=3.0e-4,
            spin_spectrum_max=8.0e-4,
        )

        self.assertEqual(
            relation,
            StateRelation.AMBIGUOUS,
        )

    def test_different_spin_is_distinct(self):

        relation = classify_metrics(
            same_spin=False,
            delta_energy_meV=0.0,
            delta_s2=0.0,
            total_spectrum_max=0.0,
            spin_spectrum_max=0.0,
        )

        self.assertEqual(
            relation,
            StateRelation.DISTINCT_STATE,
        )


if __name__ == "__main__":
    unittest.main()
