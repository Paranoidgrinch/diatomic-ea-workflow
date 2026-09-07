import unittest

from diatomic_ea_v09.molecule import allowed_spins
from diatomic_ea_v09.scf import functional_to_xc


class CoreStaticTests(unittest.TestCase):

    def test_functional_aliases(self):
        self.assertEqual(
            functional_to_xc("PBE"),
            "PBE",
        )

        self.assertEqual(
            functional_to_xc("TPSSh"),
            "TPSSh",
        )

    def test_even_electron_spin_parity(self):
        spins = allowed_spins(
            "Mg",
            "H",
            charge=-1,
            spin_max=6,
        )

        self.assertEqual(
            spins,
            [0, 2, 4, 6],
        )

    def test_odd_electron_spin_parity(self):
        spins = allowed_spins(
            "Mg",
            "H",
            charge=0,
            spin_max=5,
        )

        self.assertEqual(
            spins,
            [1, 3, 5],
        )


if __name__ == "__main__":
    unittest.main()
