import unittest

from diatomic_ea_v09.fine_pec import (
    make_fine_grid,
)


class FinePECStaticTests(
    unittest.TestCase
):

    def test_default_grid(self):

        grid = make_fine_grid(
            1.80
        )

        self.assertEqual(
            len(grid),
            41,
        )

        self.assertAlmostEqual(
            grid[0],
            1.60,
        )

        self.assertAlmostEqual(
            grid[-1],
            2.00,
        )

        self.assertAlmostEqual(
            grid[20],
            1.80,
        )


if __name__ == "__main__":
    unittest.main()
