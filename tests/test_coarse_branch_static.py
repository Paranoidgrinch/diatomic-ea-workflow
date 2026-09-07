import unittest

from diatomic_ea_v09.coarse_branch import (
    split_grid_around_seed,
)


class CoarseBranchStaticTests(
    unittest.TestCase
):

    def test_bidirectional_order(self):

        lower, upper = (
            split_grid_around_seed(
                [
                    1.30,
                    1.40,
                    1.50,
                    1.54,
                    1.55,
                    1.60,
                    1.70,
                ],
                1.54,
            )
        )

        self.assertEqual(
            lower,
            [1.50, 1.40, 1.30],
        )

        self.assertEqual(
            upper,
            [1.55, 1.60, 1.70],
        )


if __name__ == "__main__":
    unittest.main()
