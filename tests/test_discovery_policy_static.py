import unittest

from diatomic_ea_v09.discovery_policy import (
    DiscoveryPolicy,
    assess_spin_frontier,
    make_scout_seed_grid,
)


class DiscoveryPolicyStaticTests(
    unittest.TestCase
):

    def test_seed_grid(self):

        coarse = [
            round(
                1.0 + 0.1 * i,
                10,
            )
            for i in range(21)
        ]

        seeds = make_scout_seed_grid(
            coarse
        )

        self.assertEqual(
            seeds,
            [
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                2.2,
                2.4,
                2.6,
                2.8,
            ],
        )


    def test_rising_spin_tail_closes(self):

        result = assess_spin_frontier(
            scanned_spins=[
                1,
                3,
                5,
                7,
                9,
            ],

            all_allowed_spins=[
                1,
                3,
                5,
                7,
                9,
                11,
            ],

            minima_by_seed={
                1.4: {
                    5: -10.0,
                    7: -9.0,
                    9: -8.0,
                },

                1.8: {
                    5: -10.5,
                    7: -9.2,
                    9: -7.8,
                },
            },

            tail_length=3,
        )

        self.assertTrue(
            result.closed
        )


    def test_nonmonotonic_tail_stays_open(self):

        result = assess_spin_frontier(
            scanned_spins=[
                1,
                3,
                5,
                7,
                9,
            ],

            all_allowed_spins=[
                1,
                3,
                5,
                7,
                9,
                11,
            ],

            minima_by_seed={
                1.4: {
                    5: -10.0,
                    7: -9.0,
                    9: -9.5,
                },
            },

            tail_length=3,
        )

        self.assertFalse(
            result.closed
        )

        self.assertEqual(
            result.failing_seed_r_values,
            [1.4],
        )


    def test_physical_exhaustion_closes(self):

        result = assess_spin_frontier(
            scanned_spins=[
                0,
                2,
                4,
            ],

            all_allowed_spins=[
                0,
                2,
                4,
            ],

            minima_by_seed={},

            tail_length=3,
        )

        self.assertTrue(
            result.closed
        )

        self.assertTrue(
            result.physical_spin_space_exhausted
        )


if __name__ == "__main__":
    unittest.main()
