import unittest

from diatomic_ea_v09.discovery_policy import (
    DiscoveryPolicy,
)
from diatomic_ea_v09.adaptive_scout import (
    development_spin_values,
)


class AdaptiveScoutStaticTests(
    unittest.TestCase
):

    def test_development_spin_list_is_capped(self):

        policy = DiscoveryPolicy(
            max_spin_sectors=3
        )

        physical, development = (
            development_spin_values(
                atom="Fe",
                ligand="H",
                charge=0,
                policy=policy,
            )
        )

        self.assertLessEqual(
            len(development),
            3,
        )

        self.assertEqual(
            development,
            physical[:3],
        )


if __name__ == "__main__":
    unittest.main()


class AdaptiveScoutGuessFailureTests(
    unittest.TestCase
):

    def test_one_runtimeerror_does_not_abort_sector(self):

        from unittest.mock import patch

        from diatomic_ea_v09.adaptive_scout import (
            scout_spin_with_failures,
        )

        fake_spec = object()
        fake_settings = object()
        fake_solution = object()


        with patch(
            "diatomic_ea_v09.adaptive_scout.run_guess",
            side_effect=[
                RuntimeError(
                    "Huckel guess unavailable"
                ),
                fake_solution,
            ],
        ):

            solutions, failures = (
                scout_spin_with_failures(
                    spec=fake_spec,
                    settings=fake_settings,
                    guesses=[
                        "huckel",
                        "hcore",
                    ],
                )
            )


        self.assertEqual(
            solutions,
            [
                fake_solution
            ],
        )

        self.assertEqual(
            len(failures),
            1,
        )

        self.assertIn(
            "huckel",
            failures[0],
        )

        self.assertIn(
            "RuntimeError",
            failures[0],
        )


    def test_non_runtimeerror_remains_fatal(self):

        from unittest.mock import patch

        from diatomic_ea_v09.adaptive_scout import (
            scout_spin_with_failures,
        )


        with patch(
            "diatomic_ea_v09.adaptive_scout.run_guess",
            side_effect=TypeError(
                "programming error"
            ),
        ):

            with self.assertRaises(
                TypeError
            ):

                scout_spin_with_failures(
                    spec=object(),
                    settings=object(),
                    guesses=[
                        "minao"
                    ],
                )
