import unittest
from unittest.mock import patch

from diatomic_ea_v09.ground_state import (
    discover_charge_branches,
)
from diatomic_ea_v09.model import (
    SCFSettings,
)


class PrecomputedScoutPathTests(
    unittest.TestCase
):

    def test_precomputed_empty_groups_bypass_local_scout(self):

        settings = SCFSettings()

        with patch(
            "diatomic_ea_v09.ground_state.local_scout_groups"
        ) as local_scout:

            branches, records = (
                discover_charge_branches(
                    atom="Mg",
                    ligand="H",

                    charge=0,

                    seed_r_values=[
                        1.2,
                        1.4,
                    ],

                    coarse_r_values=[
                        1.0,
                        1.1,
                        1.2,
                        1.3,
                        1.4,
                        1.5,
                    ],

                    spin_max=3,

                    basis="def2-qzvpd",

                    settings=settings,

                    precomputed_groups_by_seed={
                        1.2: [],
                        1.4: [],
                    },
                )
            )

            local_scout.assert_not_called()

        self.assertEqual(
            branches,
            [],
        )

        self.assertEqual(
            records,
            [],
        )


    def test_missing_precomputed_seed_fails_loudly(self):

        settings = SCFSettings()

        with self.assertRaises(
            ValueError
        ):

            discover_charge_branches(
                atom="Mg",
                ligand="H",

                charge=0,

                seed_r_values=[
                    1.2,
                    1.4,
                ],

                coarse_r_values=[
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                ],

                spin_max=3,

                basis="def2-qzvpd",

                settings=settings,

                precomputed_groups_by_seed={
                    1.2: [],
                },
            )


if __name__ == "__main__":
    unittest.main()
