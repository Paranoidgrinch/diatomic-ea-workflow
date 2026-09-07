import unittest

from diatomic_ea_v09.branch_finalize import (
    FinalizationCycle,
)


class BranchFinalizeStaticTests(
    unittest.TestCase
):

    def test_cycle_record(self):

        record = FinalizationCycle(
            cycle=1,
            expanded=True,
            expansion_status="PASS",
            expansion_rounds=2,

            pre_stability_min_r_A=3.7,
            pre_stability_min_energy_hartree=-10.0,

            initially_stable=True,
            finally_stable=True,
            stability_reoptimizations=0,
            stability_delta_energy_meV=0.0,

            post_stability_min_r_A=3.7,
            post_stability_min_energy_hartree=-10.0,

            minimum_shift_A=0.0,
        )

        self.assertEqual(
            record.expansion_status,
            "PASS",
        )

        self.assertEqual(
            record.minimum_shift_A,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
