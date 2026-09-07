import unittest

from diatomic_ea_v09.fine_qc import (
    FinePointQC,
)


class FineQCStaticTests(
    unittest.TestCase
):

    def test_record(self):

        record = FinePointQC(
            r_A=1.75,
            status="PASS",

            stored_energy_hartree=-10.0,
            reconstructed_energy_hartree=-10.0,
            stabilized_energy_hartree=-10.0,

            reconstruction_delta_meV=0.0,
            reconstruction_relation=None,

            initially_stable=True,
            finally_stable=True,

            stability_reoptimizations=0,
            stability_delta_energy_meV=0.0,
            stability_relation=None,
        )

        self.assertEqual(
            record.status,
            "PASS",
        )


if __name__ == "__main__":
    unittest.main()
