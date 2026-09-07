import unittest
from types import SimpleNamespace

from diatomic_ea_v09.charge_pipeline import (
    pipeline_status_from_fine,
)


def outcome(
    status,
):
    return SimpleNamespace(
        status=status
    )


class ChargePipelineStaticTests(
    unittest.TestCase
):

    def test_clean_resolved_pass(self):

        resolution = SimpleNamespace(
            status="FINE_GS_RESOLVED"
        )

        status = (
            pipeline_status_from_fine(
                outcomes=[
                    outcome("PASS")
                ],

                resolution=
                    resolution,
            )
        )

        self.assertEqual(
            status,
            "PASS",
        )


    def test_ambiguous_but_dominated_pass(self):

        resolution = SimpleNamespace(
            status=(
                "FINE_GS_RESOLVED_WITH_"
                "AMBIGUOUS_ALTERNATIVES"
            )
        )

        status = (
            pipeline_status_from_fine(
                outcomes=[
                    outcome("PASS"),
                    outcome("PASS"),
                ],

                resolution=
                    resolution,
            )
        )

        self.assertEqual(
            status,
            "PASS_AMBIGUOUS_"
            "ALTERNATIVE_DOMINATED",
        )


    def test_unresolved_ambiguity_is_not_pass(self):

        resolution = SimpleNamespace(
            status="FINE_GS_AMBIGUOUS"
        )

        status = (
            pipeline_status_from_fine(
                outcomes=[
                    outcome("PASS"),
                    outcome("PASS"),
                ],

                resolution=
                    resolution,
            )
        )

        self.assertEqual(
            status,
            "FINE_GS_AMBIGUOUS",
        )


    def test_failed_candidate_cannot_disappear(self):

        resolution = SimpleNamespace(
            status="FINE_GS_RESOLVED"
        )

        status = (
            pipeline_status_from_fine(
                outcomes=[
                    outcome("PASS"),
                    outcome(
                        "QC_FAIL_FINE_POINTWISE"
                    ),
                ],

                resolution=
                    resolution,
            )
        )

        self.assertEqual(
            status,
            "QC_FAIL_FINE_CANDIDATE",
        )


if __name__ == "__main__":
    unittest.main()
