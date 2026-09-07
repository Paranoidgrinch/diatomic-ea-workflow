import unittest
from types import SimpleNamespace

from diatomic_ea_v09.fine_candidates import (
    evaluate_local_well_qc,
    fine_candidate_effective_status,
    fine_candidate_is_resolvable,
)
from diatomic_ea_v09.charge_pipeline import (
    pipeline_status_from_fine,
)


def fine_summary():
    rs = [
        round(
            1.50 + 0.01 * i,
            10,
        )
        for i in range(15)
    ]

    points = [
        SimpleNamespace(
            r_A=r,
            converged=True,
            energy_hartree=(
                -10.0
                + (r - 1.57)**2
            ),
        )
        for r in rs
    ]

    return SimpleNamespace(
        points=points,

        discrete_min_r_A=1.57,
        discrete_min_energy_hartree=-10.0,

        fitted_min_r_A=1.57,
        fitted_min_energy_hartree=-10.0,

        curvature_hartree_per_A2=1.0,

        quadratic_points_used=7,

        quadratic_r_low_A=1.54,
        quadratic_r_high_A=1.60,

        minimum_at_boundary=False,
    )


def qc_result(
    *,
    bad_r=None,
):
    rs = [
        round(
            1.50 + 0.01 * i,
            10,
        )
        for i in range(15)
    ]

    points = []

    for r in rs:

        status = (
            "PASS"
            if (
                bad_r is None
                or abs(
                    r - bad_r
                ) > 1.0e-10
            )
            else "STABILITY_ROOT_CHANGE"
        )

        points.append(
            SimpleNamespace(
                r_A=r,
                status=status,
            )
        )

    return SimpleNamespace(
        points=points,
        status="QC_REVIEW_REQUIRED",
    )


class LocalWellQCStaticTests(
    unittest.TestCase
):

    def test_outer_problem_does_not_fail_local_well(
        self,
    ):
        result = evaluate_local_well_qc(
            fine=fine_summary(),
            qc=qc_result(
                bad_r=1.50
            ),
        )

        self.assertEqual(
            result.status,
            "PASS",
        )

        self.assertEqual(
            result.n_points,
            7,
        )

        self.assertEqual(
            result.n_pass,
            7,
        )

        self.assertAlmostEqual(
            result.r_low_A,
            1.54,
        )

        self.assertAlmostEqual(
            result.r_high_A,
            1.60,
        )


    def test_problem_inside_fit_fails_local_well(
        self,
    ):
        result = evaluate_local_well_qc(
            fine=fine_summary(),
            qc=qc_result(
                bad_r=1.56
            ),
        )

        self.assertEqual(
            result.status,
            "QC_FAIL_LOCAL_WELL",
        )

        self.assertEqual(
            result.n_pass,
            6,
        )


    def test_legacy_checkpoint_is_reclassified(
        self,
    ):
        outcome = SimpleNamespace(
            status=
                "QC_FAIL_FINE_POINTWISE",

            fine=
                fine_summary(),

            qc=
                qc_result(
                    bad_r=1.50
                ),
        )

        self.assertEqual(
            fine_candidate_effective_status(
                outcome
            ),
            (
                "PASS_LOCAL_WELL_"
                "FULL_BRANCH_REVIEW"
            ),
        )

        self.assertTrue(
            fine_candidate_is_resolvable(
                outcome
            )
        )


    def test_legacy_local_failure_remains_failure(
        self,
    ):
        outcome = SimpleNamespace(
            status=
                "QC_FAIL_FINE_POINTWISE",

            fine=
                fine_summary(),

            qc=
                qc_result(
                    bad_r=1.56
                ),
        )

        self.assertEqual(
            fine_candidate_effective_status(
                outcome
            ),
            "QC_FAIL_FINE_POINTWISE",
        )

        self.assertFalse(
            fine_candidate_is_resolvable(
                outcome
            )
        )


class ChargeStatusStaticTests(
    unittest.TestCase
):

    def test_resolved_with_full_branch_review(
        self,
    ):
        outcome = SimpleNamespace(
            status=(
                "PASS_LOCAL_WELL_"
                "FULL_BRANCH_REVIEW"
            ),

            fine=None,
            qc=None,
        )

        resolution = SimpleNamespace(
            status=
                "FINE_GS_RESOLVED"
        )

        status = pipeline_status_from_fine(
            outcomes=[
                outcome
            ],
            resolution=
                resolution,
        )

        self.assertEqual(
            status,
            "PASS_FULL_BRANCH_REVIEW",
        )


    def test_dominated_ambiguity_keeps_review_flag(
        self,
    ):
        outcome = SimpleNamespace(
            status=(
                "PASS_LOCAL_WELL_"
                "FULL_BRANCH_REVIEW"
            ),

            fine=None,
            qc=None,
        )

        resolution = SimpleNamespace(
            status=(
                "FINE_GS_RESOLVED_WITH_"
                "AMBIGUOUS_ALTERNATIVES"
            )
        )

        status = pipeline_status_from_fine(
            outcomes=[
                outcome
            ],
            resolution=
                resolution,
        )

        self.assertEqual(
            status,
            (
                "PASS_AMBIGUOUS_"
                "ALTERNATIVE_DOMINATED_"
                "FULL_BRANCH_REVIEW"
            ),
        )


if __name__ == "__main__":
    unittest.main()
