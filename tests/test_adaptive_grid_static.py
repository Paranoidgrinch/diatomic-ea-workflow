import unittest

from diatomic_ea_v09.adaptive_grid import (
    minimum_guard_status,
)
from diatomic_ea_v09.coarse_branch import (
    BranchPoint,
)


def point(
    r,
    energy,
):
    return BranchPoint(
        branch_id="TEST",
        r_A=r,
        parent_r_A=None,
        spin=1,
        multiplicity=2,
        converged=True,
        energy_hartree=energy,
        s2=0.75,
        observed_multiplicity=2.0,
        homo_eV=-1.0,
        lumo_eV=1.0,
        gap_eV=2.0,
        scf_path="standard",
        density_ao=None,
        density_orth=None,
        point_source="test",
    )


class AdaptiveGridTests(
    unittest.TestCase
):

    def test_interior_minimum_with_guard(self):

        pts = [
            point(1.0, -9.0),
            point(1.1, -9.5),
            point(1.2, -10.0),
            point(1.3, -9.5),
            point(1.4, -9.0),
        ]

        status = (
            minimum_guard_status(
                pts,
                guard_points=2,
            )
        )

        self.assertFalse(
            status["need_lower"]
        )

        self.assertFalse(
            status["need_upper"]
        )

    def test_upper_boundary_requests_expansion(self):

        pts = [
            point(1.0, -9.0),
            point(1.1, -9.5),
            point(1.2, -10.0),
        ]

        status = (
            minimum_guard_status(
                pts,
                guard_points=2,
            )
        )

        self.assertTrue(
            status["need_upper"]
        )


if __name__ == "__main__":
    unittest.main()
