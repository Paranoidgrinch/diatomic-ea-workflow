import unittest

from diatomic_ea_v09.coarse_branch import (
    BranchPoint,
)
from diatomic_ea_v09.ground_state import (
    branch_minimum,
    point_at_r,
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


class GroundStateStaticTests(
    unittest.TestCase
):

    def test_branch_minimum(self):

        pts = [
            point(1.4, -10.0),
            point(1.5, -10.1),
            point(1.6, -10.05),
        ]

        result = branch_minimum(
            pts
        )

        self.assertEqual(
            result.r_A,
            1.5,
        )

    def test_point_lookup(self):

        pts = [
            point(1.4, -10.0),
            point(1.5, -10.1),
        ]

        self.assertIsNotNone(
            point_at_r(
                pts,
                1.5,
            )
        )

        self.assertIsNone(
            point_at_r(
                pts,
                1.6,
            )
        )


if __name__ == "__main__":
    unittest.main()
