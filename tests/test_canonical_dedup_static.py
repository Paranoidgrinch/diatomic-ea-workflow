import unittest

from diatomic_ea_v09.canonical_dedup import (
    CanonicalBranchRelation,
)


class CanonicalDedupStaticTests(
    unittest.TestCase
):

    def test_relation_enum(self):

        self.assertEqual(
            CanonicalBranchRelation.SAME_BRANCH.value,
            "SAME_BRANCH",
        )


if __name__ == "__main__":
    unittest.main()
