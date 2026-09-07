# v0.9 coarse ground-state candidate policy

Post-stability canonical branch deduplication is deliberately
conservative.

FeH regression showed that internal SCF stability does not guarantee
that every same-spin solution collapses into a single canonical root.

Therefore:

1. The lowest canonical coarse PEC group is the primary ground-state
   candidate.

2. A group classified DISTINCT_BRANCH relative to the primary is not
   automatically carried into ground-state fine refinement.

3. Every group classified AMBIGUOUS_BRANCH relative to the primary is
   retained as an additional fine-PEC candidate.

4. SAME_BRANCH groups surviving complete-link deduplication are also
   retained rather than forcibly merged.

5. No ad hoc energy-gap threshold is used to resolve electronic
   identity.

6. Fine PECs plus pointwise stability/QC are used to resolve the
   retained candidate set.

7. If ambiguity remains after fine refinement, it is reported
   explicitly rather than removed by loosening identity thresholds.

FeH neutral regression:

- 14 raw branches
- 11 valid stability-canonicalized branches
- 5 conservative canonical groups
- lowest manifold: quartet, 2S=3
- two low quartet groups remain mutually AMBIGUOUS_BRANCH

The lower quartet group remains the primary energetic candidate, while
both quartet groups proceed to fine refinement.

The much higher doublet groups are distinct from the primary quartet
and need not receive fine ground-state PECs.
