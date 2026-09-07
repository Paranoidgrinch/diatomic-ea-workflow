# v0.9 multi-candidate fine ground-state policy

If coarse canonical branch identity is ambiguous, all ambiguous
competitors to the lowest coarse candidate are retained into the
fine-PEC stage.

Each candidate receives independently:

1. reconstruction at its coarse minimum
2. internal stability verification
3. dense state-followed fine PEC
4. fine minimum determination
5. pointwise stability/QC in production

Fine candidate identity is then compared over multiple common points
around the fine minima.

Development comparison window:
+/- 0.05 A around either discrete fine minimum.

Development branch rule:
- any DISTINCT_STATE contradiction -> DISTINCT_BRANCH
- otherwise at least 5 SAME_STATE points -> SAME_BRANCH
- otherwise -> AMBIGUOUS_BRANCH

These are conservative development rules and are not method-frozen.

No energy-gap threshold is used to force electronic identity.

If ambiguity remains after fine PEC refinement it is reported
explicitly and both candidates remain available for further analysis.
