# v0.9 discovery seed and spin working policy

This is a development policy, not yet method-frozen.

## R seeds

Initial coarse grid:
0.10 A spacing.

Local state-discovery scouts:
0.20 A spacing, with 0.20 A margins from the initial coarse-grid ends.

For an initial 1.0-3.0 A coarse grid this gives:

1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8 A.

Every scout geometry is therefore also a coarse PEC point.

The relatively dense seed spacing is intentional during method
development. FeH demonstrated that some SCF roots can exist as distinct
solutions only over a limited R interval.

## Spin sectors

No local energy cutoff is used to terminate the spin search.

Start with the five lowest physically allowed 2S sectors.

The high-spin frontier is considered closed if the three highest tested
spin-sector minima rise strictly with increasing spin at every scout
geometry.

If this criterion is not met, add the next allowed spin sector and
reassess.

At most eight spin sectors are permitted in the development workflow.

If the frontier has not closed at the configured maximum, the charge
receives QC_FAIL_SPIN_FRONTIER rather than silently truncating the spin
search.

If all physically allowed spin sectors have been exhausted, the
frontier is closed by construction.

This policy is threshold-free with respect to electronic energy.
