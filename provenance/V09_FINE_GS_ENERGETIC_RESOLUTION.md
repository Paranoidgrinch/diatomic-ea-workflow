# v0.9 fine ground-state energetic resolution

Electronic-state identity and ground-state energetic ordering are
separate questions.

FeH/PBE/def2-QZVPD produced two internally stable quartet UKS
solutions near the ground-state minimum.

Their electronic identity remained AMBIGUOUS under the conservative
density-based state classifier.

Grid-level 3/4/5 testing showed that the approximately 0.7 meV
splitting is reproducible and is not removed by increasing the DFT
integration grid.

Therefore the workflow does not loosen the state-identity thresholds.

Fine-level ground-state policy:

1. Retain every coarse branch that is electronically ambiguous with
   the lowest coarse candidate.

2. Compute an independent fine PEC for each retained candidate.

3. Compare electronic identity over multiple common fine-grid points.

4. Separately inspect pointwise energetic ordering using
      gap(R) = E_other(R) - E_primary(R).

5. If an electronically ambiguous alternative is higher than the
   primary candidate at every common point around the minimum, the
   ground-state energy is considered resolved.

   The alternative is retained in provenance as
   AMBIGUOUS_BUT_ENERGETICALLY_DOMINATED.

6. If the energy ordering changes sign or the PECs touch, ground-state
   resolution remains FINE_GS_AMBIGUOUS.

7. No empirical energy-gap cutoff is used to force this decision.

This rule does not assert that two ambiguous SCF solutions are the same
physical electronic state. It only permits a ground-state energetic
choice when the lower solution is consistently lower throughout the
relevant minimum region.
