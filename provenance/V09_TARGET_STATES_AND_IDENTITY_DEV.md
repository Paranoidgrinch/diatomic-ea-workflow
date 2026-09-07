# v0.9 target-state policy and state-identity development rule

## Final PEC targets per molecule / functional / basis

Neutral:
- ground state only

Anion:
- ground state
- first distinct excited state
- second distinct excited state

Normal final target:
4 PECs total.

Additional branches may be retained temporarily if the state identity or
energy ordering near the selection boundary is ambiguous.

## Adaptive discovery

The number of raw SCF solutions is not fixed.

The workflow continues generating/scouting candidates until it has enough
evidence for the required number of distinct electronic branches.

Duplicate SCF solutions do not count toward the target number of states.

AMBIGUOUS candidates are retained rather than discarded.

## Development state-equivalence criteria

Pairwise classification:
- SAME_STATE
- DISTINCT_STATE
- AMBIGUOUS

Different spin sectors:
- always DISTINCT_STATE

Development SAME_STATE criteria:
- |Delta E| <= 1 meV
- |Delta S^2| <= 1e-3
- max total-density-spectrum difference <= 1e-4
- max spin-density-spectrum difference <= 5e-4

All conditions must hold.

Development DISTINCT_STATE criteria:
- |Delta E| >= 5 meV
AND at least one of:
- |Delta S^2| >= 1e-2
- max total-density-spectrum difference >= 1e-3
- max spin-density-spectrum difference >= 5e-3

Everything else:
- AMBIGUOUS

These values are pilot parameters and are not method-frozen.

Raw AO density distance remains diagnostic only and is not used directly
for the equivalence decision because symmetry-related / orbital-rotated
representations can have large raw distances.

Pilot evidence motivating this policy:
- BN triplet multistart solutions have strongly different raw AO density
  representations but nearly identical invariant density spectra.
- FeH quartet atom/huckel/hcore solutions are similarly close in invariant
  descriptors.
- FeH quartet minao solution is clearly separated in energy, S^2, total
  density spectrum and spin density spectrum.
