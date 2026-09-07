# v0.9 charge ground-state pipeline

This is the integrated ground-state pipeline for one molecular charge.

Sequence:

1. independent multiseed / multispin / multiguess SCF discovery
2. local candidate grouping
3. coarse state-followed PEC generation
4. stability canonicalization near each coarse minimum
5. automatic coarse-R expansion
6. closed expansion/stability repropagation
7. post-stability canonical branch deduplication
8. conservative coarse ground-state candidate selection
9. fine PEC for every retained candidate
10. pointwise fine PEC reconstruction / stability / identity QC
11. fine electronic branch comparison
12. independent energetic ordering comparison

Important safety rules:

- a retained coarse candidate may not disappear silently because its
  fine calculation failed;
- ambiguous electronic identity does not automatically imply ambiguous
  ground-state energy;
- an electronically ambiguous alternative may be classified as
  energetically dominated only when the primary candidate is lower at
  every compared fine-grid point near the minimum;
- no empirical energy-gap cutoff is used to force state identity;
- unresolved ambiguity remains an explicit non-PASS status.

Possible successful charge-level outcomes:

PASS
    unique/resolved fine ground state

PASS_AMBIGUOUS_ALTERNATIVE_DOMINATED
    ground-state energy is resolved, but one or more higher fine
    candidates remain electronically ambiguous with the primary

FeH/PBE/def2-QZVPD is the canonical regression for the second case.
