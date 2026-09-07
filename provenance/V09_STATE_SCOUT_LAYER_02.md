# v0.9 State Scout Layer 02

Purpose:
Generate distinct self-consistent SCF candidates at a fixed geometry.

Universal initial guesses currently tested:
- minao
- atom
- huckel
- hcore

Pilot observations:

MgH/PBE/def2-QZVPD neutral spin 1:
all four guesses converged to the same numerical solution.
Maximum observed density distance:
6.6e-6.

FeH/PBE/def2-QZVPD neutral spin 3:
- minao produced a clearly distinct lower-energy solution
- huckel and hcore produced effectively the same solution
- atom was nearly degenerate in energy with huckel/hcore but had a
  substantially different density and is therefore conservatively
  retained as a separate candidate

Development-only duplicate criterion:
- same spin
- absolute energy difference <= 1 meV
- orthonormal spin-density distance <= 0.01

Both criteria must be satisfied.

These thresholds are NOT frozen production parameters.
They must be audited across the full development pilot before method freeze.

Different spin sectors are never deduplicated against one another.
Candidate IDs are bookkeeping labels, not spectroscopic assignments.
