# v0.9 branch coalescence and stability policy

## Branch coalescence

FeH quartet diagnostic:

A local huckel-derived candidate discovered at R=1.40 A is distinct
from both established quartet branches at short R.

On propagation toward larger R:
- DISTINCT at short R
- AMBIGUOUS near R=1.45-1.50 A
- SAME_STATE as established branch G01 from approximately R=1.54 A onward

Maximum-overlap occupation control (MOM) gives the same behavior.

Operational interpretation:
the separately identifiable SCF branch coalesces with G01 in this region.

A branch that becomes electronically identical to another tracked branch
is not counted twice beyond the coalescence region.

The event must remain recorded in branch provenance.

## Stability

Every retained tracked state receives the same bounded internal UKS/UHF
stability analysis.

Baseline:
- internal stability only
- external GHF/complex stability is not part of the initial production
  workflow
- maximum 6 stability/reoptimization iterations
- if an unstable solution changes root, the stabilized root is subsequently
  repropagated using the same universal branch algorithm

This is part of the standard method and is not a rescue mechanism.
