# v0.9 Core Layer 01 validation

Status: PASS

## FeH root-continuity regression

Method:
- PBE / def2-QZVPD
- neutral
- spin 2S = 3
- seed R = 1.540 A
- one thread
- bidirectional density propagation

Result:
- 7 / 7 propagated points converged
- all points converged with ordinary SCF
- no level-shift or Newton helper required
- followed minimum moved from the independent-SCF region at 1.540 A
  to 1.570 A
- followed minimum energy:
  -1264.0384194112344 Eh
- S^2 remained smooth along the propagated branch

This reproduces the motivating FeH root-continuity regression without
using any v0.8 rescue position or jump information in the propagation
algorithm itself.

## MgH smooth-control regression

Method:
- PBE / def2-QZVPD
- neutral
- spin 2S = 1
- independent versus density-followed PEC
- R = 1.69 ... 1.81 A

Result:
- all independent calculations converged
- all followed calculations converged
- maximum absolute followed-independent energy difference:
  0.000001728 meV
- independent grid minimum: 1.75 A
- followed grid minimum: 1.75 A
- S^2 values were identical to the shown precision

Interpretation:
Density propagation changes the problematic FeH result while leaving a
smooth MgH control numerically unchanged.

Core Layer 01 is therefore accepted as the numerical propagation
foundation for further v0.9 development.
