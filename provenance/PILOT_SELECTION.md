# Paper-1 v0.9 development pilot

This pilot is used to develop the universal workflow architecture before
functional and basis benchmarking.

It is not a separate validation/holdout population. All six molecules are
members of the canonical 60-molecule validation set.

## Molecules

- FeH — root-continuity regression; additional stability/boundary behaviour
- TiH — strong stability-changing validation case
- NbC — independent minimum-jump/state-continuity case
- FeCl — selected-primary-EA boundary-minimum regression case
- MgH — comparatively simple short-hydride control
- BN — light main-group control

## Provisional development electronic-structure level

PBE / def2-QZVPD

This combination is used only to develop and regression-test the v0.9
workflow architecture. It is not frozen as the final production method.

Functional and basis benchmarking will be performed only after the
state-continuous core workflow is operational.
