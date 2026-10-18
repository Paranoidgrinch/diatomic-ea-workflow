# diatomic-ea-workflow (v0.9)

A research workflow for electron affinities of diatomic molecules from state-resolved potential-energy
curves, with every decision documented. Version 0.9 sits between the two other EA projects:

| Project | Approach |
| --- | --- |
| [DiatomicEA](https://github.com/Paranoidgrinch/diatomic-ea) (v1.0) | fixed DFT ensemble with calibrated prediction intervals; fast, but without explicit tracking of electronic states |
| diatomic-ea-workflow (v0.9, this repository) | discovery and following of the neutral and anion ground states along the bond-length curve, with stability checks and provenance |
| [OpenEA Benchmark](https://github.com/Paranoidgrinch/openea-benchmark) | successor: state-aware coupled-cluster workflow with CBS limit and corrections |

This repository is superseded by OpenEA Benchmark and kept as the record of the v0.9 state-discovery design
and its FeH and MgH pilot studies.

## Method

For each charge state (neutral and anion):

1. independent multi-spin, multi-start state scouts at several seed geometries
2. grouping of identical solutions and bidirectional state following of every distinct branch over a coarse
   bond-length grid
3. location and reconstruction of each branch minimum, with bounded UKS/UHF stability optimization
4. fine potential-energy curves around the candidates, with point-wise quality checks
5. ground-state selection, harmonic zero-point energy and the adiabatic EA₀

The policies behind each step (discovery, spin, ambiguity, failed scouts, excited states) are written up in
`provenance/`.

## Layout

```text
src/diatomic_ea_v09/   workflow: scouts, state identity and following, coarse and fine curves, QC, vibrations
scripts/               pilot runs and regressions (FeH neutral and anion stages A to D, MgH end to end)
provenance/            design and policy notes per layer, source hashes
lists/                 molecule lists for validation and prediction
tests/                 unit tests
```

## Usage

Python 3.11 or newer with PySCF and NumPy.

```bash
pip install pyscf numpy pytest
PYTHONPATH=src pytest
PYTHONPATH=src python scripts/run_mgh_adaptive_e2e.py
```

## License

GPL-3.0, see [LICENSE](LICENSE).
