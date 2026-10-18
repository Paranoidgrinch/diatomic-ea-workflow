# diatomic-ea-workflow (v0.9)

An auditable research workflow for **electron affinities of diatomic molecules from state-resolved potential
energy curves**. Version 0.9 is the bridge between the two EA projects:

| Project | Approach |
| --- | --- |
| [DiatomicEA](https://github.com/Paranoidgrinch/diatomic-ea) (v1.0) | fixed DFT ensemble, calibrated prediction intervals — fast, but blind to which electronic state it lands in |
| **diatomic-ea-workflow** (v0.9, this one) | explicit discovery and following of neutral and anion ground states along the bond-length curve, with stability checks and full provenance |
| [OpenEA Benchmark](https://github.com/Paranoidgrinch/openea-benchmark) | the successor: state-aware coupled-cluster workflow with CBS limit and corrections — **use this one** |

> **Superseded by [OpenEA Benchmark](https://github.com/Paranoidgrinch/openea-benchmark).** This repository
> is kept as the documented record of the v0.9 state-discovery architecture and its FeH and MgH pilot
> studies.

[![License](https://img.shields.io/badge/license-GPL--3.0-394B59?style=flat-square)](LICENSE)

## What v0.9 does

For each charge state (neutral and anion):

1. independent multi-spin, multi-start **state scouts** at several seed geometries,
2. grouping of identical solutions and **bidirectional state following** of every distinct branch across a
   coarse bond-length grid,
3. location and reconstruction of each branch minimum with bounded **UKS/UHF stability** optimization,
4. **fine potential-energy curves** around the candidates with point-wise quality control,
5. ground-state selection, **harmonic zero-point energy** and the adiabatic EA₀.

Every policy decision is written down in `provenance/`, from the discovery architecture to the rules for
spin, ambiguity, scout failures and excited states.

## Layout

```text
src/diatomic_ea_v09/   the workflow: scouts, state identity and following, coarse/fine PECs, QC, vibrational
scripts/               pilot runs and regressions (FeH neutral/anion stages A–D, MgH end to end)
provenance/            design and policy documents for each layer, source hashes
lists/                 canonical molecule lists for validation and prediction
tests/                 static unit tests
```

## Running

Requires Python 3.11+ with PySCF and NumPy.

```bash
pip install pyscf numpy pytest
PYTHONPATH=src pytest
PYTHONPATH=src python scripts/run_mgh_adaptive_e2e.py
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
