# v0.9 core layer 01

Implemented from scratch for the v0.9 workflow.

Scope:

- immutable molecule specification
- PySCF molecule construction
- basis/ECP loading from the installed PySCF library
- universal UKS execution
- level shift used only as a convergence helper
- Newton fallback
- ordinary-SCF polishing after helper calculations
- HOMO/LUMO and spin diagnostics
- density projection between neighboring geometries
- bidirectional propagation of one externally defined state ID

Explicitly NOT implemented in this layer:

- candidate-state discovery
- candidate deduplication
- electronic state identity criteria
- stability analysis
- stability-triggered repropagation
- automatic PEC expansion
- far-R controls
- ground-state selection
- harmonic or numerical ZPE
- EA or VDE analysis
- functional benchmarking
- basis benchmarking
- SOC

No v0.8 rescue task, jump position, selected minimum, or molecule-specific
recovery rule is consumed by this code.
