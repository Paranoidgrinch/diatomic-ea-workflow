# v0.9 adaptive scout reuse

Local multistart SCF scouting is expensive and must not be repeated
between spin-frontier discovery and coarse branch discovery.

The integrated adaptive workflow therefore keeps CandidateGroup objects
in memory.

Structure:

adaptive local scout
    -> fixed-(R,spin) candidate groups
    -> adaptive spin-frontier decision
    -> combined candidate groups at each retained seed geometry
    -> branch discovery using the same groups
    -> coarse PEC propagation

discover_charge_branches retains its original behavior when no
precomputed groups are supplied.

When precomputed groups are supplied:

- local_scout_groups is not called;
- every requested seed must be present;
- missing cached seed data is an explicit error;
- spin_max filtering is retained.

The complete physical spin list and the development-limited spin list
are kept separately.

Reaching the configured maximum number of development spin sectors is
not equivalent to exhausting the physical spin space.

If the high-spin frontier remains open at the development maximum, the
workflow returns QC_FAIL_SPIN_FRONTIER.

Historical note:
the initial FeH spin-frontier diagnostic was run before persistent
ScoutSolution/CandidateGroup reuse was implemented. Its summarized CSV
cannot reconstruct the discarded AO densities, so that one diagnostic
cannot be reused as branch-following input.
