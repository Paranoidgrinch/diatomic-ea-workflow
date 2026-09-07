# v0.9 ground-state versus excited-state policy

## Ground states

Neutral and anion ground states are identified using:

1. multispin / multistart SCF discovery
2. state-continuous coarse PEC tracking
3. internal UKS/UHF stability analysis
4. repropagation from the stabilized solution when stability changes root
5. selection of the lowest valid stable PEC

Stability optimization is therefore part of ground-state
canonicalization.

## Excited anion states

The first and second excited anion states are NOT defined as the second
and third lowest raw SCF stationary solutions.

Internal ground-state stability optimization must not be used to erase a
deliberately state-specific excited-state solution merely because a
lower-energy orbital rotation exists.

Planned production target:

- neutral ground-state PEC
- anion ground-state PEC
- anion first excited-state PEC
- anion second excited-state PEC

Preferred excited-state route to benchmark:
linear-response TDA/TDDFT from the stable anion UKS reference.

State-specific MOM delta-SCF remains a diagnostic / sensitivity option.

Excited-state identity across R must be tracked separately, e.g. using
transition-vector / NTO character, rather than local excitation-energy
rank alone.
