# v0.9 local scout initial-guess failure policy

A molecular charge/spin/geometry sector is explored with multiple
independent initial guesses.

An individual initial guess is not itself a physical electronic state
and is not guaranteed to be available for every electron count and
spin sector.

Example observed in MgH/def2-QZVPD:

PySCF's Huckel initial-guess construction raised

    RuntimeError:
    Failed to assign mo_occ. nelec (8, 5) > Nmo (7)

because the Huckel minimal-orbital model did not contain enough MOs for
that spin occupation.

Policy:

1. A RuntimeError raised while constructing/running one individual
   initial guess is recorded for that fixed (R,2S) scout sector.

2. Remaining initial guesses are still attempted.

3. Successfully converged solutions from the sector are retained and
   grouped normally.

4. A failed guess is never interpreted as a failed spin sector.

5. If no guess produces a valid solution for a required spin-tail
   sector, the spin-frontier assessment sees missing data and does not
   silently close.

6. Unexpected non-RuntimeError exceptions remain fatal. This prevents
   programming/API errors from being hidden by the multistart logic.

This policy is molecule-independent.
