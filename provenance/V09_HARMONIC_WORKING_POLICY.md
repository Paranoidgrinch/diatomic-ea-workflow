# v0.9 harmonic working policy

Current development baseline:

- fine PEC spacing: 0.01 A
- harmonic local fit: 7 points
- fit interval around discrete minimum: +/- 0.03 A
- isotope masses are explicit inputs
- 5-, 9-, and 11-point fits are retained as sensitivity diagnostics

MgH pilot:
24Mg1H, PBE/def2-QZVPD

EA0 values:
- 5 points:  0.74946681 eV
- 7 points:  0.74946947 eV
- 9 points:  0.74947262 eV
- 11 points: 0.74947597 eV

Full 5-to-11-point spread:
approximately 9 micro-eV.

The 7-point rule is a development working baseline only.
It is not method-frozen until tested on the full pilot set.
