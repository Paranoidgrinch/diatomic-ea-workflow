# v0.9 FeH coarse branch regression

System:
- FeH neutral
- PBE
- def2-QZVPD
- quartet, 2S=3
- local scout seed R = 1.54 A

Local state scout produced two distinct quartet groups:

1. S3_G01
   - origin: minao
   - seed energy: -1264.038260447082 Eh

2. S3_G02
   - origins: atom, hcore, huckel
   - seed energy: -1264.037387152805 Eh

Both groups were independently propagated toward smaller and larger R
using projected densities.

Results:

S3_G01:
- coarse minimum R = 1.55 A
- E = -1264.038358771247 Eh
- minimum is interior
- consistent with previous fine state-following regression around R = 1.57 A

S3_G02:
- coarse minimum R = 1.54 A
- E = -1264.037387152805 Eh
- minimum is interior

The two branches remain DISTINCT_STATE throughout all common points from
1.30 to 2.10 A.

Their energetic ordering changes between 1.45 and 1.50 A:
- at 1.45 A S3_G02 is lower
- at 1.50 A S3_G01 is lower

This demonstrates that electronic branch identity must be transported by
state continuity and must not be defined by local energy rank.

Regression status:
PASS
