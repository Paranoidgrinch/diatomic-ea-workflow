# v0.9 ground-state discovery architecture

Scope of this layer:
- neutral ground-state discovery
- anion ground-state discovery

Excited anion states are handled separately later.

For each charge:

1. Perform independent local multispin / multistart scouts at several
   universal seed geometries.

2. Group SAME_STATE local solutions.

3. Compare later seed solutions with already propagated branches at the
   same geometry:
   - SAME_STATE -> already represented
   - otherwise -> create and propagate another branch

4. Propagate every discovered branch bidirectionally across the coarse
   R grid using state-following.

5. Locate each raw coarse PEC minimum.

6. Reconstruct the state at its coarse minimum.

7. Apply identical bounded internal UKS/UHF stability optimization.

8. If stability changes the solution, use the stabilized solution as a
   new canonical seed and repropagate the entire coarse PEC.

9. Only finally stable canonical branches are eligible for ground-state
   selection.

10. Select the lowest canonical coarse PEC minimum as the charge
    ground-state candidate.

Development note:
Final production PECs will additionally receive pointwise stability/QC
checks. Minimum-level canonicalization here is the discovery-stage
mechanism.

No molecule-specific rescue logic is permitted.
