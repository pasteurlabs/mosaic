# XLB reference timestep diagnostics

The original forced-flow reference failed temporal convergence at 24/48
substeps per coarse timestep (maximum restricted-field discrepancy 3.19%).
It remained finite and passed recurrent-state closure. No corrector was trained
on that inadmissible dataset.

On 2026-10-05, B200 jobs 2895286 and 2895300 compared identical saved post-burn
fields using the original adapter and a diagnostic copy that retains native
populations in float64 between calls. Both use the same float32 canonical
velocity, forcing and viscosity. These are direct-kernel diagnostics, not
replacement reference datasets. The complete arrays and source copies are in
`pr116-xlb-state-precision-20261005` on the cluster.

| Native state     | IC index |  24/48 |  48/96 | 96/192 | 192/384 |
| ---------------- | -------: | -----: | -----: | -----: | ------: |
| Original float32 |        0 | 3.431% | 1.710% | 0.849% |  0.385% |
| Original float32 |        1 | 3.713% | 1.848% | 0.915% |  0.407% |
| Retained float64 |        0 | 3.431% | 1.710% | 0.849% |  0.386% |
| Retained float64 |        1 | 3.713% | 1.848% | 0.915% |  0.405% |

Values are maximum full-192² relative velocity differences over 48 frames,
not the restricted-64² metric used for formal admission. Keeping float64 native
populations barely changes these discrepancies; timestep refinement reduces
them approximately linearly. The original adapter remains the transfer target.

Job 2895278 separately checks the actual RPC path at factors 12/24/48/96.
Fresh reference preparation uses 192/384, with 384/768 registered as the next
refinement if necessary. Every IC must still pass the unchanged 0.5% admission
gate, with the full time-75 burn-in. Coarse steps, physics, grids and the neural
training recipe remain unchanged.
