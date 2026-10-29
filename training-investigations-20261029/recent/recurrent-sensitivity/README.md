# Recurrent sensitivity followups (JAX-CFD only)

These are external diagnostics, not benchmark recipe changes. H8 and H16 use
matched seed-0 selected checkpoints from the same 60-second training campaign.
All original arrays/checkpoints/API are hash-checked. The cached trajectory
wrapper version differs, but physical solver API and reference arrays match.

Long-rollout plots propagate 24 orthonormal physical input tangents through
all12288 output coordinates. This samples24 of8065 physical state directions;
reported ratios are restricted-map condition numbers, not full condition numbers.
Solver-only and corrected paths start at the same fine-reference state and then
diverge autonomously. Separate shared-state S,C,C∘S controls, initial-identity
checks, full singular spectra and band leakage are in results/report.json.

Low/mid/high refer to spatial DCT radial bands <=4,4..16,>16. Velocity inputs
are discrete curls with closed normal walls and zero divergence. Three bases
select genuinely different mode subsets; they are not orthogonal rotations of
one span. The finite-amplitude check covers the first low-band velocity tangent
at two amplitudes, not every direction. Linearity and chain-rule checks are
reported separately. No training gradient gate was waived.

Dissipation legitimately makes high-mode singular gains small. A larger ratio
is not evidence of incorrect gradients; absolute gains and forecast errors must
be read together. Public JVP data are Float32, internal native arithmetic and
post-transfer SVD are Float64. The numerical rank cutoff is descriptive, not a
certified accuracy floor. Raw minima and ratios are retained.
