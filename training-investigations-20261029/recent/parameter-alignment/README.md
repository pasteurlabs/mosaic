# Neural parameter-gradient alignment

JAX-CFD N64 only. Checkpoints come from the matched 60-second native-GPU
horizon pilot, not the public 256-update benchmark models. Compare raw parameter gradients of H8/H16/H32 prefix losses
with the H100 mean-intermediate-state objective, at identical initial states,
normalization and network parameters. Four checkpoints (initial, H8 selected,
H8 final, H16 selected) and two ICs (training0, held-out100) are retained.
This is a controlled trajectory-prefix comparison, not the actual shuffled
training-window gradient distribution and not clipped Adam update directions.
Held-out data are descriptive; no recipe was selected from these diagnostics.

At the H8 selected checkpoint on held-out IC100, cosine alignment with H100 is
−.524, −.240 and +.611 for H8/H16/H32. Unit-gradient steps of magnitude1e-4
change the H100 loss by +1.26%, +.57% and −1.17%, respectively. This demonstrates
an objective mismatch in that case, not a universal benefit of longer horizons.
Other checkpoints are nonmonotonic. A near-orthogonal direction can increase
finite-step loss despite a weakly favorable first derivative.

All8 case/checkpoint evaluations completed. The two initial-model H100 FD
checks at epsilon.003 exceed5% (5.16% and6.53%); those points are preserved.
At epsilon.0003 their errors fall to about.067%. This is a diagnostic epsilon
profile, not a retroactive change to any training admission gate. H8/H16 scan
objectives and parameter gradients were checked against the original unrolled
implementation. All176 recorded forward invariant summaries were finite,
with maximum sampled CFL.242857 and zero wall padding.

[Alignment](alignment.png), [measured versus first-order descent](descent.png),
and [full H100 FD profile](h100_fd.png) have matching interactive Vega-Lite files.
The full report records source/API/data/checkpoint hashes and every case,
finite-step probe, gradient norm and physical check. Large gradients and NPZ
arrays remain external; they are not included in this artifact publication.
