"""Render only completed fixed Warp stability comparison cells on allocated CPU."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(sys.argv[1])
out=root/'report';out.mkdir(exist_ok=True)
rows={cell:json.loads((root/'results'/cell/'outcome.json').read_text()) for cell in ['baseline-3d-32','candidate-3d-32','baseline-3d-64','candidate-3d-64']}
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
summary={}
for col,n in enumerate([32,64]):
 for method,color in [('baseline','#c44e52'),('candidate','#4c72b0')]:
  cell=f'{method}-3d-{n}';r=rows[cell]
  finite=[x for x in r['trace'] if x.get('finite')]
  for ax,key in [(axes[0,col],'energy'),(axes[1,col],'divergence_rms')]:
   ax.plot([v['time'] for v in finite],[v[key] for v in finite],label=method,color=color)
   ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
   ax.set_ylim((1e-2,1e3) if key=='energy' else (1e-9,1e3))
   if not r['burn_completed']:
    ax.axvline(r['trace'][-1]['time'], color=color, linestyle=':', alpha=.7)
  axes[0,col].set_title(f'N={n}, dt={r["dt"]:.6g}')
  axes[1,col].set_xlabel('Physical time')
  failure=next((x for x in r['trace'] if not x.get('finite',False)),None)
  summary[cell]={k:r[k] for k in ['adapter_sha256','initial_sha256','burn_completed','burn_wall_s','cold_forward_s','warm_forward_s','cold_forward_and_vjp_s','warm_forward_and_vjp_s','gradient_checks','shear_relative_error','zero_steps_exact','split_join_relative_error','taped_forward_max_error','divergent_input_stats','divergent_output_stats']}
  summary[cell]['first_failed_time']=None if failure is None else failure['time']
  summary[cell]['last_finite_time']=finite[-1]['time']
 axes[0,col].set_ylabel('Mean kinetic energy');axes[1,col].set_ylabel('Centered divergence RMS')
fig.suptitle('New forced 3-D stress test: FFT-fixed Euler baseline vs projected SSPRK3/skew candidate\nDotted lines: first nonfinite time; extreme divergent values exceed plotted range')
fig.tight_layout();fig.savefig(out/'stability-traces.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,key,label in [(axes[0],'warm_forward_s','Forward'),(axes[1],'warm_forward_and_vjp_s','Forward + VJP')]:
 labels=list(rows);means=[np.mean(rows[k][key]) for k in labels]
 ax.bar(labels,means,color=['#c44e52','#4c72b0']*2)
 ax.set_ylabel('Seconds per call');ax.set_title(label);ax.tick_params(axis='x',rotation=20)
fig.tight_layout();fig.savefig(out/'costs.png',dpi=160);plt.close(fig)
old=root/'results'/'old2d-2d-64';new=root/'results'/'candidate-2d-64'
summary['2d_generalization_parity']={f:bool(np.array_equal(np.load(old/f),np.load(new/f))) for f in ['initial.npy','final.npy']}
for n in [32,64]:
 if rows[f'baseline-3d-{n}']['initial_sha256'] != rows[f'candidate-3d-{n}']['initial_sha256']:
  raise ValueError('Initial conditions differ')
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
(out/'README.md').write_text('This is a new forced periodic 3-D stress test, not evidence that previous benchmark cases failed. Baseline includes #196 FFT normalization repair. Candidate additionally projects initial input and uses centered-compatible pressure projection, skew advection and SSPRK3. Both start from the same seed116 vector-potential Fourier field per resolution. Timing includes direct adapter forward execution and synchronized GPU work, excludes RPC. Warm VJP timing includes forward tape construction and reverse execution. Primary forward/VJP comparison is within each resolution on B200; raw hardware files are retained. No training claim follows from this diagnostic.\n')
