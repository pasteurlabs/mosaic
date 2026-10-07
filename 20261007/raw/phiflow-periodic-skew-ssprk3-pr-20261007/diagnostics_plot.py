import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/data/personal/andrinr/runner/results/mosaic/phiflow-periodic-skew-ssprk3-pr-20261007')
fig,axes=plt.subplots(4,2,figsize=(11,11),constrained_layout=True)
for index,(n,d) in enumerate([(64,2),(192,2),(32,3),(64,3)]):
 for variant,color in [('baseline','#b33b35'),('candidate','#276a9f')]:
  data=json.loads((p/'results'/f'{variant}-{d}d-{n}'/'outcome.json').read_text())
  rows=data['trajectory'];t=np.array([r['time'] for r in rows])
  for col,key in enumerate(['native_divergence_rms','max_courant']):
   v=np.array([r[key] for r in rows]);ok=np.isfinite(v)&(v>0)
   axes[index,col].plot(t[ok],v[ok],label=variant,color=color,lw=1.5)
   axes[index,col].set(xlim=(0,75),xlabel='Physical time',ylabel='Native divergence RMS' if col==0 else 'Maximum summed face Courant number',title=f'{n}'+('²' if d==2 else '³'))
   if col==0:axes[index,col].set_yscale('log');axes[index,col].set_ylim(1e-8,1e-2)
   else:axes[index,col].set_ylim(0,2)
   axes[index,col].grid(alpha=.2);axes[index,col].legend(fontsize=8)
fig.suptitle('Native MAC diagnostics; divergent tails clipped, all raw values retained')
(p/'report').mkdir(exist_ok=True)
fig.savefig(p/'report'/'diagnostics.png',dpi=160)
