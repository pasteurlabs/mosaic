import json,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path('/data/personal/andrinr/runner/results/mosaic/phiflow-periodic-skew-ssprk3-pr-20261007')
old=p.parent/'phiflow-periodic-ssprk3-pr-v2-20261007'
o=p/'report';o.mkdir(exist_ok=True)
summary={'cases':[],'scope':'Forced periodic 2D/3D; dt stability limits remain; 3D64 baseline sparse indexing error excluded from runtime/stability ratios.'}
fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
fields,fa=plt.subplots(4,2,figsize=(8,13),constrained_layout=True)
for index,(n,d) in enumerate([(64,2),(192,2),(32,3),(64,3)]):
 ax=axes.flat[index]
 for campaign,name,label,color in [(p,'baseline','Baseline Euler/CG','#b33b35'),(old,'candidate','SSPRK3 only','#da9b27'),(p,'candidate','SSPRK3 + skew','#276a9f')]:
  f=campaign/'results'/f'{name}-{d}d-{n}'/'outcome.json'
  data=json.loads(f.read_text());r=data['report'];traj=data['trajectory']
  if r['completed']: assert all(v['finite'] for v in traj), 'nonfinite completed outcome'
  t=np.array([v['time'] for v in traj]);e=np.array([v['energy'] for v in traj]);finite=np.isfinite(e)&(e>0)
  if finite.any():ax.plot(t[finite],e[finite],label=label,color=color,lw=1.8)
  if not r['completed']:
   ax.axvline(r['last_finite_time'],color=color,ls=':',alpha=.7)
  summary['cases'].append({**r,'label':label,'outcome_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 ax.set(title=f'{n}'+('²' if d==2 else '³'),xlabel='Physical time',ylabel='Kinetic energy',yscale='log',xlim=(0,75),ylim=(1e-3,100))
 ax.grid(alpha=.2);ax.legend(fontsize=8)
 with np.load(p/'results'/f'candidate-{d}d-{n}'/'fields.npz') as f:
  a=f['initial'];b=f['final']
  assert np.isfinite(a).all() and np.isfinite(b).all()
  diff=np.mean(b,axis=(0,1,2))-np.mean(a,axis=(0,1,2))
  summary.setdefault('mean_velocity_drift',[]).append({'n':n,'ndim':d,'components':diff.tolist(),'norm':float(np.linalg.norm(diff))})
  z=0 if d==2 else n//2
  aa=a[:,:,z,0];bb=b[:,:,z,0];limit=max(abs(aa).max(),abs(bb).max())
  for j,(v,label) in enumerate([(aa,'Initial'),(bb,'New method at T=75')]):
   im=fa[index,j].imshow(v.T,origin='lower',extent=(0,2*np.pi,0,2*np.pi),vmin=-limit,vmax=limit,cmap='RdBu_r')
   fa[index,j].set(title=f'{n}'+('²' if d==2 else '³ mid-z slice')+f' — {label}',xlabel='x',ylabel='y')
  fields.colorbar(im,ax=fa[index,:],label='x velocity',shrink=.85)
fig.suptitle('Long forced periodic flow: failures retained; finite curves stop at failure')
fig.savefig(o/'energy.png',dpi=170)
fields.suptitle('Full velocity fields / midplane slices; shared scale within each row')
fields.savefig(o/'fields.png',dpi=170)
summary['checks']=json.loads((p/'checks.json').read_text())
summary['runtime']=json.loads((p/'runtime.json').read_text())
(o/'report.json').write_text(json.dumps(summary,indent=2))
for filename in ['checks.json','runtime.json','candidate.py','baseline.py','validate.py','checks.py','runtime.py','report.py']:
 shutil.copy2(p/filename,o/filename)
print(json.dumps(summary['mean_velocity_drift']),flush=True)
