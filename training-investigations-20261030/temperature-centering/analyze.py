"""External CPU analysis: no training or admission changes."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.fft import dctn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
out=p/'report';out.mkdir(exist_ok=True)
jobs=json.loads((p/'jobs.json').read_text());rows=[];failures=[];metrics=[];spectra=[]
bands=['large (mode<4)','middle (4–12)','small (mode≥12)']
n=64;k=np.hypot(*np.meshgrid(np.arange(n),np.arange(n),indexing='ij'))
masks=[k<4,(k>=4)&(k<12),k>=12]
def coeff(x):
 return dctn(x,axes=(-3,-2),type=2,norm='ortho')
def benergy(x):
 cc=coeff(x)
 return np.stack([np.sum(cc[...,m,:]**2,axis=(-2,-1))/n**2 for m in masks],axis=-1)
def vel(x):
 uf=np.pad(x[...,0],((0,0),(1,0),(0,0)));vf=np.pad(x[...,1],((0,0),(0,0),(1,0)))
 return np.stack([.5*(uf[:,1:,:]+uf[:,:-1,:]),.5*(vf[:,:,1:]+vf[:,:,:-1])],axis=-1)
def phys(x):
 uv=vel(x);t=x[...,2];v=uv[...,1];y=(np.arange(n)+.5)/n
 return {'kinetic_energy':.5*np.mean(uv**2,axis=(1,2)).sum(axis=-1),'temperature_variance':t.var(axis=(1,2)),'vertical_advective_heat_flux':np.mean(v*t,axis=(1,2)),'thermal_centroid_y':np.sum(t*y[None,None,:],axis=(1,2))/np.sum(t,axis=(1,2)),'heat_mean':t.mean(axis=(1,2))}
for job in jobs:
 d=Path(job['directory']);f=d/'outcome.json'
 if not f.exists():failures.append({**job,'failure':'missing outcome'});continue
 r=json.loads(f.read_text())
 if not r.get('completed'):
  failures.append({**job,'failure':r.get('failure','incomplete'),'history_updates':len(r.get('history',[]))});continue
 ident={key:job[key] for key in ['seed','budget','arm']}
 row={**ident,'selected_update':r['selected_update'],'validation_loss':r['validation_loss'],'heldout_loss':r['selected_heldout_loss'],'completed_updates':r['completed_updates'],'training_seconds':r['training_seconds'],'fit_elapsed_s':r.get('fit_elapsed_s'),'offset_heldout':r['offset_heldout'],'research_fd_profiles':r['research_fd_profiles'],'outcome_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 rows.append(row)
 z=np.load(d/'fields.npz');ref=z['reference'].astype(float);base=z['uncorrected'].astype(float);sol=z['selected'].astype(float);pre=z['selected_pre_correction'].astype(float)
 states={'reference':ref,'uncorrected':base,'corrected':sol}
 pm={name:phys(x) for name,x in states.items()}
 for name,m in pm.items():metrics.append({**ident,'trajectory':name,**{key:np.asarray(value).tolist() for key,value in m.items()}})
 uv0=vel(pre);uv1=vel(sol[1:]);dv=uv1-uv0
 local_error_change=benergy(uv1-vel(ref[1:]))-benergy(uv0-vel(ref[1:]))
 before=.5*benergy(uv0);after=.5*benergy(uv1);cross=np.stack([np.sum((coeff(uv0)*coeff(dv))[...,mask,:],axis=(-2,-1))/n**2 for mask in masks],axis=-1);self_term=.5*benergy(dv)
 assert np.allclose(after-before,cross+self_term,rtol=1e-8,atol=1e-15)
 assert np.max(abs(sol[1:,...,2]-pre[...,2]))==0
 assert np.allclose(np.sum(before,axis=-1),phys(pre)['kinetic_energy'],rtol=1e-12,atol=1e-15)
 err={}
 for name,xx in [('uncorrected',base),('corrected',sol)]:
  err[name]={'velocity':benergy(vel(xx[1:])-vel(ref[1:])).mean(axis=0).tolist(),'temperature':benergy((xx[1:,...,2]-ref[1:,...,2])[...,None]).mean(axis=0).tolist()}
 spectra.append({**ident,'bands':bands,'energy_change_mean_per_correction':(after-before).mean(axis=0).tolist(),'energy_cross_term_mean':cross.mean(axis=0).tolist(),'energy_self_term_mean':self_term.mean(axis=0).tolist(),'local_velocity_squared_error_change':local_error_change.mean(axis=0).tolist(),'forecast_band_mse':err,'immediate_heat_flux_change_mean':float((phys(sol[1:])['vertical_advective_heat_flux']-phys(pre)['vertical_advective_heat_flux']).mean()),'immediate_temperature_change_max':0.,'heat_drift_abs_max':float(np.max(abs(pm['corrected']['heat_mean']-pm['corrected']['heat_mean'][0]))),'temperature_min':float(sol[...,2].min()),'temperature_max':float(sol[...,2].max())})
pairs=[]
for budget in ['fixed','budget']:
 for seed in [0,1,2]:
  rr=[r for r in rows if r['budget']==budget and r['seed']==seed]
  if len(rr)==2:
   a=next(r for r in rr if r['arm']=='baseline');b=next(r for r in rr if r['arm']=='temperature_centered')
   pairs.append({'budget':budget,'seed':seed,'validation_ratio_centered_over_baseline':b['validation_loss']/a['validation_loss'],'heldout_ratio_centered_over_baseline':b['heldout_loss']/a['heldout_loss']})
summary={'rows':rows,'paired_results':pairs,'failures':failures,'spectra':spectra,'notes':['JAX-CFD native RTX only; 3 model seeds, same data, single heldout IC.','Centered is velocity-only correction; does not correct temperature directly.','DCT-II orthonormal bands are descriptive closed-box spatial scales, not hydrodynamic eigenmodes or turbulent cascade flux.','Heat-flux proxy uses cell-centered velocity times temperature, not the native finite-volume Van Leer face flux. Velocity band MSE sums the two components per cell, unlike the mean-over-components normalized training loss; ratios and change signs are unaffected.','Reference physical fields are restricted N128 reference, not native fine-grid budgets.','Direct pre/post C energy increment includes cross term and correction self energy; verified exact Parseval and sum identity.','Offsets use symmetry-transformed reference, no new shifted fine simulation; native coarse covariance defect measured separately.','Research FD criterion distinct from public fixed-epsilon gate; all original failures retained.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
(out/'physical_time_series.json').write_text(json.dumps(metrics,allow_nan=False))
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':150})
colors={'baseline':'#8856a7','temperature_centered':'#008b8b','reference':'#222222','uncorrected':'#db7c26','corrected':'#008b8b'}
def save(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,budget in zip(axs,['fixed','budget']):
 for seed in [0,1,2]:
  rr=sorted([r for r in rows if r['budget']==budget and r['seed']==seed],key=lambda r:r['arm'])
  ax.plot([0 if r['arm']=='baseline' else 1 for r in rr],[r['heldout_loss'] for r in rr],'o-',label=f'seed {seed}')
 ax.set_xticks([0,1],['Original input','Centered input']);ax.set_ylabel('Held-out normalized trajectory loss');ax.set_title('256 updates' if budget=='fixed' else '60s fit including compile/validation');ax.legend()
save(fig,'paired_training')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,budget in zip(axs,['fixed','budget']):
 for arm in colors:
  if arm not in ['baseline','temperature_centered']:continue
  rr=[r for r in rows if r['budget']==budget and r['arm']==arm]
  for i,r in enumerate(rr):
   off=r['offset_heldout'];ax.plot([o['offset_scale_units'] for o in off],[o['selected_loss_to_symmetry_target'] for o in off],'o-',alpha=.65,color=colors[arm],label=arm if i==0 else None)
 ax.set_xlabel('Uniform temperature offset / training RMS');ax.set_ylabel('Held-out loss to symmetry-transformed reference');ax.set_title(budget);ax.legend()
save(fig,'offset_generalization')
fig,axs=plt.subplots(2,3,figsize=(14,7))
for ax,key,title in zip(axs.flat,['kinetic_energy','temperature_variance','vertical_advective_heat_flux','thermal_centroid_y','heat_mean','unused'],['Kinetic energy','Temperature variance','Collocated heat-flux proxy <vθ>','Thermal centroid height','Mean temperature','']):
 if key=='unused':ax.axis('off');continue
 for arm in ['baseline','temperature_centered']:
  for seed in [0,1,2]:
   for tr in ['reference','uncorrected','corrected']:
    rr=[m for m in metrics if m['budget']=='fixed' and m['seed']==seed and m['arm']==arm and m['trajectory']==tr]
    if not rr or (tr!='corrected' and (seed!=0 or arm!='baseline')):continue
    values=rr[0][key];label=(arm if tr=='corrected' else tr) if seed==0 else None
    ax.plot(np.arange(len(values))*.04,values,color=colors[arm] if tr=='corrected' else colors[tr],alpha=.6 if tr=='corrected' else 1,label=label)
 ax.set_title(title);ax.set_xlabel('Time');ax.legend(fontsize=8)
save(fig,'forward_physics')
fig,axs=plt.subplots(1,3,figsize=(15,4))
for arm,offset in [('baseline',-.16),('temperature_centered',.16)]:
 ss=[s for s in spectra if s['budget']=='fixed' and s['arm']==arm]
 if not ss:continue
 for ax,key,title in zip(axs,['energy_change_mean_per_correction','local_velocity_squared_error_change','immediate_heat_flux_change_mean'],['Direct correction Δ kinetic energy','Direct correction Δ velocity MSE','Direct correction Δ heat flux']):
  values=np.asarray([s[key] for s in ss])
  if values.ndim==1:
   ax.scatter(np.full(len(values),offset),values,color=colors[arm],label=arm);ax.set_xticks([0],['All scales'])
  else:
   for i in range(len(values)):ax.scatter(np.arange(3)+offset,values[i],color=colors[arm],label=arm if i==0 else None)
   ax.set_xticks(np.arange(3),['Large','Middle','Small'])
  ax.axhline(0,color='black',lw=.8);ax.set_title(title);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend(fontsize=8)
save(fig,'correction_budgets')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,field in zip(axs,['velocity','temperature']):
 for arm in ['baseline','temperature_centered']:
  for i,s in enumerate([s for s in spectra if s['budget']=='fixed' and s['arm']==arm]):
   a=np.asarray(s['forecast_band_mse']['uncorrected'][field]);b=np.asarray(s['forecast_band_mse']['corrected'][field])
   ax.plot(np.arange(3),b/a,'o-',alpha=.65,color=colors[arm],label=arm if i==0 else None)
 ax.set_xticks(np.arange(3),['Large: <4','Middle: 4–12','Small: ≥12']);ax.axhline(1,color='black',ls='--');ax.set_yscale('log');ax.set_ylabel('Corrected / uncorrected band MSE');ax.set_title(field+' forecast errors');ax.legend()
save(fig,'forecast_scales')
lines=['# Temperature symmetry and physical correction budgets','','JAX-CFD, three paired model seeds; original versus centered temperature input. Velocity-only curl correction in both arms.','', '| Budget | Seed | Centered/original validation | Centered/original heldout |','|---|---:|---:|---:|']
for r in pairs:lines.append(f"| {r['budget']} | {r['seed']} | {r['validation_ratio_centered_over_baseline']:.4f} | {r['heldout_ratio_centered_over_baseline']:.4f} |")
lines+=['','Ratios below one favor centered inputs. Validation alone selects checkpoints; all offset evaluations are test-only.','',f'Completed {len(rows)}/{len(jobs)} cells; failures: {len(failures)}. All missing/rejected cells remain in summary.json.','', 'Budgets: fixed 256 updates with validation cadence64; matched60s includes full training compile and validation, excludes shared preparation/gates. Native RTX research, not exact CI. Dual gradient diagnostics retained equally. Persistent compilation cache disabled.','', 'The research gate requires the same adjacent epsilon pair (.003/.001/.0003) to satisfy original5% AD/FD tolerance and5% slope agreement at both initial and exact synthetic nonzero parameters. Original fixed-epsilon failures remain recorded. No production gate or recipe changes.','','DCT-II orthonormal bands describe closed-box spatial scales; these are not divergence-free hydrodynamic eigenmodes or cascade flux. Direct correction energy includes base–correction cross term and self energy. Parseval and energy identities are checked. Reference physics uses restricted fine-grid fields.','','Offsets transform the admitted reference using physical symmetry. This is not a fresh shifted fine solver run; native coarse symmetry defects are recorded independently. One test IC and three model seeds do not establish broad generalization.']
for name in ['paired_training','offset_generalization','forward_physics','correction_budgets','forecast_scales']:lines+=['',f'![{name}]({name}.png)']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'completed_cells':len(rows),'failures':failures,'paired_results':pairs},indent=2))
