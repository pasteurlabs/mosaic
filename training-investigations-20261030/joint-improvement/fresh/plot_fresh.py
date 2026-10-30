"""Post-evaluation summaries only; never selects a model or parameter."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;out=root/'report';out.mkdir(exist_ok=True)
evaluation=root/'evaluation/outcome.json';data=json.loads(evaluation.read_text())
assert data['complete'],'Cannot publish complete-family means from incomplete fresh evaluation'
groups=sorted(set(m['group'] for m in data['models']));order=['velocity']+[g for g in groups if g!='velocity']
names={'velocity':'Velocity only','joint':'Joint, selected','joint_zero_mean':'Joint, heat constrained'}
colors={'velocity':'#2166ac','joint':'#d95f02','joint_zero_mean':'#238b45'}
for k,g in enumerate(order):
 if g not in colors:colors[g]=['#2166ac','#d95f02','#238b45'][k]
def label(g):return names.get(g,g)
plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.grid':True,'grid.alpha':.18})
def finish(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)
by={(m['group'],m['model_seed']):m for m in data['models']}
records=[]
for group in order:
 for seed in [0,1,2]:
  model=by[group,seed];velocity=by['velocity',seed]
  assert [c['ic_seed'] for c in model['cases']]==[101,102,103,104,105]
  row={'group':group,'model_seed':seed,'cases':[{'ic_seed':c['ic_seed'],'score':c['score'],'physics':c['physics']} for c in model['cases']]}
  for key in ['total','velocity_contribution','temperature_contribution']:
   a=np.mean([c['score'][key] for c in model['cases']]);b=np.mean([c['score'][key] for c in velocity['cases']])
   row[key]=float(a);row[key+'_ratio_to_velocity']=float(a/b)
  row['heat_drift_absolute_max_mean']=float(np.mean([c['physics']['heat_drift_absolute_max'] for c in model['cases']]))
  row['heat_drift_peak_percent_mean']=float(np.mean([100*c['physics']['heat_drift_absolute_max']/max(abs(c['physics']['heat_mean'][0]),1e-12) for c in model['cases']]))
  row['worst_temperature_undershoot']=float(max(c['physics']['undershoot'] for c in model['cases']))
  row['worst_temperature_overshoot']=float(max(c['physics']['overshoot'] for c in model['cases']))
  records.append(row)
summary={'complete':True,'selection_sha256':data['selection_sha256'],'evaluation_sha256':hashlib.sha256(evaluation.read_bytes()).hexdigest(),'fresh_test_seeds':[101,102,103,104,105],'model_seeds':[0,1,2],'records':records,'groups':[],'notes':['Each point averages the same five unseen ICs for one network initialization. Three points are three network initializations, not fifteen independent models.','Shared fixed training-set RMS normalization and equal velocity/temperature loss weights. Reported means use predicted frames1..100.','No test result is used to select models, hyperparameters, checkpoints or examples. Static field example is preregistered IC101/modelseed0/frame100.']}
for g in order:
 rr=[r for r in records if r['group']==g]
 summary['groups'].append({'group':g,'mean_fresh_loss':float(np.mean([r['total'] for r in rr])),'mean_paired_ratio_to_velocity':float(np.mean([r['total_ratio_to_velocity'] for r in rr])),'paired_seed_ratios':[r['total_ratio_to_velocity'] for r in rr],'mean_velocity_loss_ratio':float(np.mean([r['velocity_contribution_ratio_to_velocity'] for r in rr])),'mean_temperature_loss_ratio':float(np.mean([r['temperature_contribution_ratio_to_velocity'] for r in rr]))})
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,key,title in zip(axes,['total','velocity_contribution','temperature_contribution'],['Combined held-out error','Velocity contribution','Temperature contribution']):
 for i,g in enumerate(order):
  rr=[r for r in records if r['group']==g];values=[r[key+'_ratio_to_velocity'] for r in rr]
  ax.scatter(np.asarray([-0.13,0,.13])+i,values,color=colors[g],s=45,zorder=3)
  ax.scatter([i],[np.mean(values)],color='black',marker='_',s=380,zorder=4)
 ax.axhline(1,color='gray',ls='--');ax.set_xticks(np.arange(len(order)),[label(g).replace(', ','\n') for g in order],rotation=10);ax.set_title(title);ax.set_ylabel('Ratio to matched velocity-only model')
fig.suptitle('Fresh IC101–105 · each dot is one network seed averaged over five ICs',y=1.03)
finish(fig,'fresh_relative_errors')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,field,title in zip(axes,['frame_velocity_mse','frame_temperature_mse'],['Velocity normalized RMS error','Temperature normalized RMS error']):
 t=np.arange(101)*.04
 base=np.sqrt(np.mean([b['score'][field] for b in data['baselines']],axis=0))
 ax.plot(t,base,color='#555555',ls='--',lw=2,label='Uncorrected')
 for g in order:
  seedcurves=np.asarray([np.mean([c['score'][field] for c in by[g,s]['cases']],axis=0) for s in [0,1,2]])
  # Mean squared field error is aggregated first; square root follows.
  mean=np.sqrt(seedcurves.mean(axis=0));low=np.sqrt(seedcurves.min(axis=0));high=np.sqrt(seedcurves.max(axis=0))
  ax.plot(t,mean,color=colors[g],lw=2,label=label(g));ax.fill_between(t,low,high,color=colors[g],alpha=.12)
 ax.set_title(title);ax.set_xlabel('Physical time');ax.set_ylabel('Normalized RMS error');ax.legend(fontsize=8)
fig.suptitle('Five fresh ICs; band spans the three network-seed mean-error curves',y=1.03)
finish(fig,'fresh_error_rollouts')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,key,title,ylabel in zip(axes,['heat_drift_peak_percent_mean','worst_temperature_undershoot','worst_temperature_overshoot'],['Peak heat-integral drift','Worst temperature undershoot','Worst temperature overshoot'],['Mean over five ICs (%)','Maximum over five ICs','Maximum over five ICs']):
 for i,g in enumerate(order):
  values=[r[key] for r in records if r['group']==g]
  ax.scatter(np.asarray([-.13,0,.13])+i,values,color=colors[g],s=45);ax.scatter([i],[np.mean(values)],color='black',marker='_',s=380)
 ax.set_xticks(np.arange(len(order)),[label(g).replace(', ','\n') for g in order],rotation=10);ax.set_title(title);ax.set_ylabel(ylabel)
 if key=='heat_drift_peak_percent_mean':
  positive=[r[key] for r in records];assert min(positive)>0
  ax.set_yscale('log');ax.set_ylim(min(positive)*.4,max(positive)*2.5);ax.set_title('Peak absolute heat-integral drift');ax.text(.01,.98,'Log scale; signed drift in raw data',transform=ax.transAxes,ha='left',va='top',fontsize=8)
fig.suptitle('Physical diagnostics · each dot is one network initialization',y=1.03)
finish(fig,'fresh_heat_and_bounds')
# Fixed example, chosen before fresh evaluation.
seed=0;ic=101;field_files={}
for g in order:
 model=by[g,seed];case=next(c for c in model['cases'] if c['ic_seed']==ic);z=np.load(case['fields_path'])
 field_files[g]={'corrected':z['corrected'],'reference':z['reference'],'uncorrected':z['uncorrected']}
ref=field_files['velocity']['reference'][-1];unc=field_files['velocity']['uncorrected'][-1]
def collocated(x):
 uf=np.pad(x[...,0],((1,0),(0,0)));vf=np.pad(x[...,1],((0,0),(1,0)))
 return np.stack([.5*(uf[1:]+uf[:-1]),.5*(vf[:,1:]+vf[:,:-1])],axis=-1)
reference_fields=[ref[...,2],np.linalg.norm(collocated(ref),axis=-1)]
errors={}
for name,x in [('Uncorrected',unc)]+[(label(g),field_files[g]['corrected'][-1]) for g in order]:
 errors[name]=[abs(x[...,2]-ref[...,2]),np.linalg.norm(collocated(x)-collocated(ref),axis=-1)]
columns=['Reference']+list(errors);fig,axs=plt.subplots(2,len(columns),figsize=(3.1*len(columns),6))
for row,cmap,field_name in [(0,'Reds','Temperature'),(1,'Blues','Velocity magnitude')]:
 for col,name in enumerate(columns):
  ax=axs[row,col]
  if name=='Reference':
   values=reference_fields[row];vmax=float(values.max());title='Reference '+field_name.lower()
  else:
   values=errors[name][row];vmax=max(float(errors[x][row].max()) for x in errors);title=name+' error'
  im=ax.imshow(values.T,origin='lower',extent=(0,1,0,1),cmap=cmap,vmin=0,vmax=max(vmax,1e-12));ax.set_title(title,fontsize=9);ax.set_xlabel('x');ax.set_ylabel('y');fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.suptitle('Predeclared IC101 · network seed0 · T=4 · shared error scale within each row',y=1.02)
finish(fig,'fresh_fixed_example')
lines=['# Fresh evaluation after validation-only selection','','Five unseen plume ICs101–105, three network initialization seeds0–2. The recipe/checkpoint choices were frozen before this evaluation.','','| Group | Mean combined loss | Mean paired ratio to velocity-only | Three seed ratios |','|---|---:|---:|---|']
for g in summary['groups']:lines.append(f"| {label(g['group'])} | {g['mean_fresh_loss']:.7g} | {g['mean_paired_ratio_to_velocity']:.4f} | "+', '.join(f'{v:.4f}' for v in g['paired_seed_ratios'])+' |')
lines+=['','Each seed ratio compares five-IC mean loss to velocity-only with the same network seed. Black ticks are means of the three seed ratios. These are three model initializations, not fifteen independent model replicates. No significance tests or population claims are made.','', 'The mean-heat diagnostic uses float64 reductions of stored float32 fields. A zero-mean thermal residual preserves integrated heat to floating-point roundoff, not bitwise exactly, and does not guarantee temperature bounds.','', 'The static field example was fixed as IC101/networkseed0/frame100 before fresh evaluation; it was not chosen for visual impact. Reference panels show the restricted fine solution; all error panels share row scales.']
for name in ['fresh_relative_errors','fresh_error_rollouts','fresh_heat_and_bounds','fresh_fixed_example']:lines+=['',f'![{name}]({name}.png)']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['groups'],indent=2))
