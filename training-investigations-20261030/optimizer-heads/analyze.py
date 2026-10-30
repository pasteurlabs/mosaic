"""Scalar-only publication/report after optimizer jobs. Run on Slurm CPU."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;out=p/'report';out.mkdir(exist_ok=True)
protocol=json.loads((p/'PROTOCOL.json').read_text());jobs=json.loads((p/'jobs.json').read_text())
runs=[]
for job in jobs:
 d=Path(job['directory']);f=d/'outcome.json'
 r=json.loads(f.read_text()) if f.exists() else {'completed':False,'failure':{'message':'Missing outcome'}}
 recipe=r.get('recipe',json.loads((p/'recipes'/(job['id']+'.json')).read_text()))
 runs.append({'id':job['id'],'job':job,'recipe':recipe,'outcome':r,'sha256':hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None})
optimizers=['adam','adamw','muon','sgd','lbfgs'];colors=dict(zip(optimizers,['#1764ab','#c34800','#2d8a42','#9343a5','#343434']))
def eligible(r):
 return r['outcome'].get('completed') and r['outcome'].get('selected_validation_loss') is not None
selection={};selection_grid={}
for opt in optimizers:
 grid=[]
 lrvalues=sorted({r['recipe']['lr'] for r in runs if r['recipe']['optimizer']==opt and r['recipe'].get('phase')=='screen'})
 for lr in lrvalues:
  rr=[r for r in runs if r['recipe']['optimizer']==opt and r['recipe']['lr']==lr and r['recipe'].get('phase')=='screen']
  ok=len(rr)==2 and all(eligible(r) for r in rr)
  score=math.exp(sum(math.log(r['outcome']['selected_validation_loss']) for r in rr)/2) if ok else None
  grid.append({'lr':lr,'complete_both_modes':ok,'validation_geometric_mean':score,'ids':[r['id'] for r in rr]})
 valid=[g for g in grid if g['complete_both_modes']]
 if valid:selection[opt]=min(valid,key=lambda x:(x['validation_geometric_mean'],x['lr']))
 selection_grid[opt]=grid
def selected_run(opt,joint,scaled=False):
 lr=selection.get(opt,{}).get('lr')
 return next((r for r in runs if r['recipe']['optimizer']==opt and r['recipe']['lr']==lr and r['recipe']['joint']==joint and (r['recipe'].get('phase')=='screen_scaled')==scaled),None)
def xy(r,field,axis='work'):
 checks=r['outcome'].get('checkpoints',[])
 x=[c['optimizer_window_evaluations'] if axis=='work' else c['fit_elapsed_s'] for c in checks]
 y=[c['loss'][field] for c in checks]
 return x,y
def save(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)
plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.grid':True,'grid.alpha':.18})
for axis in ['time','work']:
 fig,axs=plt.subplots(2,3,figsize=(15,8))
 for row,joint in enumerate([False,True]):
  for col,field in enumerate(['total','velocity','temperature']):
   ax=axs[row,col]
   for opt in optimizers:
    for r in [r for r in runs if r['recipe']['optimizer']==opt and r['recipe']['joint']==joint and r['recipe'].get('phase')=='screen']:
     x,y=xy(r,field,axis);ax.plot(x,y,color=colors[opt],alpha=.17,lw=.8)
    r=selected_run(opt,joint)
    if r:
     x,y=xy(r,field,axis);ax.plot(x,y,'o-',ms=2.7,lw=1.5,color=colors[opt],label=f"{opt.upper()} LR {r['recipe']['lr']:g}")
   ax.set_yscale('log');ax.set_title(('Joint' if joint else 'Velocity only')+': '+field+' validation');ax.set_ylabel('Normalized loss contribution')
   ax.set_xlabel('Fit seconds (compile, validation, diagnostics)' if axis=='time' else 'Optimizer window evaluations (incl. line search)')
   if col==0:ax.legend(fontsize=8)
 fig.suptitle('JAX-CFD • seed 0 • same four fixed H8 windows • faint: all rates; bold: shared validation-selected rate',fontsize=11)
 save(fig,'validation_'+axis)
fig,axs=plt.subplots(5,2,figsize=(13,17))
for i,opt in enumerate(optimizers):
 for col,joint in enumerate([False,True]):
  ax=axs[i,col]
  for r in [r for r in runs if r['recipe']['optimizer']==opt and r['recipe']['joint']==joint]:
   x,y=xy(r,'total','work');scaled=r['recipe'].get('phase')=='screen_scaled';status='' if eligible(r) else ' [failed/incomplete]'
   line=ax.plot(x,y,'o--' if scaled else 'o-',ms=2,label=f"LR {r['recipe']['lr']:g}"+(' scaled' if scaled else '')+status)[0]
   if not eligible(r) and x:ax.scatter([x[-1]],[y[-1]],marker='x',s=75,lw=2,color=line.get_color(),zorder=5)
  ax.set_yscale('log');ax.set_title(opt.upper()+(' joint' if joint else ' velocity only'));ax.set_xlabel('Optimizer window evaluations');ax.set_ylabel('Validation total');ax.legend(fontsize=8)
save(fig,'all_lr_validation')
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,joint in zip(axs,[False,True]):
 for opt in optimizers:
  r=selected_run(opt,joint)
  if not r:continue
  history=r['outcome'].get('history',[]);accepted=[h for h in history if h.get('accepted')]
  ax.plot([h['optimizer_window_evaluations'] for h in accepted],[h['loss'] for h in accepted],'o-',ms=2,color=colors[opt],label=opt.upper())
  trials=[h for h in history if not h.get('accepted')]
  if trials:ax.scatter([h['optimizer_window_evaluations'] for h in trials],[h['loss'] for h in trials],marker='x',s=15,color=colors[opt],alpha=.35)
 ax.set_yscale('log');ax.set_title('Joint' if joint else 'Velocity only');ax.set_xlabel('Optimizer window evaluations');ax.set_ylabel('Fixed training objective (accepted solid; trials ×)');ax.legend(fontsize=8)
save(fig,'training_accepted_and_trials')
fig,axs=plt.subplots(2,2,figsize=(12,8))
for col,joint in enumerate([False,True]):
 for opt in optimizers:
  r=selected_run(opt,joint)
  if not r:continue
  rr=r['outcome'].get('gradient_diagnostics',[])
  x=[g['oracle_calls']*4 for g in rr]
  ratios=[g['blocks']['shared_trunk']['temperature_loss_gradient_norm']/g['blocks']['shared_trunk']['velocity_loss_gradient_norm'] if g['blocks']['shared_trunk']['velocity_loss_gradient_norm']>1e-20 else np.nan for g in rr]
  cos=[g['blocks']['shared_trunk']['cosine'] if g['blocks']['shared_trunk']['cosine'] is not None else np.nan for g in rr]
  axs[0,col].plot(x,ratios,'o-',ms=2,color=colors[opt],label=opt.upper());axs[1,col].plot(x,cos,'o-',ms=2,color=colors[opt])
 axs[0,col].set_yscale('log');axs[0,col].set_title(('Joint' if joint else 'Velocity only')+' shared trunk');axs[0,col].set_ylabel('||∇Ltemperature|| / ||∇Lvelocity||');axs[0,col].legend(fontsize=8)
 axs[1,col].set_ylabel('Component-gradient cosine');axs[1,col].set_ylim(-1.05,1.05);axs[1,col].axhline(0,color='black',lw=.7);axs[1,col].set_xlabel('Optimizer window evaluations')
save(fig,'shared_trunk_gradients')
fig,axs=plt.subplots(2,2,figsize=(12,8))
for col,opt in enumerate(['adam','sgd']):
 for scaled,color in [(False,colors[opt]),(True,'#008c8c')]:
  r=selected_run(opt,True,scaled)
  if not r:continue
  label='fixed readout scaling' if scaled else 'original joint'
  for row,axis in enumerate(['work','time']):
   x,y=xy(r,'total',axis);axs[row,col].plot(x,y,'o-',ms=2,color=color,label=label)
   axs[row,col].set_yscale('log');axs[row,col].set_title(opt.upper()+f" common LR {r['recipe']['lr']:g}");axs[row,col].set_ylabel('Joint total validation');axs[row,col].set_xlabel('Optimizer window evaluations' if axis=='work' else 'Fit seconds (calibration charged for scaled)');axs[row,col].legend(fontsize=8)
save(fig,'head_scaling')
summaryrows=[]
for opt in optimizers:
 for joint in [False,True]:
  r=selected_run(opt,joint)
  if not r:continue
  o=r['outcome']
  summaryrows.append({'optimizer':opt,'joint':joint,'lr':r['recipe']['lr'],'id':r['id'],'validation':o['selected_validation_loss'],'heldout':o['selected_heldout'],'selected_oracle_call':o['selected_oracle_call'],'optimizer_oracle_calls':o['optimizer_oracle_calls'],'accepted_iterations':o['accepted_iterations'],'fit_elapsed_s':o['fit_elapsed_s'],'gradient_diagnostic_window_gradients':o['gradient_diagnostic_window_gradients'],'physics':o['physics']['selected'],'selected_initial_identity':o['selected_oracle_call']==0})
scale_rows=[]
for opt in ['adam','sgd']:
 a=selected_run(opt,True);b=selected_run(opt,True,True)
 if a and b and eligible(a) and eligible(b):
  scale_rows.append({'optimizer':opt,'lr':a['recipe']['lr'],'original_id':a['id'],'scaled_id':b['id'],'validation_ratio_scaled_over_original':b['outcome']['selected_validation_loss']/a['outcome']['selected_validation_loss'],'heldout_ratio_scaled_over_original':b['outcome']['selected_heldout']['total']/a['outcome']['selected_heldout']['total'],'charged_calibration_seconds':b['outcome'].get('charged_calibration_seconds'),'scaled_physics':b['outcome']['physics']['selected']})
paired=[]
for opt in optimizers:
 a=selected_run(opt,False);b=selected_run(opt,True)
 if a and b and eligible(a) and eligible(b):paired.append({'optimizer':opt,'lr':a['recipe']['lr'],'validation_ratio_joint_over_velocity':b['outcome']['selected_validation_loss']/a['outcome']['selected_validation_loss'],'heldout_ratio_joint_over_velocity':b['outcome']['selected_heldout']['total']/a['outcome']['selected_heldout']['total']})
failures=[{'id':r['id'],'recipe':r['recipe'],'failure':r['outcome'].get('failure'),'checkpoints':r['outcome'].get('checkpoints',[])} for r in runs if not eligible(r)]
gradient_summary=[]
for opt in optimizers:
 for joint in [False,True]:
  r=selected_run(opt,joint)
  if not r:continue
  rr=r['outcome'].get('gradient_diagnostics',[])
  valid=[g['blocks']['shared_trunk'] for g in rr if g['blocks']['shared_trunk']['cosine'] is not None]
  ratios=[g['temperature_loss_gradient_norm']/g['velocity_loss_gradient_norm'] for g in valid if g['velocity_loss_gradient_norm']>1e-20]
  cos=[g['cosine'] for g in valid]
  selected_diag=next((g for g in rr if g['oracle_calls']==r['outcome']['selected_oracle_call']),None)
  gradient_summary.append({'optimizer':opt,'joint':joint,'coordinate_scope':'raw trainable coordinates, full fixedbatch loss components, shared hidden trunk only','nonzero_diagnostic_points':len(valid),'temperature_over_velocity_norm_ratio_median':float(np.median(ratios)) if ratios else None,'temperature_over_velocity_norm_ratio_final':ratios[-1] if ratios else None,'negative_cosine_fraction':float(np.mean(np.asarray(cos)<0)) if cos else None,'cosine_range':[min(cos),max(cos)] if cos else None,'selected_model_oracle_call':r['outcome']['selected_oracle_call'],'selected_model_trunk_metrics':selected_diag['blocks']['shared_trunk'] if selected_diag else None})
report={'complete':len(runs)==len(protocol['recipes']) and all(r['outcome'].get('completed') or r['outcome'].get('failure') for r in runs),'total_cells':len(runs),'completed_cells':sum(bool(r['outcome'].get('completed')) for r in runs),'failures':failures,'shared_lr_selection':selection,'selection_grid':selection_grid,'selected_rows':summaryrows,'joint_vs_velocity':paired,'head_scaling':scale_rows,'component_gradient_summary':gradient_summary,'manifest':[{'id':r['id'],'directory':r['job']['directory'],'slurm_id':r['job']['slurm_id'],'outcome_sha256':r['sha256']} for r in runs],'limits':['One model seed0, one validation trajectory and one heldout trajectory; exploratory screen, no causal proof or optimizerpopulationranking.','Same fixed4windows objective differs from publicshuffledtraining and can overfit.','Shared LR selected with validation geometricmean across both modes, never test. All originalgatefailures and rejectedtrials retained.','Each optimizer capped64 objective-gradient calls (256windoweval), includesALL line search. Componentgradientdiagnostics counted separately; wallclockincludesfirstfitcompile and diagnostics/validation.','NativeRTX, not coldcontainer/T4CI. Common data/native/gatepreparation excluded. Scaledfitcharges16.359s calibration.','Velocity and temperature losses are already normalized by training RMS; remaining sensitivity/coupling does not establish Adam-caused imbalance.','Fixedheadreparameterization is not HIG, inversephysics or conservationpreserving.','Componentgradient norms use raw trainable coordinates; scaled/unscaled gradients not directly identicalphysicalcovectors. Effectiveheadstepnorms accountforfixed factors.','SGD both modes and Muonjoint select initial identity: ratio1 is no validation benefit, not learnedparity.','Failure crosses mark last measured checkpoint before later rejection, not unsafe-statevalidation.']}
(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
raw=out/'raw';raw.mkdir(exist_ok=True)
for r in runs:(raw/(r['id']+'.json')).write_text(json.dumps(r['outcome'],indent=2,allow_nan=False))
lines=['# Optimizer and correction-head screen','','JAX-CFD native RTX5090, seed0, same fixed batch of four H8 windows. Velocity-only versus joint velocity/temperature correction; unchanged RMS-normalized physical objective. This differs from the public shuffled-window recipe.','',f'Completed {report["completed_cells"]}/{report["total_cells"]} cells; explicit failed cells: {len(failures)}.','', '| Optimizer | Shared LR | Joint/velocity validation | Joint/velocity heldout |','|---|---:|---:|---:|']
for r in paired:lines.append(f"| {r['optimizer']} | {r['lr']:g} | {r['validation_ratio_joint_over_velocity']:.4f} | {r['heldout_ratio_joint_over_velocity']:.4f} |")
lines+=['','Ratios below1 favor joint correction. SGD both modes and Muon joint selected the initial identity model; ratio1 is not learned parity. The LR is shared across both modes and chosen only by geometricmean validation; test never chooses LR or checkpoint.','', '| Scaled joint comparison | Shared LR | Scaled/original validation | Scaled/original heldout |','|---|---:|---:|---:|']
for r in scale_rows:lines.append(f"| {r['optimizer']} | {r['lr']:g} | {r['validation_ratio_scaled_over_original']:.4f} | {r['heldout_ratio_scaled_over_original']:.4f} |")
lines+=['','Hard64 optimizer objective-gradient calls, including all L-BFGS line-search trials, each4windows. Additional component-gradient diagnostics are separately counted. Fit clock includes first optimizer compilation, optimizer updates, validation and diagnostics; common reference/native/gate preparation excluded. The fixed scaling calibration16.359s is conservatively charged per scaled run. Native RTX timings are not exact CI forecasts.','','The readout scaling [1,0.2139727633] is fixed from training-only full-H8 directional-response calibration, not HIG. It changes parameterization while retaining the forward-function family and loss; it does not impose heat conservation or bounds.','','Component-gradient analysis separates shared trunk from distinct readout blocks. At zero-readout initialization the shared-trunk gradients vanish, so ratios/cosines are masked rather than treated as evidence. Norm imbalance and direction conflict are descriptive; neither alone proves why an optimizer wins.','','Gradient norms use raw trainable coordinates, which the head scaling changes; effective head step norms are retained separately. Failure crosses mark the last measured validation checkpoint before later rejection, not validation of an unsafe model.','','All original fixed-epsilon gradient checks, convergence profiles, failures, training/validation traces, physics checks and unsafe-trial records remain in raw/. The research convergence gate does not replace the production gate.','']
for name in ['validation_time','validation_work','training_accepted_and_trials','shared_trunk_gradients','head_scaling','all_lr_validation']:lines += [f'![{name}]({name}.png)','']
lines+=['## Shared-trunk component-gradient diagnostics','','These summarize nonzero checkpoints along the trained path, not only the selected model. Initial shared gradients vanish; selected identity models have zero shared gradients.','', '| Optimizer | Mode | Median thermal/velocity norm | Fraction negative cosine |','|---|---|---:|---:|']
for g in gradient_summary:
 lines.append('| {} | {} | {:.4g} | {:.3f} |'.format(g['optimizer'],'joint' if g['joint'] else 'velocity',g['temperature_over_velocity_norm_ratio_median'],g['negative_cosine_fraction']))
if failures:
 lines+=['## Failed or incomplete cells','']
 for r in failures:lines.append('- '+r['id']+': '+str(r['failure']))
lines+=['','One seed and one test IC support a pilot conclusion only; this is not an optimizer ranking across solvers.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'complete':report['complete'],'completed':report['completed_cells'],'failures':[(r['id'],r['failure']) for r in failures],'selection':selection,'joint_vs_velocity':paired,'head_scaling':scale_rows},indent=2))
