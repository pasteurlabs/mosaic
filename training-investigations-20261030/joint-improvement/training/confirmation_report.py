"""Validation-only three-seed confirmation report; no test/reference access."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).parent;out=p/'confirmation-report';out.mkdir(exist_ok=True)
selection=json.loads((p/'confirmation-summary.json').read_text());assert selection['complete'] and not selection['heldout_accessed'];models=selection['models'];rows=[];raw={};rawout=out/'raw';rawout.mkdir(exist_ok=True)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
for m in models:
 f=Path(m['outcome_file']);assert sha(f)==m['outcome_sha256'];r=json.loads(f.read_text());assert r['completed'] and not r['heldout_accessed'] and r['completed_updates']==256
 role=m['roles'][0];seed=m['model_seed'];ck=next(c for c in r['checkpoints'] if c['update']==m['selected_update']);raw[(role,seed)]=r
 row={'role':role,'model_seed':seed,'recipe':m['recipe'],'selected_update':m['selected_update'],'validation_loss':m['validation_loss'],'selected_validation_loss':m['selected_validation_loss'],'per_validation_loss':ck['per_validation_loss'],'selected_physics':ck['physics'],'training_seconds':r['training_seconds'],'validation_seconds':r['validation_seconds'],'fit_elapsed_s':r['fit_elapsed_s'],'bookkeeping_seconds':max(0.,r['fit_elapsed_s']-r['training_seconds']-r['validation_seconds']),'first_update_seconds':r['history'][0]['update_seconds'],'outcome_sha256':sha(f),'selected_parameter_sha256':r['selected_parameter_sha256'],'heldout_accessed':False}
 rows.append(row);shutil.copy2(f,rawout/f'{role}-seed-{seed}.json')
roles=['velocity','joint','joint_zero_mean'];labels={'velocity':'Velocity only','joint':'Joint','joint_zero_mean':'Joint: mean-zero heat'};colors={'velocity':'#225ea8','joint':'#d95f0e','joint_zero_mean':'#238b45'}
ratios=[]
for seed in [0,1,2]:
 baseline=next(r for r in rows if r['role']=='velocity' and r['model_seed']==seed)
 for role in roles[1:]:
  r=next(r for r in rows if r['role']==role and r['model_seed']==seed);ratios.append({'role':role,'model_seed':seed,'validation_ratio_vs_velocity':r['validation_loss']/baseline['validation_loss']})
summary={'complete':True,'source_head':selection['source_head'],'confirmation_summary_sha256':sha(p/'confirmation-summary.json'),'screen_selection_sha256':selection['screen_selection_sha256'],'models':rows,'paired_validation_ratios':ratios,'heldout_accessed':False,'scope':'Validation-only fixed-recipe confirmation. No heldout oradditionalhyperparameterselection. First update includescompileandtraining; nativeRTXnotCIforecast.'}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':160})
def bestcurves(r):
 best=None;xx=[];curves=[]
 for ck in r['checkpoints']:
  if ck['validation_cfl_admissible'] and (best is None or ck['aggregate_loss']['total']<best['aggregate_loss']['total']):best=ck
  xx.append(ck['update']);curves.append(best['aggregate_loss'])
 return xx,curves
fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True)
for ax,seed in zip(axs,[0,1,2]):
 for role in roles:
  xx,yy=bestcurves(raw[(role,seed)]);ax.step(xx,[y['total'] for y in yy],where='post',marker='o',color=colors[role],label=labels[role])
 ax.set_title(f'Network seed {seed}');ax.set_xlabel('Optimizer updates');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axs[0].set_ylabel('Best observed mean validation loss (IC1,5,6)');axs[0].legend(fontsize=8)
fig.suptitle('Fixed recipes across three network seeds · all256 updates run');fig.tight_layout();fig.savefig(out/'validation_confirmations.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(3,2,figsize=(12,10),sharex=True)
for ri,seed in enumerate([0,1,2]):
 for ci,key in enumerate(['velocity','temperature']):
  ax=axs[ri,ci]
  for role in roles:
   xx,yy=bestcurves(raw[(role,seed)]);ax.step(xx,[y[key] for y in yy],where='post',marker='o',color=colors[role],label=labels[role])
  ax.set_title(f'Seed {seed} · {key}');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Normalized validation loss contribution');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axs[0,0].legend(fontsize=8);fig.suptitle('Both components belong to the same TOTAL-validation-selected checkpoint');fig.tight_layout();fig.savefig(out/'component_validation.png',bbox_inches='tight');plt.close(fig)
rows=sorted(rows,key=lambda r:(roles.index(r['role']),r['model_seed']));xx=np.arange(len(rows));xt=[labels[r['role']]+f"\nseed{r['model_seed']}" for r in rows]
fig,axs=plt.subplots(1,2,figsize=(15,5))
bottom=np.zeros(len(rows))
for key,label,color in [('training_seconds','Training incl. first compile','#3182bd'),('validation_seconds','Validation/physical diagnostics','#74c476'),('bookkeeping_seconds','Bookkeeping/checkpoint I/O','#969696')]:
 values=np.asarray([r[key] for r in rows]);axs[0].bar(xx,values,bottom=bottom,label=label,color=color);bottom+=values
axs[0].set_ylabel('Fit time (seconds)');axs[0].set_title('Full256 updates + all validation');axs[0].legend(fontsize=8)
axs[1].bar(xx,[r['first_update_seconds'] for r in rows],color=[colors[r['role']] for r in rows]);axs[1].set_title('First update: includes compilation and first training step');axs[1].set_ylabel('Seconds (already included at left)')
for ax in axs:ax.set_xticks(xx,xt,rotation=55,ha='right',fontsize=7)
fig.suptitle('Native RTX timing; common preparation excluded; not a CI forecast');fig.tight_layout();fig.savefig(out/'fit_timing.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(15,5))
heat=[max(100*abs(v['final_relative_heat_drift']) for v in r['selected_physics'].values()) for r in rows]
tmin=[min(v['temperature_min'] for v in r['selected_physics'].values()) for r in rows]
axs[0].scatter(xx,heat,c=[colors[r['role']] for r in rows]);axs[0].set_yscale('log');axs[0].set_ylabel('Max absolute final heat drift across3 validation ICs (%)');axs[0].set_title('Mean-zero constraint controls total heat')
axs[1].scatter(xx,tmin,c=[colors[r['role']] for r in rows]);axs[1].axhline(0,color='black',lw=.8);axs[1].set_ylabel('Minimum temperature across3 full rollouts');axs[1].set_title('Conservation does not guarantee positivity')
for ax in axs:ax.set_xticks(xx,xt,rotation=55,ha='right',fontsize=7)
fig.tight_layout();fig.savefig(out/'validation_physics_confirmations.png',bbox_inches='tight');plt.close(fig)
lines=['# Joint correction: fixed-recipe confirmation','','Three recipes were frozen using only the54-candidate seed0 screen and mean loss on validation IC1,5,6. The exact same hyperparameters were then trained with network seeds1 and2. Everymodel completed256mixed-window H8Adamupdates. No held-out fields or scores were accessed, and no recipe was replaced after confirmation.','',
'| Family | Network seed | Selected update | Validation total | Velocity contribution | Temperature contribution | Fit seconds | First update seconds |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 v=r['selected_validation_loss'];lines.append(f"| {labels[r['role']]} | {r['model_seed']} | {r['selected_update']} | {v['total']:.7g} | {v['velocity']:.7g} | {v['temperature']:.7g} | {r['fit_elapsed_s']:.2f} | {r['first_update_seconds']:.2f} |")
lines+=['',
'All models use the same training trajectories, balanced window stream and frozen validation references. Checkpoint selection remains independent per network seed using validation only. The best joint recipe has lower validation loss in all three seeds; the mean-zero recipe is better in two and slightly worse in one. These observations do not establish held-out performance. Fresh-test scoring belongs to a separate locked evaluator.','',
'The training clock includes first-update compilation and all256 optimizer updates. Validation includes full rollouts and physical diagnostics; checkpoint/file bookkeeping is charged to fit time. Common source/reference/native/gate preparation and environment setup are excluded. These are nativeRTX measurements, not container/T4 CI forecasts.','',
'The thermal-mean constraint preserves heat to floating-point roundoff but does not enforce positivity. Original and supplementary finite-difference profiles remain in the raw outcomes. All derivative/admission criteria and benchmark defaults are unchanged. The54-cell search and its asymmetric48joint/6velocity hyperparameter budgets remain explicit.','']
for name in ['validation_confirmations','component_validation','fit_timing','validation_physics_confirmations']:lines.append(f'![{name}]({name}.png)\n')
(out/'README.md').write_text('\n'.join(lines))
manifest={'complete':True,'confirmation_summary_sha256':sha(p/'confirmation-summary.json'),'script_sha256':sha(__file__),'models_outcome_sha256':{r['role']+'-seed-'+str(r['model_seed']):r['outcome_sha256'] for r in rows},'files':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'complete':True,'paired_validation_ratios':ratios},indent=2))
