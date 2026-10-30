"""Scalar-only validation search summary and immutable seed-0 selections."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).parent;out=p/'screen-report';out.mkdir(exist_ok=True)
protocol=json.loads((p/'PROTOCOL.json').read_text());jobs=json.loads((p/'jobs.json').read_text());rows=[];failed=[];active=[];raw={}
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
for job in jobs:
 f=Path(job['directory'])/'outcome.json'
 if not f.exists():active.append(job['id']);continue
 r=json.loads(f.read_text());raw[job['id']]=r
 if not r.get('completed'):
  if r.get('failure'):failed.append({'id':job['id'],'failure':r['failure'],'completed_updates':len(r['history']),'directory':job['directory'],'outcome_sha256':sha(f)})
  else:active.append(job['id'])
  continue
 assert r['completed_updates']==256 and not r['heldout_accessed']
 row={'id':job['id'],'recipe':r['recipe'],'selected_update':r['selected_update'],'validation_loss':r['validation_loss'],'selected_validation_loss':r['selected_validation_loss'],'fit_elapsed_s':r['fit_elapsed_s'],'training_seconds':r['training_seconds'],'validation_seconds':r['validation_seconds'],'selected_parameter_sha256':r['selected_parameter_sha256'],'directory':job['directory'],'outcome_sha256':sha(f)}
 ck=next(x for x in r['checkpoints'] if x['update']==r['selected_update'])
 row['selected_per_validation_loss']=ck['per_validation_loss'];row['selected_validation_physics']=ck['physics'];rows.append(row)
complete=len(jobs)==len(protocol['recipes']) and not active and len(rows)+len(failed)==len(jobs)
groups={'velocity':lambda r:not r['recipe']['joint'],'joint':lambda r:r['recipe']['joint'],'joint_zero_mean':lambda r:r['recipe']['joint'] and r['recipe']['zero_mean_temperature']}
chosen={};models=[]
for name,pred in groups.items():
 candidates=[r for r in rows if pred(r)]
 if candidates:chosen[name]=min(candidates,key=lambda r:(r['validation_loss'],r['id']))
for name,row in chosen.items():
 if any(m['recipe']['id']==row['id'] for m in models):
  next(m for m in models if m['recipe']['id']==row['id'])['roles'].append(name);continue
 d=Path(row['directory']);recipe_file=p/'recipes'/(row['id']+'.json')
 models.append({'roles':[name],'recipe':row['recipe'],'recipe_file':str(recipe_file),'recipe_sha256':sha(recipe_file),'selected_params_file':str(d/'selected_params.npz'),'selected_params_file_sha256':sha(d/'selected_params.npz'),'selected_params_leaf_sha256':row['selected_parameter_sha256'],'selected_update':row['selected_update'],'validation_loss':row['validation_loss'],'selected_validation_loss':row['selected_validation_loss'],'outcome_file':str(d/'outcome.json'),'outcome_sha256':row['outcome_sha256'],'model_seed':0})
selection={'complete':complete,'phase':'screen-seed0-only-root-review-before-confirmation','source_head':protocol['source_head'],'protocol_sha256':sha(p/'PROTOCOL.json'),'validation_freeze_sha256':sha(p/'validation-freeze.json'),'validation_seeds':[1,5,6],'heldout_accessed':False,'tie_break':'validationtotal thenrecipeid lexicographic; earliestupdate checkpoint tie','models':models,'group_recipe_ids':{k:v['id'] for k,v in chosen.items()},'completed_recipes':len(rows),'failed_recipes':len(failed),'active':active}
if complete:
 assert len(chosen)==3
 (p/'screen-selection.json').write_text(json.dumps(selection,indent=2))
summary={'complete':complete,'selected':selection,'rows':rows,'failures':failed,'active':active}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
with (out/'grid.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['id','joint','center_temperature','zero_mean_temperature','head_scale','lr','selected_update','validation_total','validation_velocity','validation_temperature','fit_seconds'])
 for r in rows:
  q=r['recipe'];writer.writerow([r['id'],q['joint'],q['center_temperature'],q['zero_mean_temperature'],q['temperature_head_scale'],q['lr'],r['selected_update'],r['validation_loss'],r['selected_validation_loss']['velocity'],r['selected_validation_loss']['temperature'],r['fit_elapsed_s']])
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':160})
fig,axs=plt.subplots(2,2,figsize=(11,7));scales=[.1,.2139727633353161,.5,1.];lrs=[.0003,.001,.003]
values=[r['validation_loss'] for r in rows if r['recipe']['joint']];vmin=min(values);vmax=max(values)
from matplotlib.colors import LogNorm
for ax,(center,zero) in zip(axs.flat,[(False,False),(False,True),(True,False),(True,True)]):
 a=np.full((3,4),np.nan)
 for i,lr in enumerate(lrs):
  for j,scale in enumerate(scales):
   rr=[r for r in rows if r['recipe']['joint'] and r['recipe']['center_temperature']==center and r['recipe']['zero_mean_temperature']==zero and r['recipe']['lr']==lr and r['recipe']['temperature_head_scale']==scale]
   if rr:a[i,j]=rr[0]['validation_loss']
 im=ax.imshow(a,norm=LogNorm(vmin=vmin,vmax=vmax),cmap='viridis_r',aspect='auto')
 for i in range(3):
  for j in range(4):ax.text(j,i,'rejected' if not np.isfinite(a[i,j]) else f'{a[i,j]:.2g}',ha='center',va='center',fontsize=8,color='black' if not np.isfinite(a[i,j]) else 'white')
 ax.set_xticks(range(4),['.1','.214','.5','1']);ax.set_yticks(range(3),['.0003','.001','.003']);ax.set_xlabel('Temperature readout scale');ax.set_ylabel('Adam learning rate');ax.set_title(f'Input {"centered" if center else "raw"} · residual {"mean-zero" if zero else "unconstrained"}')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Mean3-IC normalized validation loss',shrink=.8);fig.suptitle('Joint correction: full mixed-window recipe · seed0 · 256 updates');fig.savefig(out/'joint_grid.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4))
colors={'velocity':'#225ea8','joint':'#d95f0e','joint_zero_mean':'#238b45'}
for role,row in chosen.items():
 r=raw[row['id']];best=None;curves=[]
 for ck in r['checkpoints']:
  if ck['validation_cfl_admissible'] and (best is None or ck['aggregate_loss']['total']<best['aggregate_loss']['total']):best=ck
  curves.append(best['aggregate_loss'])
 for ax,key in zip(axs,['total','velocity','temperature']):
  ax.step([c['update'] for c in r['checkpoints']],[c[key] for c in curves],where='post',marker='o',color=colors[role],label=f"{role}: {row['id']}")
  ax.set_title(key);ax.set_xlabel('Optimizer updates');ax.set_ylabel('Best total-selected model validation component')
axs[0].legend(fontsize=6);fig.suptitle('Best recipes: same three validation ICs; no held-out evaluation');fig.tight_layout();fig.savefig(out/'best_validation.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for role,row in chosen.items():
 pp=row['selected_validation_physics'];xx=np.arange(3)+{'velocity':-.15,'joint':0,'joint_zero_mean':.15}[role]
 axs[0].scatter(xx,[100*pp[str(s)]['final_relative_heat_drift'] for s in [1,5,6]],color=colors[role],label=role)
 axs[1].scatter(xx,[pp[str(s)]['temperature_min'] for s in [1,5,6]],color=colors[role],label=role)
for ax in axs:ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['IC1','IC5','IC6']);ax.legend(fontsize=8)
axs[0].set_ylabel('Final mean-heat drift (%)');axs[1].set_ylabel('Minimum temperature over full validation rollout');fig.suptitle('Validation physics: conservation and temperature bounds are distinct');fig.tight_layout();fig.savefig(out/'validation_physics.png',bbox_inches='tight');plt.close(fig)
lines=['# Joint corrector improvement screen','','JAX-CFD; full mixed four-case window stream; 256Adamupdates, H8, width8, modelseed0. Checkpoints and recipes selected by mean normalized loss across validation IC1,5,6 only. No held-out fields were accessed.','',f'Completed {len(rows)}/{len(jobs)}; rejected {len(failed)}.','', '| Group | Recipe | Update | Validation total | Velocity component | Temperature component |','|---|---|---:|---:|---:|---:|']
for name,r in chosen.items():lines.append(f"| {name} | {r['id']} | {r['selected_update']} | {r['validation_loss']:.7g} | {r['selected_validation_loss']['velocity']:.7g} | {r['selected_validation_loss']['temperature']:.7g} |")
lines+=['','Velocity6 andjoint48 configurations give unequal hyperparameter-search compute; per-model training budgets are equal. Mean-zero thermal redistribution preserves heat to floating-point roundoff, not a native flux formulation or positivity constraint. OriginalFDfailures and all validationphysics are retained. The same two-point, threeepsilon research gate applies to every model and leaves public admission unchanged.','', 'This freezes seed0 recipes only; root reviews before confirming the chosen2or3 recipes on modelseeds1,2, then evaluating fresh heldoutIC101–105. Repeatedly inspected IC100 is not used.','', 'Timing uses nativeRTX, single fullycompiledtrueAD+Adamwindowupdates. Compile/validation are charged tofit; shared preparation excluded. No CI forecast.','']
for name in ['joint_grid','best_validation','validation_physics']:lines.append(f'![{name}]({name}.png)\n')
(out/'README.md').write_text('\n'.join(lines))
print(json.dumps({'complete':complete,'completed':len(rows),'failed':len(failed),'selected':{k:{'id':v['id'],'validation_loss':v['validation_loss']} for k,v in chosen.items()}},indent=2))
