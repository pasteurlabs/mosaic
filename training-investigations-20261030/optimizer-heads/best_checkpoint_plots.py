"""Two legible best-observed-checkpoint plots; scalar metadata only."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
p=Path(__file__).parent;out=p/'report'
summary=json.loads((out/'summary.json').read_text())
before=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()
optimizers=['adam','adamw','muon','sgd','lbfgs']
colors=dict(zip(optimizers,['#1764ab','#c34800','#2d8a42','#9343a5','#343434']))
styles={'adam':'-','adamw':'--','muon':'-','sgd':':','lbfgs':'-.'}
plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.grid':True,'grid.alpha':.16})
curves=[]
for opt in optimizers:
 lr=summary['shared_lr_selection'][opt]['lr']
 for joint in [False,True]:
  row=next(r for r in summary['selected_rows'] if r['optimizer']==opt and r['joint']==joint)
  raw=out/'raw'/(row['id']+'.json');r=json.loads(raw.read_text())
  checks=r['checkpoints'];best=None;points=[]
  for c in checks:
   if best is None or c['loss']['total']<best['loss']['total']:best=c
   points.append({'measured_at_oracle_calls':c['oracle_calls'],'measured_at_fit_elapsed_s':c['fit_elapsed_s'],'selected_oracle_calls':best['oracle_calls'],'selected_loss':best['loss'],'selected_parameter_sha256':best['parameter_sha256']})
  curves.append({'optimizer':opt,'joint':joint,'lr':lr,'points':points,'raw_outcome_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
for axis in ['time','work']:
 fig,axs=plt.subplots(2,3,figsize=(14.5,8),sharex='col')
 for i,joint in enumerate([False,True]):
  rr=[c for c in curves if c['joint']==joint]
  for j,component in enumerate(['total','velocity','temperature']):
   ax=axs[i,j]
   initial=rr[0]['points'][0]['selected_loss'][component]
   ax.axhline(initial,color='#808080',ls=':',lw=1.2,alpha=.7,zorder=1)
   for c in rr:
    points=c['points'];x=[r['measured_at_fit_elapsed_s'] if axis=='time' else r['measured_at_oracle_calls']*4 for r in points]
    y=[r['selected_loss'][component] for r in points]
    ax.step(x,y,where='post',color=colors[c['optimizer']],ls=styles[c['optimizer']],lw=1.7,zorder=2)
    ax.scatter(x,y,s=13,facecolor='white',edgecolor=colors[c['optimizer']],linewidth=.7,zorder=3)
   values=[r['selected_loss'][component] for c in rr for r in c['points']]
   low=min(values);high=max(values);pad=max((high-low)*.12,high*.025)
   ax.set_ylim(max(0,low-pad),high+pad)
   ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3),useMathText=True)
   ax.set_title(('Velocity only' if not joint else 'Velocity + temperature')+' · '+component)
   ax.set_ylabel('Normalized validation loss contribution')
   ax.set_xlabel('Fit seconds' if axis=='time' else 'Optimizer window evaluations (all line-search trials included)')
   if i==1 and component=='total':
    ax.text(.985,.96,'SGD: initial identity remains selected',ha='right',va='top',transform=ax.transAxes,fontsize=8,color=colors['sgd'],bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
 handles=[Line2D([0],[0],color=colors[o],ls=styles[o],lw=2,label=o.upper()+f" · shared LR {summary['shared_lr_selection'][o]['lr']:g}") for o in optimizers]
 handles.append(Line2D([0],[0],color='#808080',ls=':',label='Uncorrected / initial identity'))
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=9)
 note='The lowest measured TOTAL validation selects one checkpoint; all three panels show that same model.'
 timing='Fit clock includes first optimizer compilation, validation and component-gradient diagnostics; native RTX, not CI timing.'
 fig.suptitle('Best observed checkpoint · JAX-CFD · seed 0 · fixed four-window H8 objective',fontsize=13,y=.99)
 fig.text(.5,.945,note,ha='center',fontsize=10)
 fig.text(.5,.005,timing if axis=='time' else 'Steps carry the latest measured selected model between evaluations; markers denote actual validation measurements.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.12,1,.925));fig.savefig(out/f'best_checkpoint_{axis}.png',bbox_inches='tight');plt.close(fig)
assert hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()==before,'Numerical summary modified'
record={'completed':True,'source_summary_sha256':before,'selection':'Cumulative argmin total validation only; chosen-model velocity and temperature contributions retained. No independent per-component minimum. Strict less-than preserves initial/earlier checkpoint ties.','sampling':'Markers at actual completed validation measurements; step-after carries latest measured chosen model, not interpolated/re-simulated checkpoints.','time':'Original fitelapsed includes compile,validation,diagnostics; no fresh timing measurement.','scope':'Single modelseed0, fixed4H8trainingwindows; shared LR chosen by geometricmean totalvalidation across bothheadmodes. Pure scalar plotting; original rawplots/numericalsummary unchanged.','curves':curves,'files':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['best_checkpoint_time.png','best_checkpoint_work.png']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'best-checkpoint-plot-manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in record.items() if k!='curves'},indent=2))
