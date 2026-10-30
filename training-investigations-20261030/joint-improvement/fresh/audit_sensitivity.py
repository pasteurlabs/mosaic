"""Reference sensitivity only: fixed predictions against the stored finer-time audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent;out=root/'report'
evaluation=root/'evaluation/outcome.json';x=json.loads(evaluation.read_text());assert x['complete']
rr=json.loads((root/'references/outcome.json').read_text());refs={r['seed']:r for r in rr['rows'] if r['role']=='sealed_fresh_test'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
D=np.asarray(x['scales'],np.float64)
audits={}
for seed,item in refs.items():
 assert sha(item['file'])==item['file_sha256']
 z=np.load(item['file']);audits[seed]=z['audit']
 assert digest(audits[seed])==item['audit_sha256']
def score(pred,target):
 e=((pred[1:].astype(np.float64)-target[1:].astype(np.float64))/D)**2
 vel=.5*float(np.mean(e[...,:2]));temp=.5*float(np.mean(e[...,2]))
 return {'total':vel+temp,'velocity_contribution':vel,'temperature_contribution':temp}
rows=[]
for model in x['models']:
 for case in model['cases']:
  assert sha(case['fields_path'])==case['fields_sha256']
  z=np.load(case['fields_path']);pred=z['corrected'];standard=score(pred,z['reference']);audit_score=score(pred,audits[case['ic_seed']])
  for field in standard:assert np.isclose(standard[field],case['score'][field],rtol=1e-12,atol=1e-15)
  rows.append({'group':model['group'],'model_seed':model['model_seed'],'ic_seed':case['ic_seed'],'standard':standard,'finer_time_audit':audit_score,'fields_path':case['fields_path'],'fields_sha256':case['fields_sha256']})
groups=sorted(set(r['group'] for r in rows));by={(r['group'],r['model_seed'],r['ic_seed']):r for r in rows};pairs=[]
for group in groups:
 if group=='velocity':continue
 for seed in [0,1,2]:
  r={'group':group,'model_seed':seed,'cases':[]}
  for target in ['standard','finer_time_audit']:
   vals=[by[group,seed,ic][target]['total'] for ic in [101,102,103,104,105]]
   base=[by['velocity',seed,ic][target]['total'] for ic in [101,102,103,104,105]]
   r[target+'_ratio_mean_loss']=float(np.mean(vals)/np.mean(base))
  for ic in [101,102,103,104,105]:
   a=by[group,seed,ic];b=by['velocity',seed,ic]
   ratios={target:a[target]['total']/b[target]['total'] for target in ['standard','finer_time_audit']}
   r['cases'].append({'ic_seed':ic,'ratios':ratios,'winner_flips':(ratios['standard']<1)!=(ratios['finer_time_audit']<1)})
  pairs.append(r)
summary=[]
for group in groups:
 if group=='velocity':continue
 rowsg=[p for p in pairs if p['group']==group]
 summary.append({'group':group,'standard_mean_paired_ratio':float(np.mean([r['standard_ratio_mean_loss'] for r in rowsg])),'audit_mean_paired_ratio':float(np.mean([r['finer_time_audit_ratio_mean_loss'] for r in rowsg])),'standard_seed_ratios':[r['standard_ratio_mean_loss'] for r in rowsg],'audit_seed_ratios':[r['finer_time_audit_ratio_mean_loss'] for r in rowsg],'seed_mean_winner_flips':sum((r['standard_ratio_mean_loss']<1)!=(r['finer_time_audit_ratio_mean_loss']<1) for r in rowsg),'case_winner_flips':sum(c['winner_flips'] for r in rowsg for c in r['cases']),'standard_case_wins':sum(c['ratios']['standard']<1 for r in rowsg for c in r['cases']),'audit_case_wins':sum(c['ratios']['finer_time_audit']<1 for r in rowsg for c in r['cases'])})
record={'complete':True,'scope':'Exact same frozen45predictions, no model/nativecalls orselection. Primary targetN128 dt.0025 vsstoredtemporal auditN128 dt.00125, bothrestrictedN64. Does not auditspatialrefinement.','selection_sha256':x['selection_sha256'],'evaluation_sha256':sha(evaluation),'reference_outcome_sha256':sha(root/'references/outcome.json'),'primary_metrics_recomputed_equal':True,'summary':summary,'paired':pairs,'rows':rows}
(out/'reference_sensitivity.json').write_text(json.dumps(record,indent=2,allow_nan=False))
plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.grid':True,'grid.alpha':.18})
fig,axs=plt.subplots(1,len(summary),figsize=(6*len(summary),4),squeeze=False)
for ax,g in zip(axs[0],summary):
 ax.plot([0,1,2],g['standard_seed_ratios'],'o-',color='#2166ac',label='Reference dt0.0025')
 ax.plot([0,1,2],g['audit_seed_ratios'],'^--',color='#238b45',label='Stored audit dt0.00125')
 ax.axhline(1,color='gray',ls='--');ax.set_xticks([0,1,2]);ax.set_xlabel('Network initialization seed');ax.set_ylabel('Five-IC mean loss / matched velocity-only');ax.set_title('Joint' if g['group']=='joint' else 'Joint, heat constrained');ax.legend(fontsize=8)
fig.suptitle('Reference sensitivity of the SAME frozen predictions; no retraining',y=1.02);fig.tight_layout();fig.savefig(out/'reference_sensitivity.png',bbox_inches='tight');plt.close(fig)
lines=['# Finer-time reference sensitivity','','The same45frozen predictions were scored against the preexisting N128 dt0.00125 audit; no model or native solver was called, and no choice changed. Both targets are restricted to N64. This checks temporal-reference sensitivity, not spatial-reference convergence.','','| Group | Standard mean paired ratio | Finer-time audit ratio | Seed-mean winner flips | Case winner flips |','|---|---:|---:|---:|---:|']
for s in summary:lines.append(f"| {s['group']} | {s['standard_mean_paired_ratio']:.6f} | {s['audit_mean_paired_ratio']:.6f} | {s['seed_mean_winner_flips']} | {s['case_winner_flips']} |")
lines+=['','Ratios average three paired seed ratios; each seed ratio uses mean loss on the same five ICs. All45primary metrics recomputed identically to the original report within1e-12 relative.','','![Temporal-reference sensitivity](reference_sensitivity.png)']
(out/'reference_sensitivity.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
