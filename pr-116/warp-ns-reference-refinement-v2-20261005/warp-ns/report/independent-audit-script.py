import json,hashlib,sys
from experiments.solver_in_loop.final_report import paired_ratio
from pathlib import Path
import numpy as np
c=Path(sys.argv[1])
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(c/'plan.json');r=load(c/'report/report.json');i=load(c/'report-input.json');state=load(c/'controller-state.json');b=p['base_payload']
assert sha(c/'source.tar')==p['source_sha256']==r['source_sha256']
d=load(c/'results/assembled/dataset.json');assert sha(c/'results/assembled/dataset.npz')==d['dataset_sha256'];assert d['train_seeds']==list(range(8)) and d['eval_seeds']==list(range(20000,20032))
expected={k:b[k] for k in ('solver','source_sha256','image','image_sha256')};expected['physics']=b['run']['physics'];expected['domain_extent']=b.get('domain_extent',2*np.pi)
models={};matrix={a:np.empty((8,32)) for a in ('full','stopped','supervised')};seen=set()
for row in i['training']:
 o=load(Path(row['path'])/'outcome.json');a=row['arm'];s=row['model_seed'];recipe=p['confirmation_recipes'][a]
 assert o['identity']==expected and o['completed'] and o['admitted'] and o['evaluation_seeds']==[]
 assert o['training_dataset_sha256']==d['dataset_sha256']
 assert o['optimizer_updates']==recipe['updates'] and o['training']['lr']==recipe['lr'] and o['training']['unroll']==recipe['unroll']
 assert sha(Path(row['path'])/'model.eqx')==o['model_sha256'];assert(a,s)not in models;models[a,s]=o['model_sha256']
assert len(models)==24
for row in i['test']:
 o=load(Path(row['path'])/'outcome.json');a=row['arm'];s=row['model_seed'];assert o['identity']==expected and o['completed'] and o['admitted']
 assert o['evaluation_dataset_sha256']==d['dataset_sha256'] and o['model_sha256']==models[a,s] and o['evaluation_seeds']==row['ic_seeds']
 with np.load(Path(row['path'])/'fields.npz') as data:
  errors=data['error_corrected'];assert errors.shape==(4,49) and np.isfinite(errors).all()
  for j,ic in enumerate(row['ic_seeds']):
   key=(a,s,ic);assert key not in seen;seen.add(key);matrix[a][s-8,ic-20000]=errors[j,1:].mean()
assert len(seen)==768 and len(i['test'])==192
for a,v in matrix.items():assert abs(v.mean()-r['mean_errors'][a])<1e-12
refs=state.get('active_reference_cells',[f'reference-ic{x}' for x in list(range(8))+list(range(20000,20032))]);assert len(refs)==40
for cell in refs:
 o=load(c/'results'/cell/'outcome.json');assert o['completed'] and o['admitted'];md=load(c/'results'/cell/'dataset.json');assert md['identity']==expected and md['admitted']
result={'passed':True,'source_sha256':p['source_sha256'],'image_sha256':b['image_sha256'],'shared_dataset_sha256':d['dataset_sha256'],'training_checkpoints':24,'evaluation_batches':192,'model_ic_arm_pairs':768,'admitted_references':40,'mean_errors':{a:float(v.mean())for a,v in matrix.items()},'comparisons':{a:{'ratio':float(matrix['full'].mean()/matrix[a].mean()),'model_wins':int(np.sum(matrix['full'].mean(1)<matrix[a].mean(1))),'ic_wins':int(np.sum(matrix['full'].mean(0)<matrix[a].mean(0)))}for a in ('stopped','supervised')},'report_sha256':sha(c/'report/report.json')}

for arm in ('stopped','supervised'):
 comparison=paired_ratio(matrix['full'],matrix[arm]);assert np.allclose(comparison['ci95'],r['comparisons'][arm]['ci95'],rtol=0,atol=1e-12)
 result['comparisons'][arm]['ci95']=comparison['ci95']
result['audit_script_sha256']=sha(Path(__file__))
(c/'independent-audit-script.py').write_text(Path(__file__).read_text())
(c/'independent-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
