"""Prepare locked winner confirmations; submit only with explicit --submit."""
from pathlib import Path
import json,hashlib,sys,copy
from runner import Runner,JobRegistry,JobSpec
p=Path(__file__).parent
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
selection=json.loads((p/'screen-selection.json').read_text());assert selection['complete'] and not selection['heldout_accessed']
models=[]
for m in selection['models']:
 for seed in [1,2]:
  recipe=copy.deepcopy(m['recipe']);recipe['model_seed']=seed;recipe['id']=recipe['id'].removesuffix('-s0')+f'-s{seed}'
  path=p/'recipes'/(recipe['id']+'.json')
  if path.exists():assert json.loads(path.read_text())==recipe
  else:path.write_text(json.dumps(recipe,indent=2))
  models.append({'roles':m['roles'],'recipe':recipe,'recipe_file':str(path),'recipe_sha256':sha(path),'directory':str(p/'confirmation'/recipe['id'])})
assert len(models)<=6
plan={'source_head':selection['source_head'],'screen_selection_sha256':sha(p/'screen-selection.json'),'validation_freeze_sha256':selection['validation_freeze_sha256'],'model_seeds':[1,2],'models':models,'scope':'Same validation-chosen recipes, unchanged hyperparameters and trainingstream. Validationselectscheckpoint independentlyperseed usingIC1,5,6. No additional recipechoices, no heldoutaccess, no LRrescue.'}
planpath=p/'confirmation-plan.json'
if planpath.exists():assert json.loads(planpath.read_text())==plan
else:planpath.write_text(json.dumps(plan,indent=2))
if '--submit' not in sys.argv:
 print(json.dumps(plan,indent=2));sys.exit(0)
assert sha(p/'screen-selection.json')==plan['screen_selection_sha256']
jobs=json.loads((p/'confirmation-jobs.json').read_text()) if (p/'confirmation-jobs.json').exists() else []
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
for m in models:
 recipe=m['recipe'];ident=recipe['id']
 if any(j['id']==ident for j in jobs):continue
 out=Path(m['directory']);out.mkdir(parents=True,exist_ok=True)
 spec=JobSpec(name=f'buoyancy-joint-confirm-{ident}',template='gpu',gpus='gpu:5090:1',cpus=4,mem='32G',time_limit='00:15:00',cmd=['bash',str(p/'node.sh'),str(p),'confirmation',ident],out_path=out/'job-outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(p)},extra_sbatch=[f'--output={out}/slurm-%j.out'])
 r=runner.submit(spec,cluster='kander')
 row={**m,'id':ident,'slurm_id':r.slurm_id};jobs.append(row);(out/'job.json').write_text(json.dumps(row,indent=2));(p/'confirmation-jobs.json').write_text(json.dumps(jobs,indent=2));print(row,flush=True)
