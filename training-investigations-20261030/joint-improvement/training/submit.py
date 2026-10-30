from pathlib import Path
import json,sys
from runner import Runner,JobRegistry,JobSpec
p=Path(__file__).parent
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
jobs=json.loads((p/'jobs.json').read_text()) if (p/'jobs.json').exists() else []
for ident in sys.argv[1:]:
 if any(j['id']==ident for j in jobs):print('Already submitted',ident,flush=True);continue
 recipe=json.loads((p/'recipes'/(ident+'.json')).read_text());out=p/'train'/ident;out.mkdir(parents=True,exist_ok=True)
 spec=JobSpec(name=f'buoyancy-joint-{ident}',template='gpu',gpus='gpu:5090:1',cpus=4,mem='32G',time_limit='00:15:00',cmd=['bash',str(p/'node.sh'),str(p),'train',ident],out_path=out/'job-outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(p)},extra_sbatch=[f'--output={out}/slurm-%j.out'])
 result=runner.submit(spec,cluster='kander')
 row={'id':ident,'slurm_id':result.slurm_id,'directory':str(out),'recipe':recipe}
 jobs.append(row);(out/'job.json').write_text(json.dumps(row,indent=2));(p/'jobs.json').write_text(json.dumps(jobs,indent=2));print(row,flush=True)
