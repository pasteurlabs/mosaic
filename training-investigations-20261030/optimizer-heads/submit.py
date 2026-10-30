from pathlib import Path
import json,sys
from runner import Runner,JobRegistry,JobSpec
p=Path(__file__).parent
names=sys.argv[1:]
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
jobs=json.loads((p/'jobs.json').read_text()) if (p/'jobs.json').exists() else []
for name in names:
 assert (p/'recipes'/(name+'.json')).exists()
 assert name not in [j['id'] for j in jobs]
 out=p/'results'/name;out.mkdir(parents=True,exist_ok=False)
 spec=JobSpec(name='buoyancy-opt-'+name,template='gpu',gpus='gpu:5090:1',cpus=4,mem='32G',time_limit='00:15:00',cmd=['bash',str(p/'node-v2.sh'),str(p),name],out_path=out/'job-outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(p)},extra_sbatch=[f'--output={out}/slurm-%j.out'])
 result=runner.submit(spec,cluster='kander')
 row={'id':name,'slurm_id':result.slurm_id,'directory':str(out),'driver':'driver-v2.py','driver_sha256':__import__('hashlib').sha256((p/'driver-v2.py').read_bytes()).hexdigest()}
 jobs.append(row);(out/'job.json').write_text(json.dumps(row,indent=2));(p/'jobs.json').write_text(json.dumps(jobs,indent=2));print(row,flush=True)
