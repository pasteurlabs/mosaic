from pathlib import Path
import json,sys
from runner import Runner,JobRegistry,JobSpec
p=Path(sys.argv[1])
solvers=sys.argv[2:] or ['ins-jl','oceananigans','waterlily']
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
jobs=[]
for solver in solvers:
 out=p/solver;out.mkdir(exist_ok=True)
 assert not (out/'job.json').exists(),'Refusing accidental duplicate submission'
 spec=JobSpec(name='buoyancy-main-joint-'+solver,template='cpu',cpus=8,mem='32G',time_limit='02:00:00',cmd=['bash',str(p/'cpu-node.sh'),str(p),solver],out_path=out/'job-outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(p)},extra_sbatch=[f'--output={out}/slurm-%j.out','--exclude=dgx[01-06]'])
 result=runner.submit(spec,cluster='kander')
 row={'solver':solver,'slurm_id':result.slurm_id,'output':str(out),'node_policy':'CPU only; excludes all DGX nodes dgx01..06','runtime_manifest':str(p/'julia-runtime-manifest.json')}
 (out/'job.json').write_text(json.dumps(row,indent=2));jobs.append(row)
 (p/'julia-jobs.json').write_text(json.dumps(jobs,indent=2))
 print(json.dumps(row),flush=True)
