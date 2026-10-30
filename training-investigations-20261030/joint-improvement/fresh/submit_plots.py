"""Metadata dispatch only after complete sealed evaluation."""
from pathlib import Path
import json
from runner import Runner,JobRegistry,JobSpec
root=Path(__file__).parent
x=json.loads((root/'evaluation/outcome.json').read_text());assert x['complete']
r=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner')).submit(JobSpec(name='buoyancy-joint-fresh-report',template='cpu',cpus=4,mem='16G',time_limit='00:15:00',cmd=['bash',str(root/'plot-node.sh'),str(root)],out_path=root/'plot-job-outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(root)},extra_sbatch=[f'--output={root}/plot-%j.out']),cluster='kander')
(root/'plot-job.json').write_text(json.dumps({'slurm_id':r.slurm_id}));print(r.slurm_id)
