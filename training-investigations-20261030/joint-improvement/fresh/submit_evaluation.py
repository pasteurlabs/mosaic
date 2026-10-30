"""Metadata dispatch only. Requires an explicit locked root selection receipt."""
from pathlib import Path
import argparse,hashlib,json
from runner import Runner,JobRegistry,JobSpec
p=argparse.ArgumentParser();p.add_argument('selection',type=Path);p.add_argument('selection_sha256');a=p.parse_args()
root=Path(__file__).parent
assert hashlib.sha256(a.selection.read_bytes()).hexdigest()==a.selection_sha256
lock=json.loads(a.selection.read_text());assert lock['locked'] and lock['fresh_evaluation_authorized']
assert lock['selection_basis']=='validation_only_IC1_5_6'
assert not (root/'evaluation').exists(),'Existing fresh results may not be overwritten'
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
r=runner.submit(JobSpec(name='buoyancy-joint-frozen-fresh-evaluation',template='gpu',gpus='gpu:5090:1',cpus=4,mem='32G',time_limit='00:15:00',cmd=['bash',str(root/'evaluation-node.sh'),str(root),str(a.selection),a.selection_sha256],out_path=root/'evaluation-job-outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(root)},extra_sbatch=[f'--output={root}/evaluation-%j.out']),cluster='kander')
(root/'evaluation-job.json').write_text(json.dumps({'slurm_id':r.slurm_id,'selection':str(a.selection),'selection_sha256':a.selection_sha256}));print(r.slurm_id)
