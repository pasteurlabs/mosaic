import sys
from pathlib import Path
sys.path.insert(0, '/data/personal/andrinr/tools/slurm-runner')
from runner import JobRegistry, JobSpec, Runner
root = Path('/data/personal/andrinr/runner/surrogate/replay-pilot-20261003')
spec = JobSpec(name='mosaic-replay-weak', cmd=['bash', str(root/'weak.sh')],
    template='gpu', partition='nice', qos='nice', account='research',
    time_limit='01:00:00', depends_on=['2872650'], gpus='gpu:5090:1', cpus=8, mem='48G',
    config_path=root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/sobolev.py',
    out_path=root/'job', env={'PROJECT_ISOLATED_COMMAND':'1','PYTHONUNBUFFERED':'1'})
spec.out_path.mkdir(exist_ok=True, parents=True)
r=Runner(JobRegistry(root/'registry'), Path('/data/personal/andrinr/tools/slurm-runner/runner'))
result=r.submit(spec, cluster='kander', validate_paths=True)
print(result.slurm_id, result.job.state, result.job.reason)
