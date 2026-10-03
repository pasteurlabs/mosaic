import sys
from pathlib import Path
sys.path.insert(0, '/data/personal/andrinr/tools/slurm-runner')
from runner import JobRegistry, JobSpec, Runner
root = Path('/data/personal/andrinr/runner/surrogate/standalone-20261003')
spec = JobSpec(
    name='mosaic-standalone-surrogate',
    cmd=['/data/personal/andrinr/tools/mosaic-validation.sh', 'bash', str(root/'run.sh')],
    template='gpu', partition='dev', qos='rtx5090-pool', account='research',
    time_limit='00:30:00', gpus='gpu:5090:1', cpus=8, mem='48G',
    config_path=root/'source/mosaic/benchmarks/problems/navier_stokes_3d_grid/config.py',
    out_path=root/'job',
    env={'PROJECT_ISOLATED_COMMAND':'1', 'PYTHONUNBUFFERED':'1', 'MOSAIC_SOURCE_ROOT':str(root/'source')},
)
spec.out_path.mkdir(exist_ok=True, parents=True)
r=Runner(JobRegistry(root/'registry'), Path('/data/personal/andrinr/tools/slurm-runner/runner'))
result=r.submit(spec, cluster='kander', validate_paths=True)
print(result.slurm_id, result.job.state, result.job.reason)
