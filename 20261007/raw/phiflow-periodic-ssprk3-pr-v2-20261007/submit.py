import json,hashlib
from pathlib import Path
from runner import JobRegistry, JobSpec, Runner
from runner.transport import LocalTransport
c=Path('/data/personal/andrinr/runner/results/mosaic/phiflow-periodic-ssprk3-pr-v2-20261007')
r=Runner(JobRegistry(c/'registry'),Path('/home/andrinr/slurm-runner/runner'),transport=LocalTransport())
records=[]
for n,d in [(64,2),(192,2),(32,3),(64,3)]:
 spec=JobSpec(name=f'phi-pr-{d}d-{n}',cmd=['bash',str(c/'node.sh'),str(c),str(n),str(d)],template='gpu',account='research',partition='nice',qos='nice',gpus='gpu:5090:1',time_limit='01:00:00',cpus=8,mem='64G',out_path=c/'allocations'/f'{d}d-{n}',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(c),'PYTHONUNBUFFERED':'1'},extra_sbatch=[f'--output={c}/logs/%x-%j.out','--exclude=rtx03'])
 result=r.submit(spec,cluster='kander')
 records.append({'resolution':n,'ndim':d,'result':str(result)})
 print(records[-1],flush=True)
(c/'submission.json').write_text(json.dumps({'jobs':records,'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in c.glob('*.py')}},indent=2))
