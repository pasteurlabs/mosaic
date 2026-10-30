from pathlib import Path
import argparse,json
from runner import Runner,JobRegistry,JobSpec
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['preview','full'],default='full');a.add_argument('--after',nargs='*',type=int,default=[]);v=a.parse_args()
p=Path(__file__).parent;campaign=p.parent
metadata=p/(v.mode+'-job.json');assert not metadata.exists(),'Refusing duplicate render submission; preserve prior results'
extra=[f'--output={p}/{v.mode}-%j.out','--exclude=dgx[01-06]']
if v.after:extra.append('--dependency=afterany:'+':'.join(map(str,v.after)))
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
r=runner.submit(JobSpec(name='buoyancy-main-joint-render-'+v.mode,template='cpu',cpus=4,mem='16G',time_limit='00:20:00',cmd=['bash',str(p/'node.sh'),str(campaign),v.mode],out_path=p/(v.mode+'-job-outcome.json'),env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(campaign)},extra_sbatch=extra),cluster='kander')
metadata.write_text(json.dumps({'slurm_id':r.slurm_id,'mode':v.mode,'after':v.after},indent=2));print(r.slurm_id)
