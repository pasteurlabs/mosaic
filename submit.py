from pathlib import Path
import sys
from runner import Runner,JobRegistry,JobSpec
campaign=Path(sys.argv[1]);mode=sys.argv[2]
runner=Runner(JobRegistry(Path('/data/personal/andrinr/runner/registry')),Path('/home/andrinr/slurm-runner/runner'))
spec=JobSpec(name='fd-path-eval-'+mode,template='cpu',cpus=4,mem='24G',time_limit='01:00:00',depends_on=[int(sys.argv[3])] if len(sys.argv)>3 else [],cmd=['bash',str(campaign/'node.sh'),str(campaign)],out_path=campaign/'outcome.json',env={'PROJECT_ISOLATED_COMMAND':'1','PROJECT_REPO_ROOT':str(campaign)},extra_sbatch=['--nodelist=rtx10',f'--output={campaign}/slurm-%j.out'])
r=runner.submit(spec,cluster='kander');(campaign/'submission.txt').write_text(str(r));print(r)
