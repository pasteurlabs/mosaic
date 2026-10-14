from pathlib import Path
import os,shutil,json,hashlib
from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.buoyancy_grid import plots
campaign=Path(os.environ['CAMPAIGN']);old=Path('/data/personal/andrinr/runner/results/mosaic/buoyancy-data-mix-20261013');suffix=Path('buoyancy-grid/optimization/solver_in_loop');receipts={}
for key in ['phiflow','jax-cfd','waterlily','oceananigans','ins-jl']:
 root=campaign/'rendered'/key;dest=root/suffix;dest.mkdir(parents=True,exist_ok=True)
 for name in ['result.json','corrector_fields.npz']:shutil.copy2(old/key/'results'/suffix/name,dest/name)
 plots.results_dir=lambda root=root:root
 charts=plots.plot_solver_in_loop(get_config('buoyancy-grid'));assert len(charts)==4
 files=[p for p in dest.iterdir() if p.suffix in ['.png','.json'] and p.name!='result.json'];assert len(files)==8,(key,[p.name for p in files]);receipts[key]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};print('RENDERED',key,flush=True)
(campaign/'render_receipt.json').write_text(json.dumps(receipts,indent=2))
