"""Canonical buoyancy SOL plots from completed native joint runs only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os
import numpy as np
from PIL import Image
from mosaic.benchmarks.problems import get_config
from mosaic.benchmarks.problems.buoyancy_grid.plots import plot_solver_in_loop
from mosaic.benchmarks.problems.buoyancy_grid.training import rollout_errors
from mosaic.benchmarks.problems.shared.plots.charts.common import save_chart

parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--allow-subset',action='store_true')
args=parser.parse_args();p=args.campaign;render=p/'render'
target=render/('preview-results' if args.allow_subset else 'results')
os.environ['MOSAIC_RESULTS_DIR']=str(target)
out=target/'buoyancy-grid/optimization/solver_in_loop'
assert not out.exists(),f'Preserve previous render directory: {out}'
out.mkdir(parents=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
mapping={'phiflow':'PhiFlow','jax-cfd':'JAX-CFD','ins-jl':'INS.jl','waterlily':'WaterLily','oceananigans':'Oceananigans'}
cfg=get_config('buoyancy-grid')
physics={'N':64,'nu':.0005,'diffusivity':.0005,'buoyancy':1.,'dt':.005,'steps':8,'domain_extent':1.}
training={'updates':256,'unroll':8,'train_frames':100,'eval_frames':100,'reference_factor':2,'learning_rates':[.001],'hidden_channels':8,'correct_temperature':True,'temperature_head_scale':.1,'model_seed':0,'window_seed':0,'validation_seed':1,'test_seed':100,'validation_interval':64}
result={'problem':'buoyancy-grid','suite':'optimization','experiment':'solver_in_loop','params':{'physics':physics,'training':training},'results':[],'provenance':{'scope':'Native joint-corrector research matrix; not canonical container CI admission or CI timing.','public_benchmark_defaults_changed':False,'recipe_selection':'Fixed transfer recipe; no per-backend tuning. Original main validationIC1/testIC100.','corrector':'joint velocity+temperature, raw input, no heat constraint, thermal head scale0.1','source_commit':'1c714dd9ee393a87e9a5a6710289c0ec1b783481'}}
snapshots={};names=[];sources={};stats=[]
for slug,name in mapping.items():
 folder=p/slug;required=[folder/f for f in ['outcome.json','metrics.json','fields.npz']]
 if not all(f.exists() for f in required):
  if args.allow_subset:continue
  raise RuntimeError(f'Missing completed backend files: {name}')
 outcome=json.loads(required[0].read_text())
 if not outcome.get('completed'):
  if args.allow_subset:continue
  raise RuntimeError(f'Backend not completed: {name}: {outcome.get("failure")}')
 metrics=json.loads(required[1].read_text())
 assert metrics.get('status')=='ok' and metrics['corrector_fields']=='velocity+temperature',name
 assert outcome.get('research_fd_pass') is True,f'No supplementary research admission: {name}'
 assert metrics['solver_in_loop']['selected_candidate']==0
 candidate=metrics['solver_in_loop']['candidates'][0]
 assert candidate['completed_updates']==256
 points=candidate['validation_checkpoints']
 assert [int(r['update']) for r in points]==[0,64,128,192,256],name
 selected=int(metrics['solver_in_loop']['selected_update'])
 eligible=[r for r in outcome['checkpoints'] if r.get('admissible_sampled_cfl',True)]
 assert selected==min(eligible,key=lambda r:(r['loss'],r['update']))['update']
 expected_recipe={'joint':True,'center_temperature':False,'zero_mean_temperature':False,'temperature_head_scale':.1,'lr':.001,'model_seed':0,'window_seed':0,'hidden_channels':8,'horizon':8,'updates':256,'validation_seed':1,'test_seed':100,'validation_interval':64}
 assert outcome['recipe']==expected_recipe,name
 scales=np.asarray(metrics['scales'],np.float32)
 assert scales.shape==(3,) and np.isfinite(scales).all() and np.all(scales>0)
 z=np.load(required[2],allow_pickle=False);arrays={arm:np.asarray(z[arm]) for arm in ['reference','uncorrected','solver_in_loop']}
 for arm,a in arrays.items():assert a.shape==(101,64,64,3) and np.isfinite(a).all(),(name,arm,a.shape)
 assert all(np.array_equal(a[0],arrays['reference'][0]) for a in arrays.values()),name
 for arm in ['uncorrected','solver_in_loop']:
  checked=rollout_errors(arrays[arm],arrays['reference'],scales)
  for metric in ['loss','velocity_mse','scalar_mse']:assert np.allclose(checked[metric],metrics[arm][metric],rtol=2e-6,atol=1e-10),(name,arm,metric)
 index=len(names);names.append(name)
 for arm,a in arrays.items():snapshots[f'{arm}_{index}']=a
 result['results'].append({'solver':name,'sweep_value':'64','metrics':metrics,'joint_provenance':{'native_outcome_sha256':sha(required[0]),'public_fixed_epsilon_pass':bool(outcome.get('public_fd_pass')),'supplementary_research_pass':bool(outcome.get('research_fd_pass'))}})
 sources[name]={'directory':str(folder),'files':{f.name:sha(f) for f in required},'public_fixed_epsilon_pass':bool(outcome.get('public_fd_pass')),'supplementary_research_pass':bool(outcome.get('research_fd_pass')),'selected_update':selected}
 stats.append({'solver':name,'uncorrected_mean_loss':float(np.mean(metrics['uncorrected']['loss'][1:])),'joint_mean_loss':float(np.mean(metrics['solver_in_loop']['loss'][1:])),'selected_update':selected,'public_fixed_epsilon_pass':bool(outcome.get('public_fd_pass'))})
assert names and (args.allow_subset or set(names)==set(mapping.values()))
cfg.solvers=[s for s in cfg.solvers if s.name in names]
snapshots['solver_names']=np.asarray(names)
np.savez_compressed(out/'corrector_fields.npz',**snapshots)
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
charts=plot_solver_in_loop(cfg);assert len(charts)==2
for chart,stem,title in zip(charts,['rollout_errors','validation_loss'],['Joint corrector: held-out velocity and temperature errors','Joint corrector: validation on IC1 (selected checkpoints marked)']):
 save_chart(chart.properties(title=title),stem,out)
for stem in ['rollout_errors','validation_loss']:
 spec=json.loads((out/(stem+'.vl.json')).read_text());rows=[r for data in spec.get('datasets',{}).values() for r in data]
 assert {r['solver'] for r in rows}==set(names),stem
anim=json.loads((out/'error_evolution.json').read_text())
assert set(anim['solvers'])==set(names) and anim['frames']==101 and anim['frame_interval']==.04
assert anim['palettes']['temperature']=='sequential red' and anim['palettes']['velocity']=='sequential blue'
if 'JAX-CFD' in names:assert anim['orientation']['solver']=='JAX-CFD'
with Image.open(out/'error_evolution.gif') as im:assert im.n_frames==101
for file in ['rollout_errors.png','validation_loss.png','error_evolution.png']:
 with Image.open(out/file) as im:assert min(im.size)>100
modules=[__import__('mosaic.benchmarks.problems.buoyancy_grid.'+name,fromlist=['']) for name in ['plots','animation']]
receipt={'complete':True,'preview_subset':args.allow_subset,'created_utc':datetime.now(timezone.utc).isoformat(),'scope':result['provenance'],'source_data':sources,'solvers':names,'results':stats,'output_directory':str(out),'plot_source_sha256':{str(Path(m.__file__)):sha(Path(m.__file__)) for m in modules},'source_training_archive_sha256':sha(p/'source.tar'),'frames':101,'unchanged_public_fixed_epsilon_failures':[name for name,r in sources.items() if not r['public_fixed_epsilon_pass']],'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
(render/('preview-receipt.json' if args.allow_subset else 'render-receipt.json')).write_text(json.dumps(receipt,indent=2,allow_nan=False))
print(json.dumps({'complete':True,'solvers':names,'out':str(out),'public_fixed_epsilon_failures':receipt['unchanged_public_fixed_epsilon_failures'],'results':stats},indent=2),flush=True)
