"""Evaluate only the root-frozen candidate set on sealed IC101..105."""
import argparse,hashlib,importlib.util,json,time,traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--selection-sha256',required=True);a=p.parse_args()
root=Path(__file__).parent;out=root/'evaluation';out.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
assert sha(a.selection)==a.selection_sha256,'Selection receipt changed'
lock=json.loads(a.selection.read_text());assert lock['locked'] is True and lock['fresh_evaluation_authorized'] is True
assert lock['selection_basis']=='validation_only_IC1_5_6'
assert lock['fresh_test_seeds']==[101,102,103,104,105]
models=lock['models'];assert 1<=len(models)<=9
groups=sorted(set(x['group'] for x in models));assert len(groups)<=3 and 'velocity' in groups
for group in groups:assert sorted(x['model_seed'] for x in models if x['group']==group)==[0,1,2]
refrecord=json.loads((root/'references/outcome.json').read_text());assert refrecord['complete']
fresh=[r for r in refrecord['rows'] if r['role']=='sealed_fresh_test'];assert [r['seed'] for r in fresh]==lock['fresh_test_seeds'] and all(r['admitted'] for r in fresh)
assert sha(lock['helper_path'])==lock['helper_sha256'],'Helper changed after selection freeze'
spec=importlib.util.spec_from_file_location('frozen_helper',lock['helper_path']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
nativepath=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py')
assert sha(nativepath)==refrecord['native_sha256']==lock['native_sha256']
spec=importlib.util.spec_from_file_location('native_api',nativepath);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
fixed={'viscosity':jnp.asarray([.0005],jnp.float32),'diffusivity':jnp.asarray([.0005],jnp.float32),'buoyancy':jnp.asarray([1.],jnp.float32),'dt':jnp.asarray([.005],jnp.float32),'steps':8,'domain_extent':1.}
step=jax.jit(lambda x:native.apply_jit({**fixed,'state0':x})['result'])
D=np.asarray(refrecord['scales'],np.float32);assert digest(D)==refrecord['scales_sha256']==lock['scales_sha256']
record={'complete':False,'selection_path':str(a.selection),'selection_sha256':a.selection_sha256,'source_head':lock['source_head'],'native_sha256':sha(nativepath),'helper_sha256':lock['helper_sha256'],'reference_outcome_sha256':sha(root/'references/outcome.json'),'scales':D.tolist(),'models':[],'baselines':[],'devices':[str(d) for d in jax.devices()],'notes':['Fresh IC101..105 were not used for model, LR or checkpoint selection.','Five ICs times three initializations are not 15 independent models.','Loss averages predicted frames1..100; equal .5 velocity/.5 temperature weights after fixed training RMS normalization.','Physics uses float64 reductions of stored float32 canonical fields; temperature is dimensionless anomaly. Native RTX evaluationcost is not CIperformance.']};begun=time.monotonic()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x) for x in v]
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.integer):return int(v)
 return v
def save():
 record['elapsed_s']=time.monotonic()-begun;(out/'outcome.json').write_text(json.dumps(clean(record),indent=2,allow_nan=False))
def metrics(xs,ref):
 x=np.asarray(xs,np.float64);target=np.asarray(ref,np.float64);error=((x-target)/D.astype(float))**2
 velocity=np.mean(error[...,:2],axis=(1,2,3));temperature=np.mean(error[...,2],axis=(1,2));total=.5*(velocity+temperature)
 return {'total':float(total[1:].mean()),'velocity_contribution':float(.5*velocity[1:].mean()),'temperature_contribution':float(.5*temperature[1:].mean()),'frame_total':total.tolist(),'frame_velocity_mse':velocity.tolist(),'frame_temperature_mse':temperature.tolist()}
def physics(xs):
 x=np.asarray(xs,np.float64);u=x[...,0];v=x[...,1];t=x[...,2];uf=np.pad(u,((0,0),(1,0),(0,0)));vf=np.pad(v,((0,0),(0,0),(1,0)))
 div=64*(np.diff(uf,axis=1)+np.diff(vf,axis=2));uc=.5*(uf[:,1:]+uf[:,:-1]);vc=.5*(vf[:,:,1:]+vf[:,:,:-1]);heat=t.mean(axis=(1,2));variance=t.var(axis=(1,2))
 return {'heat_mean':heat.tolist(),'heat_drift_absolute_max':float(np.max(abs(heat-heat[0]))),'heat_drift_relative_final':float((heat[-1]-heat[0])/max(abs(heat[0]),1e-12)),'temperature_min':float(t.min()),'temperature_max':float(t.max()),'undershoot':float(max(0,t[0].min()-t.min())),'overshoot':float(max(0,t.max()-t[0].max())),'temperature_variance':variance.tolist(),'kinetic_energy_collocated':(.5*np.mean(uc**2+vc**2,axis=(1,2))).tolist(),'max_divergence_rms':float(np.sqrt(np.mean(div**2,axis=(1,2))).max()),'max_stored_wall_normal':float(max(abs(u[:,-1]).max(),abs(v[:,:,-1]).max())),'max_velocity_cfl':float(np.max(.005*64*(np.max(abs(u[:,:-1]),axis=(1,2))+np.max(abs(v[:,:,:-1]),axis=(1,2)))))}
datasets={}
for item in fresh:
 assert sha(item['file'])==item['file_sha256']
 z=np.load(item['file']);ref=z['reference'];base=z['uncorrected']
 assert digest(ref)==item['reference_sha256'] and digest(base)==item['baseline_sha256']
 assert digest(z['scales'])==digest(D)
 datasets[item['seed']]=(ref,base)
 record['baselines'].append({'ic_seed':item['seed'],'score':metrics(base,ref),'uncorrected_physics':physics(base),'reference_physics':physics(ref)})
save();CF_cache={}
for model in models:
 row={'group':model['group'],'model_seed':model['model_seed'],'selected_update':model['selected_update'],'selected_validation_mean':model['selected_validation_mean'],'recipe_path':model['recipe_path'],'recipe_sha256':model['recipe_sha256'],'params_path':model['params_path'],'params_file_sha256':model['params_file_sha256'],'parameter_sha256':model['parameter_sha256'],'cases':[],'complete':False};start=time.monotonic()
 try:
  assert sha(model['recipe_path'])==model['recipe_sha256']
  assert sha(model['params_path'])==model['params_file_sha256']
  recipe=json.loads(Path(model['recipe_path']).read_text());assert int(recipe['model_seed'])==model['model_seed']
  params0=init_corrector(model['model_seed'],8,recipe['joint']);z=np.load(model['params_path']);assert len(z.files)==len(jax.tree.leaves(params0))
  params=jax.tree.unflatten(jax.tree.structure(params0),[jnp.asarray(z[f'leaf_{i}']) for i in range(len(z.files))]);assert [digest(x) for x in jax.tree.leaves(params)]==model['parameter_sha256']
  assert [x.shape for x in jax.tree.leaves(params)]==[x.shape for x in jax.tree.leaves(params0)]
  row['recipe']=recipe
  key=json.dumps({k:recipe.get(k) for k in ['joint','center_temperature','zero_mean_temperature','temperature_head_scale']},sort_keys=True)
  if key not in CF_cache:CF_cache[key]=jax.jit(lambda pp,x,rr=recipe:helper.apply_variant(pp,x,jnp.asarray(D),rr))
  CF=CF_cache[key]
  for ic_seed,(reference,baseline) in datasets.items():
   x=jnp.asarray(reference[0]);states=[np.asarray(x)]
   for _ in range(100):x=CF(params,step(x));states.append(np.asarray(x))
   states=np.stack(states)
   if not np.isfinite(states).all():raise FloatingPointError(f'Nonfinite test trajectory IC{ic_seed}')
   fields=out/f"{model['group']}-s{model['model_seed']}-ic{ic_seed}.npz";np.savez_compressed(fields,reference=reference,uncorrected=baseline,corrected=states,scales=D)
   row['cases'].append({'ic_seed':ic_seed,'score':metrics(states,reference),'physics':physics(states),'fields_path':str(fields),'fields_sha256':sha(fields)})
  row['mean_fresh_loss']=float(np.mean([c['score']['total'] for c in row['cases']]))
  row['complete']=True
 except Exception as e:row['failure']={'message':str(e),'traceback':traceback.format_exc()}
 row['evaluation_seconds']=time.monotonic()-start;record['models'].append(row);save();print(json.dumps({'group':row['group'],'seed':row['model_seed'],'complete':row['complete'],'failure':row.get('failure'),'mean_fresh_loss':row.get('mean_fresh_loss')}),flush=True)
record['paired_comparisons']=[]
for group in groups:
 if group=='velocity':continue
 for seed in [0,1,2]:
  pairs={m['group']:m for m in record['models'] if m['model_seed']==seed and m['complete']}
  if group in pairs and 'velocity' in pairs:
   a=pairs[group];b=pairs['velocity'];case_ratios=[x['score']['total']/y['score']['total'] for x,y in zip(a['cases'],b['cases'],strict=True)]
   record['paired_comparisons'].append({'group':group,'model_seed':seed,'ratio_mean_fresh_loss':a['mean_fresh_loss']/b['mean_fresh_loss'],'case_ratio':dict(zip(lock['fresh_test_seeds'],case_ratios,strict=True))})
record['complete']=all(m['complete'] for m in record['models']) and len(record['models'])==len(models);save()
