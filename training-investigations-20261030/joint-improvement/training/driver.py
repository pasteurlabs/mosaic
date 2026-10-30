"""External mixed-window joint-corrector search. Validation only; no held-out access."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gpu_adapter import GPUAdapter
from helper import apply_variant,effective_params
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector,apply_corrector,normalized_loss
from mosaic.benchmarks.problems.buoyancy_grid.training import balanced_windows,directional_fd

parser=argparse.ArgumentParser();parser.add_argument('--recipe',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
root=Path(__file__).parent;out=a.output;out.mkdir(parents=True,exist_ok=True);recipe=json.loads(a.recipe.read_text());started=time.monotonic()
record={'completed':False,'recipe':recipe,'history':[],'checkpoints':[],'devices':[str(d) for d in jax.devices()],'heldout_accessed':False}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 return x
def save():
 record['elapsed_s']=time.monotonic()-started;(out/'outcome.json').write_text(json.dumps(clean(record),indent=2,allow_nan=False))
def store(name,pp):
 leaves=jax.tree.leaves(pp);np.savez(out/(name+'.npz'),**{f'leaf_{i}':np.asarray(x) for i,x in enumerate(leaves)});return [digest(x) for x in leaves]
def physics(xs):
 x=np.asarray(xs,np.float64);n=x.shape[-2];u=x[...,0];v=x[...,1];theta=x[...,2]
 uf=np.pad(u,((0,0),(1,0),(0,0)));vf=np.pad(v,((0,0),(0,0),(1,0)));div=n*(np.diff(uf,axis=1)+np.diff(vf,axis=2))
 mean=theta.mean(axis=(1,2));variance=theta.var(axis=(1,2));uc=.5*(uf[:,1:]+uf[:,:-1]);vc=.5*(vf[:,:,1:]+vf[:,:,:-1])
 return {'heat_mean':mean.tolist(),'heat_drift_from_initial':(mean-mean[0]).tolist(),'max_abs_heat_drift':float(abs(mean-mean[0]).max()),'final_relative_heat_drift':float((mean[-1]-mean[0])/max(abs(mean[0]),1e-12)),'temperature_min':float(theta.min()),'temperature_max':float(theta.max()),'temperature_variance':variance.tolist(),'mac_divergence_rms':np.sqrt(np.mean(div**2,axis=(1,2))).tolist(),'wall_normal_max':float(max(abs(u[:,-1]).max(),abs(v[:,:,-1]).max())),'kinetic_energy_collocated':(.5*np.mean(uc**2+vc**2,axis=(1,2))).tolist(),'advective_cfl':(.005*n*(np.max(abs(u[:,:-1]),axis=(1,2))+np.max(abs(v[:,:,:-1]),axis=(1,2)))).tolist()}
try:
 inventory=json.loads((root/'inventory.json').read_text());item=inventory['solvers']['jax-cfd'];data_info=item['files'][item['data_file']]
 # Deliberately do not even read/hash any heldout file listed in the inherited inventory.
 assert sha(data_info['path'])==data_info['sha256'];data=np.load(data_info['path']);D=jnp.asarray(data['scales']);train=jnp.asarray(data['training']);validation=[data['validation']]
 for key,value in item['data_hashes'].items():assert digest(data['training' if key=='train' else key])==value
 manifest_path=Path('/data/personal/andrinr/runner/results/mosaic/buoyancy-joint-fresh-evaluation-20261030/validation-manifest.json')
 freeze=json.loads((root/'validation-freeze.json').read_text());assert sha(manifest_path)==freeze['validation_manifest_sha256'];vm=json.loads(manifest_path.read_text());assert vm['complete'] and sorted(r['seed'] for r in vm['references'])==[5,6]
 for rr in sorted(vm['references'],key=lambda r:r['seed']):
  assert sha(rr['file'])==rr['file_sha256'];xs=np.load(rr['file'])['reference'];assert xs.shape==(101,64,64,3) and digest(xs)==rr['reference_sha256'];validation.append(xs)
 record.update(data_hashes=item['data_hashes'],validation_seeds=[1,5,6],validation_reference_hashes=[digest(x) for x in validation],validation_manifest_sha256=sha(manifest_path),reference_admission={'existing':item['reference_admission'],'new_validation':vm},driver_sha256=sha(__file__),helper_sha256=sha(root/'helper.py'))
 nativepath=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py');assert sha(nativepath)==next(v for k,v in item['source_sha256'].items() if k.endswith('/tesseract_api.py'));record['native_source_sha256']=sha(nativepath)
 params=init_corrector(recipe['model_seed'],8,recipe['joint']);record['initial_parameter_sha256']=store('initial_params',params);record['parameter_count']=sum(x.size for x in jax.tree.leaves(params))
 adapter=GPUAdapter('jax-cfd',np.asarray(D),item['recipe']['physics']);step=jax.jit(lambda x:adapter.api.apply_jit({**adapter.fixed,'state0':x})['result'])
 C=lambda pp,x:apply_variant(pp,x,D,recipe);CF=jax.jit(C)
 cases,starts=balanced_windows(256,train.shape[1]-8,len(train),0);record['schedule']={'cases':cases,'starts':starts};assert record['schedule']==item['schedule'];cases=jnp.asarray(cases,jnp.int32);starts=jnp.asarray(starts,jnp.int32)
 def objective(pp,target):
  x=target[0];value=jnp.asarray(0.0);lv=jnp.asarray(0.,jnp.float32);lt=jnp.asarray(0.,jnp.float32)
  maxcfl=jnp.asarray(0.,jnp.float32);tmin=jnp.min(x[...,2]);tmax=jnp.max(x[...,2]);mass0=jnp.mean(x[...,2].astype(jnp.float64));massdrift=jnp.asarray(0.,jnp.float64);correction_mean=jnp.asarray(0.,jnp.float64);correction_actual_mean=jnp.asarray(0.,jnp.float64);correction_max=jnp.asarray(0.,jnp.float64);finite=jnp.asarray(True)
  for i in range(8):
   maxcfl=jnp.maximum(maxcfl,.005*64*(jnp.max(abs(x[:-1,:,0]))+jnp.max(abs(x[:,:-1,1]))))
   sx=step(x);x,cd=apply_variant(pp,sx,D,recipe,True);e=((x-target[i+1])/D)**2
   lv+=.5*jnp.mean(e[...,:2]);lt+=.5*jnp.mean(e[...,2]);value=value+normalized_loss(x,target[i+1],D)
   tmin=jnp.minimum(tmin,jnp.min(x[...,2]));tmax=jnp.maximum(tmax,jnp.max(x[...,2]));massdrift=jnp.maximum(massdrift,abs(jnp.mean(x[...,2].astype(jnp.float64))-mass0));correction_mean=jnp.maximum(correction_mean,abs(cd[0]));correction_actual_mean=jnp.maximum(correction_actual_mean,abs(cd[2]));correction_max=jnp.maximum(correction_max,cd[1]);finite=finite&jnp.all(jnp.isfinite(x))
   maxcfl=jnp.maximum(maxcfl,.005*64*(jnp.max(abs(x[:-1,:,0]))+jnp.max(abs(x[:,:-1,1]))))
  return value/8,jnp.stack([lv/8,lt/8,maxcfl,tmin,tmax,massdrift,correction_mean,correction_actual_mean,correction_max,finite])
 vg=jax.value_and_grad(objective,has_aux=True)
 def gate(point):
  rows=[directional_fd(lambda pp:objective(pp,train[0,:9])[0],point,e) for e in [.003,.001,.0003]];pairs=[]
  for l,r in zip(rows[:-1],rows[1:]):
   agree=abs(l['finite_difference']-r['finite_difference'])/max(abs(l['finite_difference']),abs(r['finite_difference']),1e-8)
   pairs.append({'epsilons':[l['epsilon'],r['epsilon']],'slope_relative_disagreement':agree,'passes':l['relative_error']<.05 and r['relative_error']<.05 and agree<.05})
  return {'checks':rows,'pairs':pairs}
 synthetic=tuple({**l} for l in params);synthetic[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),synthetic[-1]['w'].shape,dtype=jnp.float32)
 record['research_fd_profiles']={'initial':gate(params),'synthetic_nonzero':gate(synthetic)}
 passing=[l['passes'] and r['passes'] for l,r in zip(record['research_fd_profiles']['initial']['pairs'],record['research_fd_profiles']['synthetic_nonzero']['pairs'])];record['research_fd_profiles']['same_pair_passes_both']=passing;save();assert any(passing),'Research convergence FD gate rejected'
 optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(recipe['lr']));optstate=optimizer.init(params)
 @jax.jit
 def update(pp,state,index):
  target=jax.lax.dynamic_slice_in_dim(train[cases[index]],starts[index],9,axis=0);(value,aux),g=vg(pp,target)
  delta,state=optimizer.update(g,state,pp);new=optax.apply_updates(pp,delta)
  gn=jnp.sqrt(sum(jnp.vdot(x,x) for x in jax.tree.leaves(g)));dn=jnp.sqrt(sum(jnp.vdot(x,x) for x in jax.tree.leaves(delta)))
  finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((g,new,state))]))
  return new,state,jnp.concatenate([jnp.asarray([value,gn,dn,finite]),aux])
 def trajectory(pp,initial):
  x=jnp.asarray(initial);xs=[np.asarray(x)]
  for _ in range(100):x=CF(pp,step(x));xs.append(np.asarray(x))
  xs=np.stack(xs);assert np.isfinite(xs).all(),'Nonfinite validation rollout';return xs
 def score(xs,ref):
  e=((np.asarray(xs[1:],np.float64)-np.asarray(ref[1:],np.float64))/np.asarray(D,np.float64))**2
  lv=.5*np.mean(e[...,:2]);lt=.5*np.mean(e[...,2]);return {'total':float(lv+lt),'velocity':float(lv),'temperature':float(lt)}
 best=float('inf');bestparams=params;bestfields=None;selected=0;train_s=val_s=0.
 def evaluate(iteration):
  global best,bestparams,bestfields,selected,val_s
  t=time.monotonic();fields=[trajectory(params,v[0]) for v in validation];ss=[score(x,v) for x,v in zip(fields,validation)];aggregate={k:float(np.mean([s[k] for s in ss])) for k in ['total','velocity','temperature']};validation_physics=[physics(x) for x in fields];duration=time.monotonic()-t;val_s+=duration
  row={'update':iteration,'aggregate_loss':aggregate,'per_validation_loss':dict(zip(['1','5','6'],ss)),'physics':dict(zip(['1','5','6'],validation_physics)),'validation_cfl_admissible':all(max(x['advective_cfl'])<=.5 for x in validation_physics),'validation_seconds':duration,'fit_elapsed_s':time.monotonic()-fit_started,'parameter_sha256':store(f'checkpoint_{iteration}',params)}
  record['checkpoints'].append(row)
  if all(max(x['advective_cfl'])<=.5 for x in validation_physics) and aggregate['total']<best:best=aggregate['total'];bestparams=params;bestfields=fields;selected=iteration
  save();print(recipe['id'],iteration,aggregate,flush=True)
 record['common_preparation_seconds']=time.monotonic()-started;record['timing_scope']='Single fully JIT native trueAD+clip+Adam per window, first training compilation and validation included in fit time, shared source/reference/gate preparation excluded. No duplicate gradient instrumentation. Persistent compilation cache disabled. Native RTX not CI timing.'
 fit_started=time.monotonic();evaluate(0)
 for i in range(256):
  t=time.monotonic();new,nextstate,diag=update(params,optstate,jnp.asarray(i,jnp.int32));v=np.asarray(diag);duration=time.monotonic()-t
  row={'update':i+1,'loss':float(v[0]),'gradient_norm':float(v[1]),'parameter_step_norm':float(v[2]),'finite_parameters_gradient':bool(v[3]),'velocity_loss':float(v[4]),'temperature_loss':float(v[5]),'max_cfl':float(v[6]),'temperature_min':float(v[7]),'temperature_max':float(v[8]),'max_window_heat_drift':float(v[9]),'max_predicted_thermal_correction_mean':float(v[10]),'max_actual_thermal_correction_mean':float(v[11]),'max_thermal_correction':float(v[12]),'finite_states':bool(v[13]),'update_seconds':duration,'fit_elapsed_s':time.monotonic()-fit_started}
  record['history'].append(row);train_s+=duration
  assert row['finite_parameters_gradient'] and row['finite_states'],'Nonfinite training value/state/gradient/parameters'
  assert row['max_cfl']<=.5,f"Native CFL admissibility rejected query {row['max_cfl']}"
  params,optstate=new,nextstate
  if (i+1)%64==0:evaluate(i+1)
 record.update(selected_update=selected,validation_loss=best,selected_validation_loss=next(c['aggregate_loss'] for c in record['checkpoints'] if c['update']==selected),selected_parameter_sha256=store('selected_params',bestparams),final_parameter_sha256=store('final_params',params),training_seconds=train_s,validation_seconds=val_s,fit_elapsed_s=time.monotonic()-fit_started,completed_updates=256)
 np.savez_compressed(out/'selected_validation_fields.npz',**{f'validation_{seed}':fields for seed,fields in zip([1,5,6],bestfields)})
 record['completed']=True;save()
except Exception as exc:record['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save();raise
