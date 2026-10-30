"""Fixed common joint recipe for the main five-solver SOL gallery; external native run."""
import argparse,hashlib,importlib.util,json,time,traceback,shutil
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from helper import apply_variant,effective_params
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector,apply_corrector,normalized_loss
from mosaic.benchmarks.problems.buoyancy_grid.training import balanced_windows,directional_fd,rollout_errors,temperature_diagnostics,divergence_rms
from mosaic.benchmarks.problems.buoyancy_grid.config import problem

p=argparse.ArgumentParser();p.add_argument('--solver',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
root=Path(__file__).parent;out=a.output;out.mkdir(parents=True,exist_ok=True)
recipe={'joint':True,'center_temperature':False,'zero_mean_temperature':False,'temperature_head_scale':.1,'lr':.001,'model_seed':0,'window_seed':0,'hidden_channels':8,'horizon':8,'updates':256,'validation_seed':1,'test_seed':100,'validation_interval':64}
started=time.monotonic();bridge=None
record={'completed':False,'solver':a.solver,'recipe':recipe,'history':[],'checkpoints':[],'devices':[str(d) for d in jax.devices()],'scope':'Native joint gallery run, not canonical CI admission. Same fixed recipe across five backends, original validation1/test100 split; each own fine reference. No solver-specific tuning.'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.bool_):return bool(x)
 return x
def save():
 record['elapsed_s']=time.monotonic()-started;(out/'outcome.json').write_text(json.dumps(clean(record),indent=2,allow_nan=False))
def store(name,params):
 leaves=jax.tree.leaves(params);np.savez(out/(name+'.npz'),**{f'leaf_{i}':np.asarray(x) for i,x in enumerate(leaves)})
 return [digest(x) for x in leaves]
def load_params(path,template):
 with np.load(path) as z:
  assert len(z.files)==len(jax.tree.leaves(template))
  return jax.tree.unflatten(jax.tree.structure(template),[jnp.asarray(z[f'leaf_{i}']) for i in range(len(z.files))])
def physical(xs):
 x=np.asarray(xs,np.float64);n=x.shape[-2];u=x[...,0];v=x[...,1];t=x[...,2]
 uf=np.pad(u,((0,0),(1,0),(0,0)));vf=np.pad(v,((0,0),(0,0),(1,0)))
 div=n*(np.diff(uf,axis=1)+np.diff(vf,axis=2));m=t.mean(axis=(1,2));variance=t.var(axis=(1,2))
 uc=.5*(uf[:,1:]+uf[:,:-1]);vc=.5*(vf[:,:,1:]+vf[:,:,:-1])
 return {'heat_mean':m.tolist(),'heat_drift':(m-m[0]).tolist(),'maximum_absolute_mean_heat_drift':float(abs(m-m[0]).max()),'final_relative_heat_drift':float((m[-1]-m[0])/max(abs(m[0]),1e-12)),'temperature_min':float(t.min()),'temperature_max':float(t.max()),'undershoot':float(max(0,t[0].min()-t.min())),'overshoot':float(max(0,t.max()-t[0].max())),'temperature_variance':variance.tolist(),'kinetic_energy_collocated':(.5*np.mean(uc**2+vc**2,axis=(1,2))).tolist(),'max_mac_divergence_rms':float(np.sqrt(np.mean(div**2,axis=(1,2))).max()),'max_stored_wall_normal':float(max(abs(u[:,-1]).max(),abs(v[:,:,-1]).max())),'max_sampled_advective_cfl':float((.005*n*(np.max(abs(u[:,:-1]),axis=(1,2))+np.max(abs(v[:,:,:-1]),axis=(1,2)))).max())}
try:
 inventory=json.loads((root/'inventory.json').read_text());item=inventory['solvers'][a.solver]
 for f in item['files'].values():assert sha(f['path'])==f['sha256'],f['path']
 record['input_files']=item['files'];record['inventory_sha256']=sha(root/'inventory.json')
 data=np.load(item['files'][item['data_file']]['path']);training=np.asarray(data['training']);validation=np.asarray(data['validation']);D=jnp.asarray(data['scales'])
 for key,value in item['data_hashes'].items():assert digest(data['training' if key in ('train','training') else key])==value,key
 with np.load(item['files']['heldout_fields.npz']['path']) as z:
  reference=np.asarray(z['reference']);uncorrected=np.asarray(z['uncorrected'])
 assert training.shape==(4,101,64,64,3) and validation.shape==reference.shape==uncorrected.shape==(101,64,64,3)
 record.update(data_hashes=item['data_hashes'],reference_sha256=digest(reference),uncorrected_sha256=digest(uncorrected),scales=np.asarray(D).tolist(),reference_admission=item['reference_admission'],driver_sha256=sha(__file__),helper_sha256=sha(root/'helper.py'))
 api=Path(f'mosaic/tesseracts/buoyancy-grid/{a.solver}/tesseract_api.py').resolve()
 assert sha(api)==next(v for k,v in item['source_sha256'].items() if k.endswith('/tesseract_api.py'))
 record['native_source_sha256']=sha(api)
 phys=dict(item['recipe']['physics']);n=phys.pop('N');dt=phys.pop('dt');steps=phys.pop('steps');phys.pop('domain_extent',None)
 solver=next(s for s in problem.solvers if s.key==a.solver.replace('-','_'))
 if a.solver=='jax-cfd':
  spec=importlib.util.spec_from_file_location('native_api',api);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
  step=jax.jit(lambda x:native.apply_jit(problem.make_inputs(solver.name,x,domain_extent=1.,dt=dt,steps=steps,**phys))['result'])
 else:
  from native_bridge import NativeBridge
  bridge=NativeBridge(api)
  sample=problem.make_inputs(solver.name,jnp.asarray(reference[0]),domain_extent=1.,dt=dt,steps=steps,**phys)
  sample={k:(np.asarray(v) if hasattr(v,'dtype') else v) for k,v in sample.items()}
  jax.block_until_ready(bridge.apply(None,sample)['result']);assert len(bridge.cache)==1
  step=jax.jit(next(iter(bridge.cache.values())))
 first=np.asarray(step(jnp.asarray(reference[0])));gap=float(np.linalg.norm(first.astype(float)-uncorrected[1].astype(float))/max(np.linalg.norm(uncorrected[1].astype(float)),1e-30))
 record['cached_native_first_step_relative_gap']=gap;assert gap<1e-5
 params=init_corrector(0,8,True);record['parameter_count']=sum(x.size for x in jax.tree.leaves(params));assert record['parameter_count']==954
 record['initial_parameter_sha256']=store('initial_params',params)
 C=lambda pp,x:apply_variant(pp,x,D,recipe);CF=jax.jit(C)
 x=jnp.asarray(training[0,50]);assert np.array_equal(np.asarray(C(params,x)),np.asarray(x))
 probe=tuple({**l} for l in params);probe[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),probe[-1]['w'].shape,dtype=jnp.float32)
 actual=C(probe,x);expected=apply_corrector(effective_params(probe,recipe),x,D)
 assert np.array_equal(np.asarray(actual),np.asarray(expected))
 q=jax.random.normal(jax.random.PRNGKey(303),x.shape,dtype=jnp.float32)
 g1=jax.grad(lambda pp:jnp.vdot(C(pp,x),q))(probe);g2=jax.grad(lambda pp:jnp.vdot(apply_corrector(effective_params(pp,recipe),x,D),q))(probe)
 assert all(np.array_equal(np.asarray(l),np.asarray(r)) for l,r in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)))
 record['corrector_controls']={'zero_head_identity':True,'source_forward_bitwise':True,'source_parameter_vjp_bitwise':True}
 train=jnp.asarray(training);cases,starts=balanced_windows(256,93,4,0)
 assert {'cases':cases,'starts':starts}==item['schedule']
 record['schedule']={'cases':cases,'starts':starts};cases=jnp.asarray(cases,jnp.int32);starts=jnp.asarray(starts,jnp.int32)
 def objective(pp,target):
  x=target[0];loss=jnp.asarray(0.);cfl=jnp.asarray(0.,jnp.float32);tmin=jnp.min(x[...,2]);tmax=jnp.max(x[...,2])
  for i in range(8):
   x=C(pp,step(x));loss=loss+normalized_loss(x,target[i+1],D)
   cfl=jnp.maximum(cfl,.005*64*(jnp.max(abs(x[:-1,:,0]))+jnp.max(abs(x[:,:-1,1]))));tmin=jnp.minimum(tmin,jnp.min(x[...,2]));tmax=jnp.maximum(tmax,jnp.max(x[...,2]))
  return loss/8,jnp.stack([cfl,tmin,tmax,jnp.all(jnp.isfinite(x))])
 def gate(point):
  rows=[directional_fd(lambda pp:objective(pp,train[0,:9])[0],point,e) for e in [.003,.001,.0003]];pairs=[]
  for l,r in zip(rows[:-1],rows[1:]):
   delta=abs(l['finite_difference']-r['finite_difference'])/max(abs(l['finite_difference']),abs(r['finite_difference']),1e-8)
   pairs.append({'epsilons':[l['epsilon'],r['epsilon']],'slope_relative_disagreement':delta,'passes':l['relative_error']<.05 and r['relative_error']<.05 and delta<.05})
  return {'checks':rows,'pairs':pairs}
 replay=Path('/data/personal/andrinr/runner/results/mosaic/buoyancy-joint-improvement-20261030/train/joint-c0-m0-t0.1-lr0.001-s0') if a.solver=='jax-cfd' else None
 old=None
 if replay:
  old=json.loads((replay/'outcome.json').read_text());assert old['completed'] and old['completed_updates']==256
  assert old['recipe']['lr']==.001 and old['recipe']['temperature_head_scale']==.1 and not old['recipe']['center_temperature'] and not old['recipe']['zero_mean_temperature']
  assert old['data_hashes']==record['data_hashes'] and old['native_source_sha256']==record['native_source_sha256'] and old['helper_sha256']==record['helper_sha256'] and old['initial_parameter_sha256']==record['initial_parameter_sha256'] and old['schedule']==record['schedule']
  record['research_fd_profiles']=old['research_fd_profiles']
  record['replay_provenance']={'directory':str(replay),'outcome_sha256':sha(replay/'outcome.json'),'reason':'Identical256trainingupdates/checkpointweights; mainval1 selection replayed from all saved64updatecheckpoints, instead of previous3ICselection. No test-basedchoice.','original_validation_seeds':old['validation_seeds'],'original_training_seconds':old['training_seconds'],'original_fit_elapsed_s_including3valIC':old['fit_elapsed_s']}
 else:record['research_fd_profiles']={'initial':gate(params),'synthetic_nonzero':gate(probe)}
 rr=record['research_fd_profiles'];passed=[l['passes'] and r['passes'] for l,r in zip(rr['initial']['pairs'],rr['synthetic_nonzero']['pairs'])];rr['same_pair_passes_both']=passed
 record['research_fd_pass']=any(passed);record['public_fd_pass']=all(rr[key]['checks'][0]['relative_error']<.05 for key in ['initial','synthetic_nonzero'])
 record['admission_scope']='public_fd_pass reports unchanged two-point epsilon.003 criterion; research_fd_pass additionally requires same adjacent epsilon pair meeting5%AD/FD AND5%slope agreement atbothinitial/nonzero. Native gallery completion does not claim canonical CIadmission.'
 save();assert record['research_fd_pass'],'Supplementary research gradient criterion rejected'
 def trajectory(pp,initial):
  x=jnp.asarray(initial);xs=[np.asarray(x)]
  for _ in range(100):x=CF(pp,step(x));xs.append(np.asarray(x))
  xs=np.stack(xs);assert np.isfinite(xs).all(),'Nonfinite corrected rollout'
  return xs
 def score(xs,target):
  ee=((np.asarray(xs[1:],float)-np.asarray(target[1:],float))/np.asarray(D,float))**2
  vv=.5*np.mean(ee[...,:2]);tt=.5*np.mean(ee[...,2]);return {'total':float(vv+tt),'velocity':float(vv),'temperature':float(tt)}
 best=float('inf');selected=0;bestparams=params;training_seconds=validation_seconds=0.;finalparams=params
 fit_started=time.monotonic()
 def evaluate(pp,index):
  global best,bestparams,selected,validation_seconds
  t=time.monotonic();xs=trajectory(pp,validation[0]);s=score(xs,validation);phy=physical(xs);elapsed=time.monotonic()-t;validation_seconds+=elapsed
  row={'update':index,'loss':s['total'],'components':s,'physics':phy,'admissible_sampled_cfl':phy['max_sampled_advective_cfl']<=.5,'elapsed_s':time.monotonic()-started,'fit_elapsed_s':time.monotonic()-fit_started,'validation_seconds':elapsed,'parameter_sha256':store(f'checkpoint_{index}',pp)}
  record['checkpoints'].append(row)
  if row['admissible_sampled_cfl'] and s['total']<best:best=s['total'];bestparams=pp;selected=index
  save();print(a.solver,index,s,flush=True)
 if replay:
  for index in [0,64,128,192,256]:
   path=replay/f'checkpoint_{index}.npz';pp=load_params(path,params)
   oldrow=next(r for r in old['checkpoints'] if r['update']==index)
   assert [digest(x) for x in jax.tree.leaves(pp)]==oldrow['parameter_sha256']
   evaluate(pp,index);finalparams=pp
  record['history']=old['history'];training_seconds=old['training_seconds'];record['training_execution']='reusedimmutableJAXcheckpointsequence'
 else:
  optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.001));optstate=optimizer.init(params);vg=jax.value_and_grad(objective,has_aux=True)
  @jax.jit
  def update(pp,state,index):
   target=jax.lax.dynamic_slice_in_dim(train[cases[index]],starts[index],9,axis=0);(value,aux),g=vg(pp,target)
   delta,state=optimizer.update(g,state,pp);new=optax.apply_updates(pp,delta)
   finite=jnp.isfinite(value)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((g,new,state))]))
   gn=jnp.sqrt(sum(jnp.vdot(x,x) for x in jax.tree.leaves(g)))
   return new,state,jnp.concatenate([jnp.asarray([value,finite,gn]),aux])
  evaluate(params,0)
  for i in range(256):
   t=time.monotonic();new,state,diag=update(params,optstate,jnp.asarray(i,jnp.int32));v=np.asarray(diag);elapsed=time.monotonic()-t
   row={'update':i+1,'loss':float(v[0]),'finite':bool(v[1]),'gradient_norm':float(v[2]),'max_cfl':float(v[3]),'temperature_min':float(v[4]),'temperature_max':float(v[5]),'finite_final_state':bool(v[6]),'seconds':elapsed}
   record['history'].append(row);training_seconds+=elapsed
   assert row['finite'] and row['finite_final_state'],'Nonfinite training state/value/gradient'
   assert row['max_cfl']<=.5,f"Sampled training CFL rejected: {row['max_cfl']}"
   params,optstate=new,state
   if (i+1)%64==0:evaluate(params,i+1)
  finalparams=params;record['training_execution']='new256updates'
 assert len(record['history'])==256
 record.update(selected_update=selected,validation_loss=best,training_seconds=training_seconds,validation_seconds=validation_seconds,selection_replay_elapsed_s=time.monotonic()-fit_started if replay else None,fit_elapsed_s=time.monotonic()-fit_started,completed_updates=256,selected_parameter_sha256=store('selected_params',bestparams),final_parameter_sha256=store('final_params',finalparams))
 corrected=trajectory(bestparams,reference[0]);record['heldout_loss']=score(corrected,reference);record['uncorrected_loss']=score(uncorrected,reference)
 record['physical_properties']={name:physical(x) for name,x in [('reference',reference),('uncorrected',uncorrected),('solver_in_loop',corrected)]}
 np.savez_compressed(out/'fields.npz',reference=reference,uncorrected=uncorrected,solver_in_loop=corrected,scales=np.asarray(D))
 checks=[{'update':r['update'],'loss':r['loss']} for r in record['checkpoints']]
 candidate={'learning_rate':.001,'training_loss':[r['loss'] for r in record['history']],'completed_updates':256,'validation_loss':best,'selected_update':selected,'validation_checkpoints':checks,'training_seconds':training_seconds,'validation_seconds':validation_seconds}
 metrics={'status':'ok','execution_scope':'completednativejointgallery;supplementaryresearchadmissionnotcanonicalCI','corrector_fields':'velocity+temperature','corrector_parameters':954,'scales':np.asarray(D).tolist(),'reference':item['reference_admission'],'updates':256,'unroll':8,'eval_frames':100,'train_frames':100,'learning_rates':[.001],'seeds':{'train':0,'validation':1,'test':100},'training_cases':item['recipe']['training']['train_cases'],'gradient_check':rr['initial']['checks'][0],'gradient_check_nonzero':rr['synthetic_nonzero']['checks'][0],'public_fd_pass':record['public_fd_pass'],'research_fd_pass':record['research_fd_pass'],'uncorrected':rollout_errors(uncorrected,reference,D),'solver_in_loop':{**rollout_errors(corrected,reference,D),'selected_candidate':0,'selected_update':selected,'candidates':[candidate],'total_training_seconds':training_seconds,'total_validation_seconds':validation_seconds},'temperature_head_scale':.1,'joint_recipe':recipe,'temperature_diagnostics':{name:temperature_diagnostics(x) for name,x in [('reference',reference),('uncorrected',uncorrected),('solver_in_loop',corrected)]},'divergence_rms':{name:divergence_rms(x,1.) for name,x in [('reference',reference),('uncorrected',uncorrected),('solver_in_loop',corrected)]}}
 metrics['final_loss']=metrics['solver_in_loop']['loss'][-1]
 (out/'metrics.json').write_text(json.dumps(clean(metrics),indent=2,allow_nan=False))
 record['metrics_sha256']=sha(out/'metrics.json');record['fields_sha256']=sha(out/'fields.npz');record['completed']=True;save()
except Exception as exc:
 record['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save();raise
finally:
 if bridge is not None:bridge.close()
