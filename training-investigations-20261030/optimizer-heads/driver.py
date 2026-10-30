"""Bounded fixed-batch native optimizer comparison; research only."""
import argparse, hashlib, json, time, traceback
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from scipy.optimize import minimize
from gpu_adapter import GPUAdapter
from optimizers import build,check_muon
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector,apply_corrector
from mosaic.benchmarks.problems.buoyancy_grid.training import directional_fd

parser=argparse.ArgumentParser()
parser.add_argument('--recipe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();root=Path(__file__).parent;out=a.output;out.mkdir(parents=True,exist_ok=True)
recipe=json.loads(a.recipe.read_text());begun=time.monotonic()
record={'completed':False,'recipe':recipe,'history':[],'checkpoints':[],'accepted':[],'gradient_diagnostics':[],'devices':[str(d) for d in jax.devices()]}
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 return x
def save():
 record['elapsed_s']=time.monotonic()-begun
 (out/'outcome.json').write_text(json.dumps(clean(record),indent=2,allow_nan=False))
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def store(name,params):
 leaves=jax.tree.leaves(params)
 np.savez(out/(name+'.npz'),**{f'leaf_{i}':np.asarray(x) for i,x in enumerate(leaves)})
 return [digest(x) for x in leaves]
def tree_dot(a,b):return sum(jnp.vdot(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True))
def flat_norm(a):return float(jnp.sqrt(tree_dot(a,a)))
def step_norms(delta):
 result={'shared_trunk':flat_norm(delta[:-1]),'velocity_head':flat_norm({k:x[...,0] for k,x in delta[-1].items()})}
 result['temperature_head']=flat_norm({k:x[...,1] for k,x in delta[-1].items()}) if recipe['joint'] else None
 return result
try:
 inventory=json.loads((root/'inventory.json').read_text());item=inventory['solvers']['jax-cfd']
 for entry in item['files'].values():assert sha(entry['path'])==entry['sha256']
 native=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py')
 assert sha(native)==next(v for k,v in item['source_sha256'].items() if k.endswith('/tesseract_api.py'))
 data=np.load(item['files'][item['data_file']]['path']);D=jnp.asarray(data['scales']);train=jnp.asarray(data['training']);validation=data['validation'];reference=np.load(item['files']['heldout_fields.npz']['path'])['reference']
 for key,value in item['data_hashes'].items():assert digest(data['training' if key=='train' else key])==value
 record.update(data_hashes=item['data_hashes'],reference_admission=item['reference_admission'],native_sha256=sha(native),driver_sha256=sha(__file__),inventory_sha256=sha(root/'inventory.json'))
 params=init_corrector(recipe['seed'],8,recipe['joint']);record['initial_parameter_sha256']=store('initial_params',params)
 head_scales=jnp.asarray(recipe.get('head_scales',[1.,1.])[:2 if recipe['joint'] else 1],dtype=jnp.float32)
 def effective(pp):
  return (*pp[:-1],{'w':pp[-1]['w']*head_scales,'b':pp[-1]['b']*head_scales})
 def C(pp,x):return apply_corrector(effective(pp),x,D)
 adapter=GPUAdapter('jax-cfd',np.asarray(D),item['recipe']['physics'])
 step=jax.jit(lambda x:adapter.api.apply_jit({**adapter.fixed,'state0':x})['result'])
 starts=[0,24,48,72];targets=[train[i,start:start+9] for i,start in enumerate(starts)]
 record['objective']={'cases':[0,1,2,3],'starts':starts,'horizon':8,'batch_size':4,'description':'same deterministic mean of four H8 windows for every optimizer oracle; different from shuffled public training','raw_temperature_input':True}
 def component_loss(pp,target):
  x=target[0];lv=jnp.asarray(0.,jnp.float32);lt=jnp.asarray(0.,jnp.float32)
  maxcfl=jnp.asarray(0.,jnp.float32);tmin=jnp.min(x[...,2]);tmax=jnp.max(x[...,2]);wall=jnp.asarray(0.,jnp.float32);heat_drift=jnp.asarray(0.,jnp.float32);heat0=jnp.mean(x[...,2])
  for i in range(8):
   x=C(pp,step(x));error=((x-target[i+1])/D)**2
   lv=lv+.5*jnp.mean(error[...,:2]);lt=lt+.5*jnp.mean(error[...,2])
   maxcfl=jnp.maximum(maxcfl,.005*64*(jnp.max(abs(x[:-1,:,0]))+jnp.max(abs(x[:,:-1,1]))))
   tmin=jnp.minimum(tmin,jnp.min(x[...,2]));tmax=jnp.maximum(tmax,jnp.max(x[...,2]));wall=jnp.maximum(wall,jnp.maximum(jnp.max(abs(x[-1,:,0])),jnp.max(abs(x[:,-1,1]))));heat_drift=jnp.maximum(heat_drift,abs(jnp.mean(x[...,2])-heat0))
  return (lv+lt)/8,(jnp.stack([lv/8,lt/8]),jnp.stack([maxcfl,tmin,tmax,wall,heat_drift]))
 vg=jax.jit(jax.value_and_grad(component_loss,has_aux=True))
 component_jac=jax.jit(jax.jacrev(lambda pp,target:component_loss(pp,target)[1][0]))
 def gate(point):
  rows=[directional_fd(lambda pp:component_loss(pp,train[0,:9])[0],point,e) for e in [.003,.001,.0003]]
  pairs=[]
  for x,y in zip(rows[:-1],rows[1:]):
   delta=abs(x['finite_difference']-y['finite_difference'])/max(abs(x['finite_difference']),abs(y['finite_difference']),1e-8)
   pairs.append({'epsilons':[x['epsilon'],y['epsilon']],'slope_relative_disagreement':delta,'passes':x['relative_error']<.05 and y['relative_error']<.05 and delta<.05})
  return {'checks':rows,'pairs':pairs}
 synthetic=tuple({**layer} for layer in params);synthetic[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),synthetic[-1]['w'].shape,dtype=jnp.float32)
 record['gate']={'initial':gate(params),'synthetic_nonzero':gate(synthetic),'scope':'Research convergence criterion; original fixed-epsilon initial/synthetic results retained, no public gate change'}
 passing=[x['passes'] and y['passes'] for x,y in zip(record['gate']['initial']['pairs'],record['gate']['synthetic_nonzero']['pairs'])]
 record['gate']['same_pair_passes_both']=passing;save();assert any(passing),'Research convergence FD gate rejected'
 CF=jax.jit(C)
 def trajectory(pp,initial,corrected=True):
  x=jnp.asarray(initial);xs=[np.asarray(x)]
  for i in range(100):
   x=step(x);x=CF(pp,x) if corrected else x
   xs.append(np.asarray(x))
  xs=np.stack(xs)
  assert np.isfinite(xs).all(),'nonfinite rollout'
  return xs
 def score(xs,ref):
  error=((np.asarray(xs[1:],np.float64)-ref[1:])/np.asarray(D,np.float64))**2
  lv=.5*np.mean(error[...,:2]);lt=.5*np.mean(error[...,2])
  return {'total':float(lv+lt),'velocity':float(lv),'temperature':float(lt)}
 def physics(xs):
  x=np.asarray(xs,np.float64);u=x[...,0];v=x[...,1];theta=x[...,2];uf=np.pad(u,((0,0),(1,0),(0,0)));vf=np.pad(v,((0,0),(0,0),(1,0)))
  div=64*(np.diff(uf,axis=1)+np.diff(vf,axis=2));heat=theta.mean(axis=(1,2));variance=theta.var(axis=(1,2));uc=.5*(uf[:,1:]+uf[:,:-1]);vc=.5*(vf[:,:,1:]+vf[:,:,:-1])
  return {'mean_temperature':heat.tolist(),'heat_drift_relative_final':float((heat[-1]-heat[0])/max(abs(heat[0]),1e-12)),'maximum_abs_heat_drift':float(np.max(abs(heat-heat[0]))),'minimum_temperature':float(theta.min()),'maximum_temperature':float(theta.max()),'maximum_velocity_cfl':float(np.max(.005*64*(np.max(abs(u[:,:-1]),axis=(1,2))+np.max(abs(v[:,:,:-1]),axis=(1,2))))),'maximum_mac_divergence_rms':float(np.sqrt(np.mean(div**2,axis=(1,2))).max()),'maximum_stored_wall_normal':float(max(abs(u[:,-1]).max(),abs(v[:,:,-1]).max())),'temperature_variance':variance.tolist(),'kinetic_energy':(.5*np.mean(uc**2+vc**2,axis=(1,2))).tolist()}
 optimizer=None;state=None
 if recipe['optimizer']!='lbfgs':
  optimizer=build(params,recipe['optimizer'],recipe['lr'],64);state=optimizer.init(params)
  if recipe['optimizer']=='muon':record['muon_partition_check']=check_muon(params)
 @jax.jit
 def first_order_update(pp,state,g):
  delta,next_state=optimizer.update(g,state,pp)
  return optax.apply_updates(pp,delta),next_state,delta
 best=float('inf');bestparams=params;selected_call=0;last_accepted=params;last_validated=None;next_validation=4
 train_seconds=validation_seconds=gradient_diagnostic_seconds=0.;gradient_diagnostic_window_gradients=0
 def phash(pp):return [digest(x) for x in jax.tree.leaves(pp)]
 def diagnostics(pp):
  global gradient_diagnostic_seconds,gradient_diagnostic_window_gradients
  t=time.monotonic();gg=[component_jac(pp,target) for target in targets]
  j=jax.tree.map(lambda *xs:sum(xs)/4,*gg)
  gv=jax.tree.map(lambda x:x[0],j);gt=jax.tree.map(lambda x:x[1],j)
  nv=flat_norm(gv);nt=flat_norm(gt);cos=float(tree_dot(gv,gt))/(nv*nt) if nv*nt>1e-20 else None
  # Distinguish shared-trunk conflicts from final-head coupling.
  def subgroup(grads,name):
   if name=='shared_trunk':return grads[:-1]
   if name=='velocity_head':return {key:x[...,0] for key,x in grads[-1].items()}
   if recipe['joint']:return {key:x[...,1] for key,x in grads[-1].items()}
   return {'empty':jnp.zeros((1,),jnp.float32)}
  blocks={}
  for name in ['shared_trunk','velocity_head','temperature_head']:
   a=subgroup(gv,name);b=subgroup(gt,name);av=flat_norm(a);bt=flat_norm(b);blocks[name]={'velocity_loss_gradient_norm':av,'temperature_loss_gradient_norm':bt,'cosine':float(tree_dot(a,b))/(av*bt) if av*bt>1e-20 else None}
  duration=time.monotonic()-t;gradient_diagnostic_seconds+=duration;gradient_diagnostic_window_gradients+=8
  return {'velocity_loss_gradient_norm':nv,'temperature_loss_gradient_norm':nt,'cosine':cos,'blocks':blocks,'seconds':duration}
 def evaluate(pp,reason):
  global best,bestparams,selected_call,last_validated,validation_seconds
  t=time.monotonic();xs=trajectory(pp,validation[0]);s=score(xs,validation);validation_seconds+=time.monotonic()-t
  hashes=phash(pp);row={'oracle_calls':len(record['history']),'optimizer_window_evaluations':4*len(record['history']),'accepted_iterations':len(record['accepted']),'reason':reason,'loss':s,'fit_elapsed_s':time.monotonic()-fit_started,'parameter_sha256':hashes}
  record['checkpoints'].append(row);last_validated=hashes
  if s['total']<best:best=s['total'];bestparams=pp;selected_call=len(record['history']);record['selected_parameter_sha256']=store('selected_params',pp)
  record['gradient_diagnostics'].append({'oracle_calls':len(record['history']),'accepted_iterations':len(record['accepted']),'parameter_sha256':hashes,**diagnostics(pp),'fit_elapsed_s':time.monotonic()-fit_started})
  save();print(recipe['id'],reason,len(record['history']),s,flush=True)
 class BudgetExhausted(Exception):pass
 def oracle(pp):
  global train_seconds
  if len(record['history'])>=64:raise BudgetExhausted('Hard64 objective-gradient calls (256 training windows)')
  t=time.monotonic();results=[vg(pp,target) for target in targets]
  vals=[float(r[0][0]) for r in results];components=np.stack([np.asarray(r[0][1][0]) for r in results]);physical=np.stack([np.asarray(r[0][1][1]) for r in results]);g=jax.tree.map(lambda *xs:sum(xs)/4,*[r[1] for r in results])
  gradflat,_=ravel_pytree(g);gn=float(np.linalg.norm(np.asarray(gradflat)));duration=time.monotonic()-t;train_seconds+=duration
  value=float(np.mean(vals))
  row={'oracle_call':len(record['history'])+1,'optimizer_window_evaluations':4*(len(record['history'])+1),'loss':value,'velocity_loss':float(components[:,0].mean()),'temperature_loss':float(components[:,1].mean()),'gradient_norm':gn,'max_advective_cfl':float(physical[:,0].max()),'temperature_min':float(physical[:,1].min()),'temperature_max':float(physical[:,2].max()),'stored_wall_normal_max':float(physical[:,3].max()),'max_window_heat_drift':float(physical[:,4].max()),'seconds':duration,'fit_elapsed_s':time.monotonic()-fit_started,'accepted':False,'parameter_sha256':phash(pp)}
  record['history'].append(row)
  assert np.isfinite(value) and np.isfinite(np.asarray(gradflat)).all(),'nonfinite training oracle'
  assert row['max_advective_cfl']<=.5,f"Native CFL admissibility rejected training query {row['max_advective_cfl']}"
  return value,g
 record['common_preparation_seconds']=time.monotonic()-begun
 record['compute_scope']='First optimizer objective compilation, all component-gradient diagnostics, validation and optimizer steps included in fit_elapsed_s. Native/JAX/reference/gate preparation outside. Optimizer oracles limited64; diagnostic gradient work reported separately. Persistent cache disabled.'
 fit_started=time.monotonic();evaluate(params,'initial')
 termination={}
 if recipe['optimizer']=='lbfgs':
  flat0,unravel=ravel_pytree(params)
  def scipy_oracle(vector):
   pp=unravel(jnp.asarray(vector,dtype=flat0.dtype));value,g=oracle(pp);flatg,_=ravel_pytree(g);return value,np.asarray(flatg,np.float64)
  def callback(vector):
   global last_accepted,next_validation
   pp=unravel(jnp.asarray(vector,dtype=flat0.dtype));change=jax.tree.map(lambda x,y:x-y,pp,last_accepted);last_accepted=pp;hashes=phash(pp)
   for row in reversed(record['history']):
    if row['parameter_sha256']==hashes:row['accepted']=True;break
   record['accepted'].append({'iteration':len(record['accepted'])+1,'oracle_calls':len(record['history']),'parameter_sha256':hashes,'fit_elapsed_s':time.monotonic()-fit_started,'step_norms':step_norms(change)})
   if len(record['history'])>=next_validation:
    evaluate(pp,'accepted iterate crossing four-call milestone');next_validation=(len(record['history'])//4+1)*4
  try:
   result=minimize(scipy_oracle,np.asarray(flat0,np.float64),jac=True,method='L-BFGS-B',callback=callback,options={'maxfun':64,'maxiter':64,'maxls':10,'maxcor':10,'ftol':1e-12,'gtol':1e-8})
   last_accepted=unravel(jnp.asarray(result.x,dtype=flat0.dtype))
   termination={'message':str(result.message),'status':int(result.status),'success':bool(result.success),'iterations':int(result.nit),'scipy_nfev':int(result.nfev)}
  except BudgetExhausted as exc:termination={'message':str(exc),'status':'hard_budget_exhausted','iterations':len(record['accepted'])}
 else:
  for i in range(64):
   value,g=oracle(params);new,state,delta=first_order_update(params,state,g)
   assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((new,state))),'nonfinite parameters/optimizer state'
   params=new;last_accepted=params;record['history'][-1]['accepted']=True
   record['history'][-1]['parameter_step_norm']=flat_norm(delta);record['history'][-1]['step_norms']=step_norms(delta)
   record['accepted'].append({'iteration':i+1,'oracle_calls':len(record['history']),'parameter_sha256':phash(params),'fit_elapsed_s':time.monotonic()-fit_started})
   if (i+1)%4==0:evaluate(params,'four-update checkpoint')
  termination={'message':'64 first-order updates completed','status':'budget_complete','iterations':64}
 if phash(last_accepted)!=last_validated:evaluate(last_accepted,'final accepted iterate')
 record.update(termination=termination,selected_oracle_call=selected_call,selected_validation_loss=best,training_seconds=train_seconds,validation_seconds=validation_seconds,gradient_diagnostic_seconds=gradient_diagnostic_seconds,gradient_diagnostic_window_gradients=gradient_diagnostic_window_gradients,fit_elapsed_s=time.monotonic()-fit_started,final_parameter_sha256=store('final_params',last_accepted),optimizer_oracle_calls=len(record['history']),accepted_iterations=len(record['accepted']))
 arrays={'reference':reference,'scales':np.asarray(D)}
 native=trajectory(bestparams,reference[0],False);arrays['uncorrected']=native
 record['uncorrected_heldout']=score(native,reference);record['physics']={'uncorrected':physics(native),'reference':physics(reference)}
 for name,pp in [('selected',bestparams),('final',last_accepted)]:
  xs=trajectory(pp,reference[0]);arrays[name]=xs;record[name+'_heldout']=score(xs,reference);record['physics'][name]=physics(xs)
 np.savez_compressed(out/'fields.npz',**arrays)
 record['completed']=True;save()
except Exception as exc:
 record['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};save();raise
