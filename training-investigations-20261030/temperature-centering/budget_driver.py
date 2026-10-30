"""Single predeclared H8/256-step physics-gradient pilot; external only."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gpu_adapter import GPUAdapter
from closed_mac import project_velocity_jax,physical_metrics
from gradient_filters import physical_h1
from mosaic.benchmarks.problems.buoyancy_grid.corrector import apply_corrector,init_corrector,normalized_loss
from mosaic.benchmarks.problems.buoyancy_grid.training import balanced_windows,directional_fd

p=argparse.ArgumentParser();p.add_argument('--solver',required=True);p.add_argument('--url');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);arm=os.environ['PILOT_ARM'];model_seed=int(os.environ['MODEL_SEED']);assert arm in ['baseline','whole_state_p','whole_state_h1','temperature_centered'];assert a.solver=='jax-cfd'
root=Path(__file__).parent;started=time.monotonic();inventory=json.loads((root/'inventory.json').read_text());item=inventory['solvers']['jax-cfd']
record={'completed':False,'arm':arm,'recipe':{'updates':None,'fit_budget_s':60.,'horizon':8,'lr':.003,'channels':8,'model_seed':model_seed,'window_seed':0,'validation_interval':64,'clip_global_norm':1.,'ell_cells':1.},'scope':'Native RTX research pilot, not CI performance. Both arms use the identical infinite window stream and60s fit budget, including compilation and validation. H1/P are intentionally modified backward directions; original forward objectives and true native AD checks retained.','gradient_placement':'Identity boundary after every corrected physical state, before intermediate loss and recurrence. H1 filters velocity in closed curl space and Neumann heat, preserving heat constant mode.','compute_accounting':'Each update executes candidate value-gradient AND separate trueAD value-gradient for all arms, then Adam; both calls included in update time. Checkpoint clocks include validation and prior diagnostics. Persistent cache may be warm.','history':[],'checkpoints':[],'devices':[str(d) for d in jax.devices()],'inventory_sha256':hashlib.sha256((root/'inventory.json').read_bytes()).hexdigest()}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save():record['elapsed_s']=time.monotonic()-started;(a.output/'outcome.json').write_text(json.dumps(record,indent=2,allow_nan=False))
def store(name,pp):
 leaves=jax.tree.leaves(pp);np.savez(a.output/(name+'.npz'),**{f'leaf_{i}':np.asarray(x) for i,x in enumerate(leaves)});return [digest(x) for x in leaves]
def forward_metrics(xs):
 x=np.asarray(xs,dtype=np.float64);n=x.shape[-2];u=x[...,0];v=x[...,1];theta=x[...,2]
 uf=np.pad(u,((0,0),(1,0),(0,0)));vf=np.pad(v,((0,0),(0,0),(1,0)));div=n*(np.diff(uf,axis=1)+np.diff(vf,axis=2));uc=.5*(uf[:,1:]+uf[:,:-1]);vc=.5*(vf[:,:,1:]+vf[:,:,:-1]);omega=n*(v[:,1:,:-1]-v[:,:-1,:-1]-u[:,:-1,1:]+u[:,:-1,:-1]);mean=theta.mean(axis=(1,2));variance=theta.var(axis=(1,2))
 return {'heat_mean':mean.tolist(),'heat_drift_from_initial':(mean-mean[0]).tolist(),'temperature_min':theta.min(axis=(1,2)).tolist(),'temperature_max':theta.max(axis=(1,2)).tolist(),'temperature_mean_square':np.mean(theta**2,axis=(1,2)).tolist(),'temperature_variance':variance.tolist(),'temperature_variance_increments':np.diff(variance).tolist(),'mac_divergence_rms':np.sqrt(np.mean(div**2,axis=(1,2))).tolist(),'stored_wall_normal_max':np.maximum(np.max(abs(u[:,-1,:]),axis=1),np.max(abs(v[:,:,-1]),axis=1)).tolist(),'kinetic_energy_collocated':(.5*np.mean(uc**2+vc**2,axis=(1,2))).tolist(),'enstrophy_interior_corners':(.5*np.mean(omega**2,axis=(1,2))).tolist(),'advective_cfl':(.005*n*(np.max(abs(u[:,:-1,:]),axis=(1,2))+np.max(abs(v[:,:,:-1]),axis=(1,2)))).tolist()}
try:
 for _,info in item['files'].items():assert sha(info['path'])==info['sha256']
 nativepath=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py');assert sha(nativepath)==next(v for k,v in item['source_sha256'].items() if k.endswith('/tesseract_api.py'))
 data=np.load(item['files'][item['data_file']]['path']);D=jnp.asarray(data['scales']);train=jnp.asarray(data['training']);validation=data['validation'];reference=np.load(item['files']['heldout_fields.npz']['path'])['reference']
 for key,value in item['data_hashes'].items():assert digest(data['training' if key=='train' else key])==value
 record['data_hashes']=item['data_hashes'];record['reference_admission']=item['reference_admission'];record['native_source_sha256']=sha(nativepath)
 params=init_corrector(model_seed,8,False);record['initial_parameter_sha256']=store('initial_params',params)
 adapter=GPUAdapter('jax-cfd',np.asarray(D),item['recipe']['physics']);step=jax.jit(lambda x:adapter.api.apply_jit({**adapter.fixed,'state0':x})['result'])
 def centered(pp,x):
  shifted=x.at[...,2].add(-jnp.mean(x[...,2]));return apply_corrector(pp,shifted,D).at[...,2].set(x[...,2])
 C=centered if arm=='temperature_centered' else lambda pp,x:apply_corrector(pp,x,D)
 @jax.custom_vjp
 def boundary(x):return x
 def fw(x):return x,None
 def bw(_,g):
  if arm=='whole_state_p':g=jnp.concatenate([project_velocity_jax(g[...,:2]),g[...,2,None]],axis=-1)
  elif arm=='whole_state_h1':g=physical_h1(g,1.)
  return (g,)
 boundary.defvjp(fw,bw)
 def window_stream():
  rng=np.random.default_rng(0);streams=[np.random.default_rng(x) for x in np.random.SeedSequence(0).spawn(len(train))];pending=[[] for _ in range(len(train))]
  while True:
   for case in rng.permutation(len(train)).tolist():
    if not pending[case]:pending[case]=streams[case].permutation(train.shape[1]-8).tolist()
    yield case,pending[case].pop()
 preview=window_stream();prefix=[next(preview) for _ in range(256)];assert [x[0] for x in prefix]==item['schedule']['cases'] and [x[1] for x in prefix]==item['schedule']['starts'];stream=window_stream();record['schedule']=[]

 def objective(pp,target,modified):
  x=target[0];value=jnp.asarray(0.,dtype=x.dtype)
  for i in range(8):
   x=C(pp,step(x));x=boundary(x) if modified else x;value=value+normalized_loss(x,target[i+1],D)
  return value/8
 candidate_vg=jax.jit(jax.value_and_grad(lambda pp,target:objective(pp,target,True)));true_vg=jax.jit(jax.value_and_grad(lambda pp,target:objective(pp,target,False)))

 def convergence_gate(point):
  rows=[directional_fd(lambda pp:objective(pp,train[0,:9],False),point,e) for e in [.003,.001,.0003]]
  pairs=[]
  for left,right in zip(rows[:-1],rows[1:]):
   agreement=abs(left['finite_difference']-right['finite_difference'])/max(abs(left['finite_difference']),abs(right['finite_difference']),1e-8)
   pairs.append({'epsilons':[left['epsilon'],right['epsilon']],'slope_relative_disagreement':agreement,'passes':left['relative_error']<.05 and right['relative_error']<.05 and agreement<.05})
  return {'checks':rows,'adjacent_pairs':pairs,'passes':any(r['passes'] for r in pairs)}
 probe=tuple({**layer} for layer in params)
 probe[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),probe[-1]['w'].shape,dtype=jnp.float32)
 record['research_fd_profiles']={'initial':convergence_gate(params),'synthetic_nonzero':convergence_gate(probe)}
 record['original_initial_trueAD_fd']=record['research_fd_profiles']['initial']['checks'][0]
 record['gate_scope']='Research-only convergence-controlled gate; unchanged5% AD/FD and slope tolerances. Same adjacent pair at initial and exact production synthetic nonzero; original fixed-epsilon failures retained. Not public benchmark admission.'
 common=[a['passes'] and b['passes'] for a,b in zip(record['research_fd_profiles']['initial']['adjacent_pairs'],record['research_fd_profiles']['synthetic_nonzero']['adjacent_pairs'])]
 record['research_fd_profiles']['same_pair_passes_both_points']=common
 save()
 assert any(common),'Research convergence gate rejected'

 # Nonzero learned head verifies centering's residual-gradient constraint.
 template=jax.tree.structure(params);blob=np.load(item['files']['selected_params.npz']['path']);testparams=jax.tree.unflatten(template,[jnp.asarray(blob[f'leaf_{i}']) for i in range(len(blob.files))]);testx=step(jnp.asarray(reference[50]));qv=(apply_corrector(testparams,testx,D)[...,:2]-jnp.asarray(reference[51,...,:2]))/(2*64**2*D[:2]**2);qv=qv/jnp.linalg.norm(qv);q=jnp.zeros_like(testx).at[...,:2].set(qv).at[...,2].set(1/64**2)
 _,pb=jax.vjp(lambda x:centered(testparams,x)-x,testx);gr=pb(q)[0];_,pb=jax.vjp(lambda x:centered(testparams,x),testx);gf=pb(q)[0]
 record['centering_control']={'residual_temperature_gradient_sum':float(jnp.sum(gr[...,2])),'full_temperature_gradient_sum':float(jnp.sum(gf[...,2])),'expected_identity_temperature_sum':float(jnp.sum(q[...,2]))}
 record['centering_control']['residual_gradient_norm']=float(jnp.linalg.norm(gr));assert record['centering_control']['residual_gradient_norm']>1e-12
 assert abs(record['centering_control']['residual_temperature_gradient_sum'])<1e-4
 assert abs(record['centering_control']['full_temperature_gradient_sum']-record['centering_control']['expected_identity_temperature_sum'])<1e-4
 CF=jax.jit(C)
 def trajectory(pp,initial,frames=100,corrected=True):
  xs=[np.asarray(initial)];x=jnp.asarray(initial)
  for _ in range(frames):
   x=step(x);x=CF(pp,x) if corrected else x;xs.append(np.asarray(x))
  xs=np.stack(xs);assert np.isfinite(xs).all();return xs
 def score(xs,ref):return float(np.mean(.5*(np.mean(((xs[1:,...,:2]-ref[1:,...,:2])/np.asarray(D[:2]))**2,axis=(1,2,3))+np.mean(((xs[1:,...,2]-ref[1:,...,2])/float(D[2]))**2,axis=(1,2)))))
 best=float('inf');bestparams=params;selected=0;checkpoint_models={};validation_s=0.;train_s=0.
 def evaluate(update,requests):
  global best,bestparams,selected,validation_s
  t=time.monotonic();xs=trajectory(params,validation[0]);v=score(xs,validation);hashes=store(f'checkpoint_{update}',params);duration=time.monotonic()-t;validation_s+=duration
  completion=time.monotonic()-fit_start;eligible=completion<=60.
  row={'update':update,'validation_loss':v,'elapsed_s':time.monotonic()-started,'fit_elapsed_s':completion,'within_fit_budget':eligible,'requested_milestones':requests,'validation_duration_s':duration,'parameter_sha256':hashes};record['checkpoints'].append(row);checkpoint_models[update]=params
  if eligible and v<best:best=v;bestparams=params;selected=update
  save();return duration

 optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.003));optstate=optimizer.init(params)
 @jax.jit
 def update(pp,state,g,trueg,value,truth):
  delta,state=optimizer.update(g,state,pp);new=optax.apply_updates(pp,delta)
  dot=lambda x,y:sum(jnp.vdot(a,b) for a,b in zip(jax.tree.leaves(x),jax.tree.leaves(y),strict=True))
  ng=jnp.sqrt(dot(g,g));nt=jnp.sqrt(dot(trueg,trueg));nd=jnp.sqrt(dot(delta,delta))
  finite=jnp.isfinite(value)&jnp.isfinite(truth)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((g,trueg,new))]))
  return new,state,jnp.stack([value,truth,finite,ng,nt,dot(g,trueg)/jnp.maximum(ng*nt,1e-30),-dot(delta,trueg)/jnp.maximum(nd*nt,1e-30),nd])
 record['common_preparation_seconds']=time.monotonic()-started;record['cache_policy']='JAX persistent compilation cache disabled before process import; common source/reference/native-gate preparation outside fit budget; first whole-fit compilation inside budget.'
 fit_start=time.monotonic();reserve=evaluate(0,[0.]);record['initial_validation_reserve_s']=reserve;milestones=[7.5,15.,22.5,30.,37.5,45.,52.5];next_m=0;i=0
 while time.monotonic()-fit_start<60.-reserve:
  case,start=next(stream);record['schedule'].append({'case':case,'start':start});target=train[case,start:start+9]
  t=time.monotonic();value,g=candidate_vg(params,target);truth,trueg=true_vg(params,target);new,nextstate,diag=update(params,optstate,g,trueg,value,truth);v=np.asarray(diag);duration=time.monotonic()-t;completion=time.monotonic()-fit_start;eligible=completion<=60.
  record['history'].append({'update':i+1,'loss':float(v[0]),'true_loss':float(v[1]),'candidate_gradient_norm':float(v[3]),'true_gradient_norm':float(v[4]),'gradient_cosine_true':float(v[5]),'adam_descent_cosine_true':float(v[6]),'parameter_step_norm':float(v[7]),'update_seconds':duration,'fit_completion_s':completion,'eligible_before_deadline':eligible,'elapsed_s':time.monotonic()-started});train_s+=duration
  assert bool(v[2]),f'Nonfinite update {i}'
  if not eligible:record['late_update_parameter_sha256']=store('late_uncredited_params',new);break
  params,optstate=new,nextstate;i+=1
  if next_m<len(milestones) and completion>=milestones[next_m]:
   requested=[]
   while next_m<len(milestones) and milestones[next_m]<=completion:requested.append(milestones[next_m]);next_m+=1
   evaluate(i,requested)
   while next_m<len(milestones) and milestones[next_m]<=time.monotonic()-fit_start:next_m+=1
 if time.monotonic()-fit_start<60. and record['checkpoints'][-1]['update']!=i:evaluate(i,[60.])
 record.update(selected_update=selected,validation_loss=best,training_seconds=train_s,validation_seconds=validation_s,fit_elapsed_s=time.monotonic()-fit_start,fit_overshoot_s=max(0.,time.monotonic()-fit_start-60.),completed_updates=i,training_and_selection_elapsed_s=time.monotonic()-started,final_parameter_sha256=store('final_params',params),selected_parameter_sha256=store('selected_params',bestparams))
 arrays={'reference':reference};native=trajectory(params,reference[0],corrected=False);arrays['uncorrected']=native;record['forward_physics']={'reference':forward_metrics(reference),'uncorrected':forward_metrics(native)}
 for label,pp in [('selected',bestparams),('final',params)]:
  xs=trajectory(pp,reference[0]);arrays[label]=xs;record[label+'_heldout_loss']=score(xs,reference);record['forward_physics'][label]=forward_metrics(xs)
  if label=='selected':
   symmetry=[]
   for sign in [-1,1]:
    c=sign*.1*float(D[2]);initial=reference[0].copy();initial[...,2]+=c;shifted=trajectory(pp,initial);arrays[f'selected_offset_{sign}']=shifted
    for h in [32,100]:symmetry.append({'offset_physical':c,'horizon':h,'velocity_rms_difference':float(np.sqrt(np.mean((shifted[h,...,:2]-xs[h,...,:2])**2))),'temperature_shift_covariance_rms':float(np.sqrt(np.mean((shifted[h,...,2]-xs[h,...,2]-c)**2)))})
   record['selected_offset_invariance']=symmetry
   x=jnp.asarray(reference[50]);sx=step(x);fx=CF(pp,sx);error=fx-jnp.asarray(reference[51]);q=jnp.zeros((2,64,64,3),jnp.float32);q=q.at[0,...,:2].set(error[...,:2]/(2*64**2*D[:2]**2));q=q.at[1,...,2].set(error[...,2]/(64**2*D[2]**2));gradrows=[]
   for operator,fn,point in [('S',step,x),('C',lambda y:CF(pp,y),sx),('C_after_S',lambda y:CF(pp,step(y)),x)]:
    _,pull=jax.vjp(fn,point)
    for k,field in enumerate(['velocity','temperature']):
     g=np.asarray(pull(q[k])[0]);arrays[f'gradient_{operator}_{field}']=g;gradrows.append({'operator':operator,'field':field,'matched_q_source':'selected local composed component-loss residual','metrics':physical_metrics(g,np.ones(3,np.float32))})
   record['selected_local_physical_gradients']=gradrows

 arrays['selected_pre_correction']=np.stack([np.asarray(step(jnp.asarray(xx))) for xx in arrays['selected'][:-1]])
 offset_rows=[]
 for c_units in [-1.,-.5,-.1,.1,.5,1.]:
  c=c_units*float(D[2]);initial=reference[0].copy();initial[...,2]+=c
  shifted_reference=reference.copy();shifted_reference[...,2]+=c
  shifted=trajectory(bestparams,initial);shifted_native=trajectory(bestparams,initial,corrected=False)
  offset_rows.append({'offset_scale_units':c_units,'offset_physical':c,'selected_loss_to_symmetry_target':score(shifted,shifted_reference),'uncorrected_loss_to_symmetry_target':score(shifted_native,shifted_reference),'selected_velocity_rms_symmetry_defect':float(np.sqrt(np.mean((shifted[1:,...,:2]-arrays['selected'][1:,...,:2])**2))),'selected_temperature_rms_symmetry_defect':float(np.sqrt(np.mean((shifted[1:,...,2]-arrays['selected'][1:,...,2]-c)**2))),'native_velocity_rms_symmetry_defect':float(np.sqrt(np.mean((shifted_native[1:,...,:2]-native[1:,...,:2])**2))),'native_temperature_rms_symmetry_defect':float(np.sqrt(np.mean((shifted_native[1:,...,2]-native[1:,...,2]-c)**2)))})
 record['offset_heldout']=offset_rows
 record['offset_reference_note']='Physical uniform-temperature symmetry transforms admitted fine reference; not freshly simulated shifted fine reference. Native coarse symmetry defects measured separately.'
 arrays['scales']=np.asarray(D)
 np.savez_compressed(a.output/'fields.npz',**arrays)
 record['wall_frontier']={str(b):min((r for r in record['checkpoints'] if r['within_fit_budget'] and r['fit_elapsed_s']<=b),key=lambda r:(r['validation_loss'],r['update']),default=None) for b in [7.5,15.,22.5,30.,37.5,45.,52.5,60.]}
 record['completed']=True;save()
except Exception as exc:record['failure']=f'{type(exc).__name__}: {exc}';record['traceback']=traceback.format_exc();save();raise
