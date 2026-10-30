"""Training-only fixed readout block sensitivity calibration; external experiment."""
from pathlib import Path
import hashlib,json,time,traceback
import jax
import jax.numpy as jnp
import numpy as np
from gpu_adapter import GPUAdapter
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector,apply_corrector,normalized_loss
root=Path(__file__).parent
out=root/'results';out.mkdir(exist_ok=True)
begun=time.monotonic()
record={'completed':False,'method':'fixed readout block-sensitivity reparameterization, not HIG/SIP','model_seed':0,'directions_per_head':6,'random_direction_seed_base':7900,'fixed_windows':{'cases':[0,1,2,3],'starts':[0,24,48,72],'horizon':8},'rule':'head_scales=[1,sigma_velocity_readout/sigma_temperature_readout]; sigma is RMS of full normalized H8 recurrent output directional response over six unit readout weight directions and four training windows; factor selected at initial zero readout only. Synthetic nonzero point is diagnostic only.','sources':{},'devices':[str(d) for d in jax.devices()]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save():
 record['elapsed_s']=time.monotonic()-begun
 (out/'calibration.json').write_text(json.dumps(record,indent=2,allow_nan=False))
def physical_correction(pp,x):
 y=apply_corrector(pp,x,D)
 delta=np.asarray(y-x)
 return {'normalized_velocity_rms':float(np.sqrt(np.mean((delta[...,:2]/np.asarray(D[:2]))**2))),'normalized_temperature_rms':float(np.sqrt(np.mean((delta[...,2]/float(D[2]))**2))),'temperature_mean_change':float(delta[...,2].mean()),'temperature_min_after':float(np.asarray(y[...,2]).min()),'temperature_max_after':float(np.asarray(y[...,2]).max())}
try:
 item=json.loads((root/'inventory.json').read_text())['solvers']['jax-cfd']
 for name,info in item['files'].items():assert sha(info['path'])==info['sha256'],name
 z=np.load(item['files'][item['data_file']]['path']);D=jnp.asarray(z['scales']);train=jnp.asarray(z['training'])
 record['data_sha256']=item['data_hashes']
 native=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py')
 assert sha(native)==next(v for k,v in item['source_sha256'].items() if k.endswith('/tesseract_api.py'))
 record['sources']={str(native):sha(native),str(root/'driver.py'):sha(root/'driver.py'),str(root/'gpu_adapter.py'):sha(root/'gpu_adapter.py')}
 params=init_corrector(0,8,True)
 adapter=GPUAdapter('jax-cfd',np.asarray(D),item['recipe']['physics'])
 step=jax.jit(lambda x:adapter.api.apply_jit({**adapter.fixed,'state0':x})['result'])
 targets=[train[i,start:start+9] for i,start in enumerate([0,24,48,72])]
 def residual(pp,target):
  x=target[0];values=[]
  for i in range(8):
   x=apply_corrector(pp,step(x),D)
   values.append((x-target[i+1])/D)
  rr=jnp.stack(values)
  weight=jnp.asarray([1/jnp.sqrt(2.),1/jnp.sqrt(2.),1.],dtype=rr.dtype)
  return rr*weight/jnp.sqrt(jnp.asarray(4*8*64**2,dtype=rr.dtype))
 residual_jit=jax.jit(residual)
 jvp_jit=jax.jit(lambda pp,dd,target:jax.jvp(lambda p:residual(p,target),(pp,),(dd,)))
 def full_loss(pp):
  total=0.
  for target in targets:
   x=target[0]
   for i in range(8):
    x=apply_corrector(pp,step(x),D);total+=normalized_loss(x,target[i+1],D)/(4*8)
  return total
 fd_rows=[];responses=[];norm_checks=[]
 for point_name,pp in [('initial',params),('synthetic_nonzero',tuple({**layer} for layer in params))]:
  if point_name=='synthetic_nonzero':
   pp[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),pp[-1]['w'].shape,dtype=jnp.float32)
  loss=float(full_loss(pp));normloss=sum(float(.5*jnp.sum(residual_jit(pp,t)**2)) for t in targets)
  assert abs(loss-normloss)<=max(1e-10,abs(loss)*2e-6),(loss,normloss)
  norm_checks.append({'point':point_name,'public_normalized_loss':loss,'half_squared_residual_norm':normloss,'relative_disagreement':abs(loss-normloss)/max(abs(loss),1e-30)})
  for head,label in enumerate(['velocity','temperature']):
   for seed in range(6):
    dd=jax.tree.map(jnp.zeros_like,pp)
    direction=jax.random.normal(jax.random.PRNGKey(7900+seed),pp[-1]['w'][...,head].shape,dtype=jnp.float32);direction=direction/jnp.linalg.norm(direction)
    dd[-1]['w']=dd[-1]['w'].at[...,head].set(direction)
    rr=[];jj=[];callstart=time.monotonic()
    for t in targets:
     r,j=jvp_jit(pp,dd,t);rr.append(np.asarray(r));jj.append(np.asarray(j))
    rr=np.stack(rr).astype(np.float64);jj=np.stack(jj).astype(np.float64)
    assert np.isfinite(rr).all() and np.isfinite(jj).all()
    responses.append({'point':point_name,'head':label,'direction_seed':7900+seed,'direction_norm':float(jnp.linalg.norm(direction)),'jvp_norm':float(np.linalg.norm(jj)),'velocity_output_jvp_norm':float(np.linalg.norm(jj[...,:2])),'temperature_output_jvp_norm':float(np.linalg.norm(jj[...,2])),'seconds_including_compile_if_first':time.monotonic()-callstart})
    if seed==0:
     errs=[]
     for eps in [.003,.001,.0003]:
      plus=jax.tree.map(lambda x,d:x+eps*d,pp,dd);minus=jax.tree.map(lambda x,d:x-eps*d,pp,dd)
      rplus=np.stack([np.asarray(residual_jit(plus,t),dtype=np.float64) for t in targets])
      rminus=np.stack([np.asarray(residual_jit(minus,t),dtype=np.float64) for t in targets])
      fd=(rplus-rminus)/(2*eps)
      err=float(np.linalg.norm(fd-jj)/max(np.linalg.norm(jj),np.linalg.norm(fd),1e-30))
      errs.append({'epsilon':eps,'relative_vector_error':err,'fd_norm':float(np.linalg.norm(fd)),'jvp_norm':float(np.linalg.norm(jj))})
     fd_rows.append({'point':point_name,'head':label,'direction_seed':7900,'checks':errs,'passes':sum(e['relative_vector_error']<.05 for e in errs)>=2})
 record['response_rows']=responses;record['finite_difference_rows']=fd_rows;record['normalized_metric_checks']=norm_checks
 assert all(r['passes'] for r in fd_rows),'Directional vector JVP/FD controls rejected'
 point_summaries={}
 for point in ['initial','synthetic_nonzero']:
  ss={}
  for head in ['velocity','temperature']:
   vals=np.asarray([r['jvp_norm'] for r in responses if r['point']==point and r['head']==head])
   ss[head]={'rms_response':float(np.sqrt(np.mean(vals**2))),'minimum':float(vals.min()),'maximum':float(vals.max()),'std_over_mean':float(vals.std()/vals.mean())}
  ss['thermal_factor']=ss['velocity']['rms_response']/ss['temperature']['rms_response']
  point_summaries[point]=ss
 record['point_summaries']=point_summaries
 factor=point_summaries['initial']['thermal_factor']
 assert np.isfinite(factor) and factor>0
 # Verify exact zero-readout identity, parameter-map equivalence, and transformed JVP scaling.
 from headscale import scaled_parameters,apply_scaled_corrector
 hs=[1.,factor]
 record['head_scales']=hs
 zeros=scaled_parameters(params,hs)
 for t in targets:
  x=step(t[0])
  assert np.array_equal(np.asarray(apply_scaled_corrector(params,x,D,hs)),np.asarray(x))
 point=tuple({**layer} for layer in params);point[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),point[-1]['w'].shape,dtype=jnp.float32)
 control_x=step(targets[0][0]);explicit=apply_corrector(scaled_parameters(point,hs),control_x,D);helper=apply_scaled_corrector(point,control_x,D,hs)
 assert np.array_equal(np.asarray(explicit),np.asarray(helper))
 record['controls']={'zero_readout_exact_identity':True,'helper_equals_explicit_parameter_map':True,'normalized_residual_matches_public_objective':True}
 record['synthetic_direct_correction_physics']={'unscaled':physical_correction(point,control_x),'scaled':physical_correction(scaled_parameters(point,hs),control_x)}
 record['cost_scope']='Calibration measured separately; its full compile and directional-evaluation cost must be charged to scaled methods if judging total runtime. Training-only cached references shared; no CI container claim.'
 record['completed']=True;save()
 summary={'completed':True,'head_scales':hs,'calibration_file':str(out/'calibration.json'),'calibration_sha256':sha(out/'calibration.json'),'selection_point':'initial_zero_readout','reference':'training-only full-H8 block JVP RMS, not HIG or full Jacobian whitening','calibration_elapsed_s':record['elapsed_s'],'helper_file':str(root/'headscale.py'),'helper_sha256':sha(root/'headscale.py'),'scope':'Experimental fixed head reparameterization only; no constraint preserving total heat or temperature bounds'}
 (root/'headscale.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
except Exception as exc:
 record['failure']=str(exc);record['traceback']=traceback.format_exc();save();raise
