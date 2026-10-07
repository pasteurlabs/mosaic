import hashlib,importlib.util,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
apis={}
for name in ['baseline','candidate']:
 spec=importlib.util.spec_from_file_location(name,'/validation/'+name+'.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);apis[name]=api
jax.config.update('jax_enable_x64',False)
for api in apis.values():api.math.set_global_precision(32)
bc={a+'_'+s:{'type':'no_slip'} for a in 'xyz' for s in ['lo','hi']}
rows=[];arrays={}
for ndim,n in [(2,24),(3,12)]:
 shape=(n,n,1) if ndim==2 else(n,n,n)
 x=(jnp.arange(n,dtype=jnp.float32)+.5)/n
 xx,yy=jnp.meshgrid(x,x,indexing='ij')
 v=jnp.zeros((*shape,ndim),dtype=jnp.float32).at[...,0].set(.2*jnp.sin(jnp.pi*xx[:,:,None])**2*jnp.sin(2*jnp.pi*yy[:,:,None])).at[...,1].set(-.2*jnp.sin(2*jnp.pi*xx[:,:,None])*jnp.sin(jnp.pi*yy[:,:,None])**2)
 if ndim==3:v=v*jnp.sin(jnp.pi*x[None,None,:,None])**2
 # Shared projected native initial state; same arrays used by every comparison.
 initial,_,state=apis['candidate'].phiflow_fwd(v,.01,.001,1,1.,bc,return_state=True)
 arrays[f'initial_{ndim}d']=np.asarray(initial)
 outputs={}
 for name in ['baseline','candidate']:
  for dt in [.01,.005,.0025,.00125]:
   api=apis[name];started=time.perf_counter();row={'variant':name,'ndim':ndim,'n':n,'dt':dt,'duration':2.,'adapter_sha256':hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()}
   try:
    f=jax.jit(lambda:api.phiflow_fwd(initial,.01,dt,round(2/dt),1.,bc,state=state,return_state=True))
    result,_,native=f();jax.block_until_ready(result)
    row.update(finite=bool(jnp.isfinite(result).all()),energy=float(.5*jnp.mean(jnp.sum(result*result,axis=-1))),wall_time_s=time.perf_counter()-started)
    outputs[name,dt]=np.asarray(result);arrays[f'{name}_{ndim}d_{dt}']=np.asarray(result)
   except Exception as exc:row['error']=repr(exc)
   rows.append(row);print(row,flush=True)
 reference=outputs.get(('candidate',.00125))
 if reference is not None:
  for row in rows:
   if row['ndim']==ndim and (row['variant'],row['dt']) in outputs:
    actual=outputs[row['variant'],row['dt']];row['relative_to_fine_candidate']=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
Path('/validation/closed-walls.json').write_text(json.dumps({'rows':rows,'device':str(jax.devices()),'scope':'same projected initial state; closed no-slip walls, decay T2, nu .01'},indent=2))
np.savez_compressed('/validation/closed-walls.npz',**arrays)
