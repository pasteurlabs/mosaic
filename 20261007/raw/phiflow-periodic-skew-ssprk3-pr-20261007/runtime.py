import importlib.util,time,json,hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
rows=[]
for variant in ['baseline','candidate']:
 spec=importlib.util.spec_from_file_location('api_'+variant,'/validation/'+variant+'.py');api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 jax.config.update('jax_enable_x64',False);api.math.set_global_precision(32)
 for n,dim in [(64,2),(192,2),(32,3),(64,3)]:
  row={'variant':variant,'n':n,'ndim':dim,'adapter_sha256':hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()}
  try:
   shape=(n,n,1,dim) if dim==2 else (n,n,n,dim)
   y=jnp.arange(n,dtype=jnp.float32)*2*jnp.pi/n
   v=jnp.zeros(shape,dtype=jnp.float32).at[...,0].set(.1*jnp.sin(6*y)[None,:,None])
   bc={a+'_'+s:{'type':'periodic'} for a in 'xyz' for s in ('lo','hi')}
   dt=.01*64/n if dim==2 else .01
   run=jax.jit(lambda v:api.phiflow_fwd(v,.001,dt,4,2*jnp.pi,bc,return_state=True))
   t=time.perf_counter();result=run(v);jax.block_until_ready(result);row['first_four_steps_s']=time.perf_counter()-t
   t=time.perf_counter()
   for _ in range(30):
    result=run(v);jax.block_until_ready(result)
   row['warmed_four_steps_s']=(time.perf_counter()-t)/30
  except Exception as e:row['error']=repr(e)
  rows.append(row);print(row,flush=True)
Path('/validation/runtime.json').write_text(json.dumps({'device':str(jax.devices()),'rows':rows},indent=2))
