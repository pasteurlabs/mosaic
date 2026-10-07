"""Allocated-GPU-only assimilation and local collision conditioning diagnostics."""
import importlib.util,json,sys,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
spec=importlib.util.spec_from_file_location('audit_xlb','/tesseract/tesseract_api.py')
api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
dataset=Path('/reference/dataset.npz')
with np.load(dataset,allow_pickle=False) as z:v=jnp.asarray(z['train'][0,1])
force=jnp.zeros_like(v).at[:,:,0,0].set(jnp.sin(6*jnp.arange(64)[None,:]*(2*jnp.pi/64)))
state=None;rows=[];scale=.01/15/(2*np.pi/64)
for frame in range(5):
 result,_,state=api.xlb_fwd(v+.02*force,viscosity=.001,dt=.01,steps=4,domain_extent=2*np.pi,state=state,return_state=True,_sub_k=15,_use_f64=True,_collision_kind_override='kbc')
 v=result+.02*force
 for f64 in [False,True]:
  ops=api._OPS[(2,f64,'kbc')];dtype=ops['fdtype'];f,canonical=api._state_to_internal(state.astype(dtype),ndim=2,spatial=(64,64))
  rho,u=ops['macro'](f);eq=ops['eq'];e=eq(rho,u)
  original=f+e-e;reassociated=f+(e-e)
  delta=jnp.asarray(np.random.default_rng(116).normal(scale=.001,size=u.shape),dtype=dtype)*scale
  reconciled=f+(eq(rho,u+delta)-eq(rho,u));r2,u2=ops['macro'](reconciled)
  rows.append(dict(frame=frame+1,precision=str(dtype),no_op_max_error=float(jnp.max(jnp.abs(original-f))),no_op_serialized_equal=bool(jnp.array_equal(original.astype(jnp.float32),f.astype(jnp.float32))),reassociated_exact=bool(jnp.array_equal(reassociated,f)),density_change=float(jnp.max(jnp.abs(r2-rho))),momentum_correction_error=float(jnp.max(jnp.abs(u2-u-delta))),min_population=float(jnp.min(f)),min_equilibrium=float(jnp.min(e))))
 # Collision is local; compare full9x9 Jacobian at a fixed cell without RPCcasts.
 ops=api._OPS[(2,True,'kbc')];f=state[:9,:,:,0].astype(jnp.float64);point=f[:,17,23];omega=1/(3*.001*(.01/15)/(2*np.pi/64)**2+.5)
 for kind in ['kbc','bgk']:
  op=api._OPS[(2,True,kind)]
  def collide(p):
   q=p[:,None,None];rho,u=op['macro'](q);return op['bgk'](q,op['eq'](rho,u),rho,u,omega)[:,0,0]
  jac=jax.jacfwd(collide)(point);direction=jnp.asarray(np.random.default_rng(117).normal(size=9));direction=direction/jnp.linalg.norm(direction)
  ad=jac@direction;checks=[]
  for eps in [1e-3,1e-5,1e-7,1e-9,1e-11]:
   fd=(collide(point+eps*direction)-collide(point-eps*direction))/(2*eps)
   checks.append(dict(epsilon=eps,relative_error=float(jnp.linalg.norm(ad-fd)/(jnp.linalg.norm(ad)+jnp.linalg.norm(fd))),fd_norm=float(jnp.linalg.norm(fd))))
  rows.append(dict(frame=frame+1,collision=kind,local_jacobian_spectral_norm=float(jnp.linalg.svd(jac,compute_uv=False)[0]),local_jacobian_max_eigenvalue_abs=float(jnp.max(jnp.abs(jnp.linalg.eigvals(jac)))),ad_norm=float(jnp.linalg.norm(ad)),checks=checks))
 np.savez('/audit/native-frame'+str(frame+1)+'.npz',state=np.asarray(state),result=np.asarray(result))
 Path('/audit/outcome.json').write_text(json.dumps(dict(dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),state_origin='regenerated from exact forced train[0,1] with15substeps; not prior saved failing native state',rows=rows),indent=2,allow_nan=False))
