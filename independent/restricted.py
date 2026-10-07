"""GPU collision derivatives restricted to exactly conserved-moment tangent space."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
spec=importlib.util.spec_from_file_location('audit_xlb','/tesseract/tesseract_api.py')
api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
ops=api._OPS[(2,True,'kbc')]
c=np.asarray(ops['C']);c=c if c.shape==(2,9) else c.T
moments=np.concatenate([np.ones((1,9)),c],axis=0)
_,_,vh=np.linalg.svd(moments,full_matrices=True);basis=vh[3:].T
omega=1/(3*.001*(.01/15)/(2*np.pi/64)**2+.5)
rows=[];arrays={'kinetic_basis':basis,'moments':moments}
for frame in range(1,6):
 with np.load('/audit/native-frame'+str(frame)+'.npz') as z:point=jnp.asarray(z['state'][:9,17,23,0],dtype=jnp.float64)
 for kind in ['kbc','bgk']:
  op=api._OPS[(2,True,kind)]
  def collide(p):
   q=p[:,None,None];rho,u=op['macro'](q);return op['bgk'](q,op['eq'](rho,u),rho,u,omega)[:,0,0]
  jac=np.asarray(jax.jacfwd(collide)(point));restricted=basis.T@jac@basis
  direction=basis@np.random.default_rng(118).normal(size=6);direction/=np.linalg.norm(direction)
  ad=jac@direction;checks=[]
  for eps in [1e-3,1e-5,1e-7,1e-9,1e-11]:
   fd=np.asarray((collide(point+eps*direction)-collide(point-eps*direction))/(2*eps))
   checks.append(dict(epsilon=eps,relative_error=float(np.linalg.norm(ad-fd)/(np.linalg.norm(ad)+np.linalg.norm(fd))),fd_norm=float(np.linalg.norm(fd))))
  rows.append(dict(frame=frame,collision=kind,kinetic_spectral_norm=float(np.linalg.svd(restricted,compute_uv=False)[0]),kinetic_eigenvalue_maxabs=float(np.max(np.abs(np.linalg.eigvals(restricted)))),moment_conservation_jacobian_error=float(np.max(np.abs(moments@jac-moments))),basis_moment_error=float(np.max(np.abs(moments@basis))),checks=checks))
  arrays[f'{kind}_frame{frame}_jacobian']=jac;arrays[f'frame{frame}_point']=np.asarray(point)
with np.load('/reference/dataset.npz',allow_pickle=False) as z:np.save('/audit/initial.npy',z['train'][0,1])
np.savez('/audit/local-jacobians.npz',**arrays)
Path('/audit/restricted-outcome.json').write_text(json.dumps({'rows':rows,'omega':omega,'scope':'Local collision Jacobian at fixedcell17,23; not fullmap stability or formal modelFD admission'},indent=2,allow_nan=False))
