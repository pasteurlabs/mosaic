"""Small shared-subspace Jacobian audit; does not estimate full-space rank."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import surrogate_model as fno
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
jax.config.update('jax_enable_x64',True)
spec=importlib.util.spec_from_file_location('conditioning_teacher',root/'source/mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py')
api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
def teacher(x):
 return api.xlb_fwd(x.astype(jnp.float64),viscosity=.01,dt=.02,steps=100,domain_extent=2*np.pi,_use_f64=True,_sub_k=1,_collision_kind_override='kbc')[0]
with np.load('/teacher-data/recovery_3d_xlb_trajectories_16k.split.npz') as data:
 idx=int(np.flatnonzero((data['split']==1)&(data['amplitudes']>.4))[0])
x=np.asarray(np.load('/teacher-data/recovery_3d_xlb_trajectories_16k.npy',mmap_mode='r')[idx,0])
k=np.fft.fftfreq(16,d=1/16);kz=np.fft.rfftfreq(16,d=1/16)
kx,ky,kzz=np.meshgrid(k,k,kz,indexing='ij');mask=(kx*kx+ky*ky+kzz*kzz<=16)[...,None]
rng=np.random.default_rng(20261006)
raw=rng.normal(size=(32,16,16,16,3)).astype(np.float32)
raw=np.fft.irfftn(np.fft.rfftn(raw,axes=(1,2,3))*mask,s=(16,16,16),axes=(1,2,3))
raw=np.asarray(fno.helmholtz_project(jnp.asarray(raw,dtype=jnp.float32)))
basis=np.linalg.qr(raw.reshape(32,-1).T.astype(np.float64))[0].T.reshape(32,16,16,16,3).astype(np.float32)
report={'validation_index':idx,'basis_seed':20261006,'basis_dimension':32,'input_dimension':12288,'restriction':'32 orthonormal projected random directions with |k| <= 4; not the full Fourier subspace or full Jacobian','models':{}}
references={}
models=['XLB','baseline','field','vjp001','vjp01','vjp1','secant01','linear_vjp01','path_field','path_vjp01']
for name in models:
 if name=='XLB':forward=teacher
 else:
  path=root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/weights.npz' if name=='baseline' else root/'results'/f'{name}.npz'
  with np.load(path) as d:c={k:d[k] for k in d.files}
  params={k:jnp.asarray(c[k]) for k in fno.init_params(width=32,modes=6,layers=6,seed=0)}
  if 'w_linear' in c:params['w_linear']=jnp.asarray(c['w_linear'])
  def forward(v):
   return fno.rollout(params,v.astype(jnp.float32)[None],steps=20,input_scale=jnp.asarray(c['input_scale']),correction_scale=jnp.asarray(c['correction_scale']),modes=6,layers=6)[0,-1]
 action=jax.jit(lambda state,d:jax.jvp(forward,(state,),(d,))[1])
 rows=[]
 for scale in [0.,.1,1.]:
  matrix=np.stack([np.asarray(action(jnp.asarray(x*scale),jnp.asarray(d))).ravel() for d in basis],axis=1).astype(np.float64)
  singular=np.linalg.svd(matrix,compute_uv=False)
  if name=='XLB':references[scale]=matrix
  ref=references[scale]
  tolerance=float(max(matrix.shape)*np.finfo(np.float32).eps*singular[0])
  row={'scale':scale,'singular_values':singular.tolist(),'condition':float(singular[0]/singular[-1]),'rank_float32':int(np.sum(singular>tolerance)),'rank_tolerance':tolerance,'relative_frobenius_error':float(np.linalg.norm(matrix-ref)/np.linalg.norm(ref))}
  rows.append(row)
 report['models'][name]=rows
 (root/'results/conditioning.json').write_text(json.dumps(report,indent=2))
 print(name,[(r['scale'],r['condition'],r['relative_frobenius_error']) for r in rows],flush=True)
 jax.clear_caches()
