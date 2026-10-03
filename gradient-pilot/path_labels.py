"""Cache teacher labels at baseline recovery iterates, split by source IC."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import surrogate_model as fno
from train import _sha256
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
jax.config.update('jax_enable_x64',True)
spec=importlib.util.spec_from_file_location('path_teacher',root/'source/mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py')
api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
@jax.jit
def teacher(x):
    return api.xlb_fwd(x.astype(jnp.float64),viscosity=.01,dt=.02,steps=100,domain_extent=2*np.pi,_use_f64=True,_sub_k=1,_collision_kind_override='kbc')[0]
weights=root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/weights.npz'
with np.load(weights) as d:c={k:d[k] for k in d.files}
params={k:jnp.asarray(c[k]) for k in fno.init_params(width=32,modes=6,layers=6,seed=0)}
@jax.jit
def forward(x):
    return fno.rollout(params,x.astype(jnp.float32)[None],steps=20,input_scale=jnp.asarray(c['input_scale']),correction_scale=jnp.asarray(c['correction_scale']),modes=6,layers=6)[0,-1]
@jax.jit
def value_grad(z,target):
    return jax.value_and_grad(lambda zz:jnp.mean((forward(zz.reshape(16,16,16,3))-target)**2))(z)
@jax.jit
def label(x,q):
    y,back=jax.vjp(teacher,x.astype(jnp.float64));return y,back(q.astype(y.dtype))[0]
base=Path('/teacher-data/recovery_3d_xlb_trajectories_16k.npy')
with np.load(base.with_suffix('.split.npz')) as d:
    split=d['split'];amplitudes=d['amplitudes']
indices=np.concatenate([np.flatnonzero((split==part)&(amplitudes>.4))[:count] for part,count in [(0,8),(1,2)]])
trajectories=np.load(base,mmap_mode='r'); rng=np.random.default_rng(20261005)
records=[]
for idx in indices:
    truth=np.asarray(trajectories[idx,0],dtype=np.float32)
    target=np.asarray(teacher(jnp.asarray(truth)),dtype=np.float32)
    states=[np.zeros_like(truth)]; n=[0]
    def objective(z):
        value,grad=value_grad(jnp.asarray(z),jnp.asarray(target));return float(value),np.asarray(grad,dtype=np.float64)
    def callback(z):
        n[0]+=1
        if n[0] in [1,2,5,10,20,50,100]:states.append(z.reshape(truth.shape).astype(np.float32))
    result=minimize(objective,np.zeros(truth.size),jac=True,method='L-BFGS-B',callback=callback,
        options=dict(maxiter=100,maxls=30,gtol=1e-12,ftol=1e-15,maxcor=10))
    states.append(result.x.reshape(truth.shape).astype(np.float32))
    for i,x in enumerate(states):
        q=rng.normal(size=x.shape).astype(np.float32) if i%2==0 else np.asarray(teacher(x))-target
        if np.linalg.norm(q)<1e-12:q=rng.normal(size=x.shape).astype(np.float32)
        q=(q/np.linalg.norm(q)).astype(np.float32)
        y,vjp=label(x,q)
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(vjp)):raise RuntimeError('nonfinite path label')
        records.append((x,np.asarray(y,dtype=np.float32),q,np.asarray(vjp,dtype=np.float32),split[idx],idx))
    print('LABELED',int(idx),'split',int(split[idx]),'states',len(states),flush=True)
output=root/'results/path-labels.npz'
np.savez(output,**{key:np.asarray([r[i] for r in records]) for i,key in enumerate(['initial','target','cotangent','teacher_vjp','split','indices'])})
output.with_suffix('.json').write_text(json.dumps(dict(labels_sha256=_sha256(output),initial_weights_sha256=_sha256(weights),source_indices=indices.tolist(),train_cases=8,validation_cases=2,cotangents='alternating random and teacher recovery residual',states=len(records)),indent=2))
