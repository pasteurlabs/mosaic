"""Offline validation pilot. Final registered benchmarks run separately."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import surrogate_model as fno

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
root=a.root
jax.config.update('jax_enable_x64',True)
teacher_path=root/'source/mosaic/tesseracts/navier-stokes-grid/xlb/tesseract_api.py'
spec=importlib.util.spec_from_file_location('pilot_teacher',teacher_path)
api=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=api
spec.loader.exec_module(api)

def teacher(x):
    return api.xlb_fwd(x.astype(jnp.float64),viscosity=.01,dt=.02,steps=100,
        domain_extent=2*np.pi,_use_f64=True,_sub_k=1,_collision_kind_override='kbc')[0]

teacher=jax.jit(teacher)
with np.load('/teacher-data/recovery_3d_xlb_trajectories_16k.split.npz') as d:
    val_idx=np.flatnonzero((d['split']==1)&(d['amplitudes']>.4))[:3]
trajectories=np.load('/teacher-data/recovery_3d_xlb_trajectories_16k.npy',mmap_mode='r')
initials=np.asarray(trajectories[val_idx,0],dtype=np.float32)
true_targets=[np.asarray(teacher(jnp.asarray(x))) for x in initials]
rng=np.random.default_rng(20261004)
qs=rng.normal(size=initials.shape).astype(np.float32)
qs/=np.linalg.norm(qs.reshape(3,-1),axis=1)[:,None,None,None,None]

@jax.jit
def teacher_vjp(x,q):
    y,back=jax.vjp(teacher,x.astype(jnp.float64))
    return back(q.astype(y.dtype))[0]

def metrics(x,y):
    x=np.asarray(x,dtype=np.float64).ravel(); y=np.asarray(y,dtype=np.float64).ravel()
    return dict(relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)),
                cosine=float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-20)))

teacher_derivatives=[np.asarray(teacher_vjp(x,q)) for x,q in zip(initials,qs)]

def load(path):
    with np.load(path) as d:
        c={k:d[k] for k in d.files}
    keys=fno.init_params(width=int(c['width']),modes=int(c['modes']),layers=int(c['layers']),seed=0)
    params={k:jnp.asarray(c[k]) for k in keys}
    if 'w_linear' in c: params['w_linear']=jnp.asarray(c['w_linear'])
    def forward(x):
        return fno.rollout(params,x.astype(jnp.float32)[None],steps=20,
            input_scale=jnp.asarray(c['input_scale']),correction_scale=jnp.asarray(c['correction_scale']),
            modes=int(c['modes']),layers=int(c['layers']))[0,-1]
    return jax.jit(forward)

k=jnp.fft.fftfreq(16,d=1/16)
kz=jnp.fft.rfftfreq(16,d=1/16)
kx,ky,kzz=jnp.meshgrid(k,k,kz,indexing='ij')
mask=(kx*kx+ky*ky+kzz*kzz<=16)[...,None]

def project(x):
    projected=fno.helmholtz_project(x.astype(jnp.float32))
    return jnp.fft.irfftn(jnp.fft.rfftn(projected,axes=(0,1,2))*mask,s=(16,16,16),axes=(0,1,2)).astype(x.dtype)

def recovery(forward,truth,target,reduced=False):
    shape=truth.shape
    transform=project if reduced else lambda x:x
    @jax.jit
    def objective(z):
        predicted=forward(transform(z.reshape(shape)))
        return jnp.mean((predicted-target.astype(predicted.dtype))**2)
    value_grad=jax.jit(jax.value_and_grad(objective))
    def callback(z):
        value,grad=value_grad(jnp.asarray(z))
        return float(value),np.asarray(grad,dtype=np.float64)
    started=time.perf_counter()
    result=minimize(callback,np.zeros(truth.size,dtype=np.float64),jac=True,method='L-BFGS-B',
        options=dict(maxiter=100,maxls=30,gtol=1e-12,ftol=1e-15,maxcor=10))
    recovered=np.asarray(transform(jnp.asarray(result.x.reshape(shape))))
    return dict(ic_relative_l2=metrics(recovered,truth)['relative_l2'],residual_mse=float(result.fun),
        iterations=int(result.nit),evaluations=int(result.nfev),success=bool(result.success),
        message=str(result.message),seconds=time.perf_counter()-started)

paths={'baseline':root/'source/mosaic/tesseracts/navier-stokes-grid/xlb-3d-surrogate/weights.npz'}
for arm in ['field','vjp001','vjp01','vjp1','secant01','linear_vjp01','path_field','path_vjp01']:
    paths[arm]=root/'results'/f'{arm}.npz'
report={'validation_indices':val_idx.tolist(),'seed':20261004,
        'protocol':'native JAX + scipy L-BFGS-B; validation pilot, not registered benchmark', 'models':{}}
for name,path in [('XLB',None),*paths.items()]:
    print('EVALUATE',name,flush=True)
    forward=teacher if path is None else load(path)
    @jax.jit
    def derivative(x,q):
        y,back=jax.vjp(forward,x)
        return y,back(q.astype(y.dtype))[0]
    rows=[]
    for index,(initial,target,q,tvjp) in enumerate(zip(initials,true_targets,qs,teacher_derivatives)):
        prediction,vjp=derivative(jnp.asarray(initial),jnp.asarray(q))
        row={'case':index,'forward':metrics(prediction,target),'vjp':metrics(vjp,tvjp),'path':[]}
        for scale in [0.,.05,.1,.5,1.]:
            x=jnp.asarray(initial*scale)
            y,back=jax.vjp(forward,x)
            common_q=np.asarray(teacher(x))-target
            row['path'].append(dict(scale=scale,**metrics(back(jnp.asarray(common_q,dtype=y.dtype))[0],teacher_vjp(x,common_q))))
        row['self_recovery']=recovery(forward,initial,np.asarray(prediction))
        row['xlb_recovery']=recovery(forward,initial,target)
        row['fourier_xlb_recovery']=recovery(forward,initial,target,reduced=True)
        rows.append(row)
        print(name,index,'vjp',row['vjp'],'recovery',row['xlb_recovery']['ic_relative_l2'],flush=True)
    report['models'][name]=rows
    (root/'results/validation.json').write_text(json.dumps(report,indent=2))
    del forward
    jax.clear_caches()
# Fixed validation criterion, independent of the training loss weight.
scores={name:float(np.mean([row['xlb_recovery']['ic_relative_l2'] for row in rows]))
        for name,rows in report['models'].items() if name!='XLB'}
winner=min(scores,key=scores.get)
report['selection']={'criterion':'mean validation XLB-target IC recovery error, full space','scores':scores,'winner':winner,'weights':str(paths[winner])}
(root/'results/validation.json').write_text(json.dumps(report,indent=2))
print('SELECTED',winner,flush=True)
