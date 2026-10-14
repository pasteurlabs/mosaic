"""Fixed-forward derivative audit. Run only on Slurm, no optimization or repair."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import dctn, idctn
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

EPSILONS = [1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1, .3, 1.]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


def relative(a, b):
    return float(abs(a-b)/max(abs(a), abs(b), 1e-30))


def projection(g):
    g = np.asarray(g, dtype=np.float64).copy()
    n = g.shape[0]
    g[-1,:,0] = 0
    g[:,-1,1] = 0
    ux = np.pad(g[:-1,:,0], ((1,1),(0,0)))
    uy = np.pad(g[:,:-1,1], ((0,0),(1,1)))
    rhs = dctn(np.diff(ux,axis=0)+np.diff(uy,axis=1), norm='ortho')
    kx,ky = np.meshgrid(np.arange(n),np.arange(n), indexing='ij')
    eig = -4*(np.sin(np.pi*kx/(2*n))**2+np.sin(np.pi*ky/(2*n))**2)
    eig[0,0] = 1
    rhs[0,0] = 0
    p = idctn(rhs/eig,norm='ortho')
    g[:-1,:,0] -= np.diff(p,axis=0)
    g[:,:-1,1] -= np.diff(p,axis=1)
    return g


def probes(n=64):
    kx,ky = np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    q = np.hypot(kx,ky)
    bands = {'large':(q>0)&(q<=4),'medium':(q>4)&(q<=16),'small':q>16}
    result = {}
    for fi, field in enumerate(['scalar','velocity']):
        channels = [2] if field == 'scalar' else [0,1]
        for bi,(band,mask) in enumerate(bands.items()):
            locations = np.argwhere(mask)
            coeff,_ = np.linalg.qr(np.random.default_rng(4700+100*fi+bi).normal(size=(len(locations)*len(channels),8)), mode='reduced')
            arrays=[]
            for rank in range(8):
                modes=np.zeros((n,n,3))
                for ci,channel in enumerate(channels):
                    modes[locations[:,0],locations[:,1],channel] = coeff[ci*len(locations):(ci+1)*len(locations),rank]
                arrays.append(idctn(modes,norm='ortho',axes=(0,1)))
            result[(field,band)] = np.asarray(arrays,dtype=np.float32)
    return result


def finite_differences(fn, x, direction, gradient, cotangent=None):
    """Float64 reductions of unchanged Float32 outputs; retain realized input offsets."""
    x=jnp.asarray(x); direction=jnp.asarray(direction,dtype=x.dtype)
    gradient=np.asarray(gradient,dtype=np.float64)
    base=np.asarray(fn(x),dtype=np.float64)
    def scalar(value):
        return float(np.sum(value*np.asarray(cotangent,dtype=np.float64))) if cotangent is not None else float(value)
    base_scalar=scalar(base)
    ad=float(np.sum(gradient*np.asarray(direction,dtype=np.float64)))
    rows=[]
    for epsilon in EPSILONS:
        try:
            xp=x+epsilon*direction; xm=x-epsilon*direction
            rp=(np.asarray(xp,dtype=np.float64)-np.asarray(x,dtype=np.float64))/epsilon
            rm=(np.asarray(x,dtype=np.float64)-np.asarray(xm,dtype=np.float64))/epsilon
            plus=np.asarray(fn(xp),dtype=np.float64); minus=np.asarray(fn(xm),dtype=np.float64)
            central=scalar(plus-minus)/(2*epsilon)
            right=scalar(plus-base)/epsilon; left=scalar(base-minus)/epsilon
            vals=[central,right,left]
            if not np.isfinite(vals).all():
                raise FloatingPointError('nonfinite finite-difference outputs')
            rows.append({'epsilon':epsilon,'central':central,'right':right,'left':left,
                         'ad':ad,'central_absolute_error':abs(central-ad),'central_relative_error':relative(central,ad),
                         'one_sided_gap':abs(right-left), 'realized_plus_direction_norm':float(np.linalg.norm(rp)),
                         'realized_minus_direction_norm':float(np.linalg.norm(rm)),
                         'realized_plus_ad':float(np.sum(gradient*rp)), 'realized_minus_ad':float(np.sum(gradient*rm)),
                         'plus_changed_fraction':float(np.mean(rp!=0)), 'minus_changed_fraction':float(np.mean(rm!=0))})
        except Exception as exc:
            rows.append({'epsilon':epsilon,'unavailable':type(exc).__name__,'reason':str(exc)})
    return {'base_objective':base_scalar,'autodiff_directional':ad,'epsilon_sweep':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['reduced','full'],required=True)
    parser.add_argument('--saved-result',type=Path,required=True)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from mosaic.benchmarks.problems import get_config
    cfg=get_config('buoyancy-grid');spec=next(s for s in cfg.solvers if s.key=='jax_cfd')
    api=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py').resolve()
    mspec=importlib.util.spec_from_file_location('_native_audit_solver',api)
    native=importlib.util.module_from_spec(mspec);sys.modules[mspec.name]=native;mspec.loader.exec_module(native)
    t=Tesseract.from_tesseract_api(str(api))
    metrics=json.loads((args.saved_result/'result.json').read_text())['results'][0]['metrics']
    scales=jnp.asarray(metrics['scales'],dtype=jnp.float32)
    with np.load(args.saved_result/'corrector_fields.npz',allow_pickle=False) as data:
        idx=list(map(str,data['solver_names'])).index(spec.name)
        reference=np.asarray(data[f'reference_{idx}'],dtype=np.float32)
    def inputs(state):
        return cfg.make_inputs(spec.name,state,nu=.0005,diffusivity=.0005,buoyancy=1.,dt=.005,steps=8,domain_extent=1.)
    def direct_step(state):
        return native.apply_jit(inputs(state))['result']
    def wrapper_step(state):
        return apply_tesseract(t,inputs(state))['result']
    mask=np.ones((64,64,3),dtype=np.float32);mask[-1,:,0]=0;mask[:,-1,1]=0
    def flow(z,step):
        s=z*scales*mask
        for _ in range(8):s=step(s)
        ux=jnp.pad(s[:-1,:,0],((1,1),(0,0)));uy=jnp.pad(s[:,:-1,1],((0,0),(1,1)))
        return jnp.stack(((ux[:-1]+ux[1:])/2,(uy[:,:-1]+uy[:,1:])/2,s[...,2]),axis=-1)/scales
    direct=lambda z:flow(z,direct_step)
    wrapper=lambda z:flow(z,wrapper_step)
    basis=probes()
    report={'complete':False,'mode':args.mode,'source_commit':'a2339b9587b2eccb845c324325cb5288aae6aa04',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [api,Path(__file__)]},
            'saved_artifacts_sha256':{name:hashlib.sha256((args.saved_result/name).read_bytes()).hexdigest() for name in ['result.json','corrector_fields.npz']},
            'basis_hashes':{f'{f}/{b}':digest(v) for (f,b),v in basis.items()},
            'epsilon_ladder':EPSILONS,'horizon':8,'samples':[],
            'interpretation':'JVP/VJP agreement is same-AD consistency; FD evaluates the unchanged Float32 public map. Non-smoothness and finite precision are not automatically derivative defects.'}
    historical_hashes=json.loads(Path(__file__).with_name('historical_basis_hashes.json').read_text())
    assert report['basis_hashes']==historical_hashes, 'historical probe hashes differ'
    report['historical_basis_hashes_match']=True
    started=time.perf_counter()
    cases=[(72,'scalar','small'),(0,'scalar','small')] if args.mode=='reduced' else [(f,a,b) for f in [0,24,48,72] for a,b in basis]
    for frame,field,band in cases:
        z=jnp.asarray(reference[frame])/scales;c=jnp.asarray(basis[(field,band)][0])
        yd,bd=jax.vjp(direct,z);yw,bw=jax.vjp(wrapper,z)
        gd=np.asarray(bd(c)[0]);gw=np.asarray(bw(c)[0])
        assert np.array_equal(yd,yw),'native/wrapper forward mismatch'
        assert np.isfinite(gd).all() and np.isfinite(gw).all()
        projected=projection(gd)
        random=projection(np.random.default_rng(9100+frame).normal(size=z.shape))
        row={'frame':frame,'field':field,'band':band,'input_sha256':digest(z),'probe_sha256':digest(c),
             'forward_bitwise_equal':True,'native_wrapper_vjp_relative_l2':float(np.linalg.norm(gd-gw)/max(np.linalg.norm(gd),1e-30)),
             'native_wrapper_vjp_max_absolute':float(np.max(np.abs(gd-gw))),
             'historical_gradient_norm':float(np.linalg.norm(projection(gw))), 'directions':{}}
        for name,d in [('projected_gradient',projected),('fixed_random',random)]:
            d=jnp.asarray(d/np.linalg.norm(d),dtype=jnp.float32)
            _,jvp=jax.jvp(direct,(z,),(d,))
            left=float(np.sum(np.asarray(jvp,dtype=np.float64)*np.asarray(c,dtype=np.float64)))
            right=float(np.sum(np.asarray(gd,dtype=np.float64)*np.asarray(d,dtype=np.float64)))
            record=finite_differences(wrapper,z,d,gd,c)
            record['jvp_vjp_same_ad_check']={'jvp_dot':left,'vjp_dot':right,'relative_error':relative(left,right),'absolute_error':abs(left-right)}
            row['directions'][name]=record
        report['samples'].append(row);save(args.output/'native_gradient_audit.json',report)
        print('AUDITED',frame,field,band,flush=True)
    if args.replay:
        neural_audit(args.replay,args.mode,direct_step,wrapper_step,report,args.output)
    else:
        report['neural_audit']={'available':False,'reason':'requires separately admitted clean replay weights and training references'}
    report.update(complete=True,seconds=time.perf_counter()-started)
    save(args.output/'native_gradient_audit.json',report)


def neural_audit(replay,mode,direct_step,wrapper_step,report,out):
    from jax.flatten_util import ravel_pytree
    from mosaic.benchmarks.problems.buoyancy_grid.training import window_loss
    admission=json.loads((replay/'replay_admission.json').read_text());assert admission['exact_prior_metrics_match'] is True
    with np.load(replay/'training_data.npz') as data:
        references=np.asarray(data['training_jax']);scales=jnp.asarray(data['scales']);cases=np.asarray(data['case_ids']);starts=np.asarray(data['starts'])
    windows=[(int(cases[0]),int(starts[0]))]+[(i,24) for i in range(4)]
    if mode=='reduced':windows=windows[:1]
    replay_hashes={name:hashlib.sha256((replay/name).read_bytes()).hexdigest() for name in ['replay_admission.json','training_data.npz','params_0.npz','params_192.npz']}
    rows=[]
    for update in [0,192]:
        with np.load(replay/f'params_{update}.npz') as data:
            params=tuple({key:jnp.asarray(data[f'{i}_{key}']) for key in ['w','b']} for i in range(3))
        flat,unravel=ravel_pytree(params)
        for case,start in windows:
            targets=jnp.asarray(references[case,start:start+9])
            direct=lambda p:window_loss(unravel(p),targets,direct_step,scales)
            wrapped=lambda p:window_loss(unravel(p),targets,wrapper_step,scales)
            vd,gd=jax.value_and_grad(direct)(flat);vw,gw=jax.value_and_grad(wrapped)(flat)
            assert np.array_equal(vd,vw)
            random=np.random.default_rng(32019).normal(size=flat.shape)
            row={'update':update,'case':case,'start':start,'params_file_sha256':replay_hashes[f'params_{update}.npz'],'targets_sha256':digest(targets),'base_loss':float(vd),'forward_bitwise_equal':True,
                 'native_wrapper_gradient_relative_l2':float(np.linalg.norm(np.asarray(gd-gw))/np.linalg.norm(gd)), 'directions':{}}
            for name,d in [('native_gradient',np.asarray(gd)),('fixed_random',random)]:
                d=jnp.asarray(d/np.linalg.norm(d),dtype=flat.dtype)
                row['directions'][name]=finite_differences(wrapped,flat,d,gd)
                _,jvp=jax.jvp(direct,(flat,),(d,))
                dot=float(np.dot(np.asarray(gd,dtype=np.float64),np.asarray(d,dtype=np.float64)))
                row['directions'][name]['jvp_vjp_same_ad_check']={'jvp':float(jvp),'vjp_dot':dot,'relative_error':relative(float(jvp),dot),'absolute_error':abs(float(jvp)-dot)}
            rows.append(row);report['neural_audit']={'available':True,'weights_origin':'newly reconstructed clean replay admitted against exact original metrics; not original saved weights','replay_file_sha256':replay_hashes,'rows':rows};save(out/'native_gradient_audit.json',report)


if __name__=='__main__':main()
