"""External evaluation of local vs finite-path derivative diagnostics, not CI code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

EPSILONS=[.001,.03,.1,.3,1.]
ORDERS=[8,16,32]
ABS_TOL=1e-6
REL_TOL=1e-3


def agreement(a,b):
    return bool(abs(a-b)<=ABS_TOL+REL_TOL*max(abs(a),abs(b)))


def error(a,b):
    return {'absolute':float(abs(a-b)), 'relative':float(abs(a-b)/max(abs(a),abs(b),1e-30)),
            'illustrative_agreement':agreement(a,b)}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def controls():
    """Known forward functions and supplied backward rules; no fitted thresholds."""
    cases=[
        ('correct_linear',0.,lambda x:x,lambda x:1.,lambda x:1.,
         'x','1','correct constant derivative; numerical sanity control'),
        ('smooth_curvature',.2,lambda x:x+x**3,lambda x:1+3*x*x,lambda x:1+3*x*x,
         'x+x³','1+3x²','correct smooth derivative; finite secant differs by epsilon²'),
        ('continuous_kink_nearby',0.,lambda x:abs(x-.02)+x,lambda x:np.sign(x-.02)+1,lambda x:np.sign(x-.02)+1,
         '|x-.02|+x','sign(x-.02)+1','local derivative exists at center; larger intervals cross a continuous kink'),
        ('continuous_kink_at_center',.02,lambda x:abs(x-.02)+x,lambda x:np.sign(x-.02)+1,lambda x:None if x==.02 else np.sign(x-.02)+1,
         '|x-.02|+x','sign(x-.02)+1 (chosen value1 at kink)','classical derivative undefined at center; agreement cannot validate a unique derivative'),
        ('genuine_backward_bias',.2,lambda x:np.sin(x)+x,lambda x:1.1*(np.cos(x)+1),lambda x:np.cos(x)+1,
         'sin(x)+x','1.1(cos(x)+1)','wrong derivative throughout path'),
        ('localized_wrong_center',0.,lambda x:x,lambda x:1+.5*np.exp(-(x/1e-4)**2),lambda x:1.,
         'x','1+.5 exp(-(x/1e-4)²)','wrong center derivative can be missed by quadrature nodes'),
        ('canceling_offcenter_errors',0.,lambda x:x,lambda x:1+20*x,lambda x:1.,
         'x','1+20x','center derivative correct; nonzero offcenter errors cancel in symmetric average'),
        ('discontinuous_forward',0.,lambda x:x+float(x>=.02),lambda x:1.,lambda x:None if x==.02 else 1.,
         'x+H(x-.02)','1 away from jump','derivative exists locally but path crossing jump violates continuity assumption'),
    ]
    result=[]
    for name,x,fn,back,true,formula,back_formula,meaning in cases:
        center=float(back(x));truth=true(x)
        row={'name':name,'center':x,'forward_formula':formula,'supplied_backward_formula':back_formula,'meaning':meaning,
             'true_center_derivative':None if truth is None else float(truth),
             'local_derivative_exists':truth is not None,'supplied_center_derivative':center,
             'center_backward_error':None if truth is None else error(center,truth),'epsilons':[]}
        for eps in EPSILONS:
            fd=float((fn(x+eps)-fn(x-eps))/(2*eps))
            item={'epsilon':eps,'central_fd':fd,'right_fd':float((fn(x+eps)-fn(x))/eps),
                  'left_fd':float((fn(x)-fn(x-eps))/eps),'local_ad_fd':error(center,fd),'orders':[]}
            for order in ORDERS:
                nodes,weights=np.polynomial.legendre.leggauss(order)
                positions=x+eps*nodes
                supplied=np.array([back(v) for v in positions]);truths=np.array([true(v) for v in positions],dtype=float)
                avg=float(.5*np.dot(weights,supplied));trueavg=float(.5*np.dot(weights,truths))
                item['orders'].append({'order':order,'average_supplied_derivative':avg,'average_true_derivative_quadrature':trueavg,
                                       'path_ad_fd':error(avg,fd),'sampled_backward_error_max':float(np.max(np.abs(supplied-truths))),
                                       'nodes':[{'node':float(n),'weight':float(w),'position':float(p),'supplied':float(s),'true':float(t)}
                                                for n,w,p,s,t in zip(nodes,weights,positions,supplied,truths,strict=True)]})
            if name=='correct_linear':
                assert abs(fd-1)<1e-12
                assert all(abs(o['average_supplied_derivative']-1)<1e-12 for o in item['orders'])
            if name=='smooth_curvature':
                assert abs(fd-(1+3*x*x+eps*eps))<1e-10
            if name=='genuine_backward_bias':
                assert all(abs(o['average_supplied_derivative']/o['average_true_derivative_quadrature']-1.1)<1e-12 for o in item['orders'])
            row['epsilons'].append(item)
        result.append(row)
    return result


def native_cases(mode,out):
    import jax
    import jax.numpy as jnp
    from tesseract_core import Tesseract
    from tesseract_jax import apply_tesseract
    from mosaic.benchmarks.problems import get_config
    from mosaic.benchmarks.core.experiment import random_direction
    from mosaic.benchmarks.problems.shared.gradient import _vjp_grad
    # Preserve main solver's ordinary precision; use its actual shared objective.
    cfg=get_config('ns-grid');solver=next(s for s in cfg.solvers if s.key=='jax_cfd')
    api=Path('mosaic/tesseracts/navier-stokes-grid/jax-cfd/tesseract_api.py').resolve()
    native=Tesseract.from_tesseract_api(str(api))
    x=cfg.make_ic['multimode'](N=32,L=2*np.pi,seed=0)
    assert x.shape==(32,32,1,2) and x.dtype==jnp.float32
    random=random_direction(x.shape,jax.random.PRNGKey(3401))
    class Counted:
        def __init__(self,target):
            self.target=target
            self.counts={'forward_api':0,'vjp_api':0}
        def __getattr__(self,name):
            return getattr(self.target,name)
        def apply(self,*args,**kwargs):
            self.counts['forward_api']+=1
            return self.target.apply(*args,**kwargs)
        def vector_jacobian_product(self,*args,**kwargs):
            self.counts['vjp_api']+=1
            return self.target.vector_jacobian_product(*args,**kwargs)
    results=[]
    for steps in ([1] if mode=='reduced' else [1,20]):
        t=Counted(native);counts=t.counts
        base_inputs=cfg.make_inputs(solver.name,x,nu=.01,dt=.01,steps=steps,domain_extent=2*np.pi)
        ic_scale=float(jnp.sqrt(jnp.mean(x**2)+1e-30))
        def forward(value):
            return np.asarray(apply_tesseract(t,{**base_inputs,'v0':value})['result'])
        def gradient(value):
            return np.asarray(_vjp_grad(t,{**base_inputs,'v0':value},'result','v0'))
        def timed(fn):
            started=time.perf_counter();value=fn();return value,time.perf_counter()-started
        base,cold_forward=timed(lambda:forward(x))
        grad,cold_vjp=timed(lambda:gradient(x))
        assert counts['forward_api']>=2 and counts['vjp_api']==1,counts
        warm_forward=[];warm_vjp=[]
        for _ in range(3):
            value,seconds=timed(lambda:forward(x));assert np.array_equal(value,base);warm_forward.append(seconds)
            value,seconds=timed(lambda:gradient(x));assert np.array_equal(value,grad);warm_vjp.append(seconds)
        assert np.isfinite(base).all() and np.isfinite(grad).all() and np.linalg.norm(grad)>0
        row={'steps':steps,'physics':{'N':32,'dt':.01,'nu':.01,'L':2*np.pi,'ic':'multimode','ic_seed':0},
             'precision':str(x.dtype),'jax_enable_x64':bool(jax.config.x64_enabled),'ic_scale':ic_scale,
             'api_source_sha256':hashlib.sha256(api.read_bytes()).hexdigest(),
             'shared_gradient_source_sha256':hashlib.sha256(Path(sys.modules[_vjp_grad.__module__].__file__).read_bytes()).hexdigest(),
             'ic_sha256':digest(x),'objective':'Exact shared _vjp_grad via Tesseract: jnp.sum(result**2). FD jnp.sum(plus**2-minus**2)/(2*abs_eps). AD dot jnp.dot. abs_eps=epsilon*IC_RMS.',
             'base_objective':float(jnp.sum(jnp.asarray(base)**2)),
             'costs':{'cold_forward_seconds':cold_forward,'cold_objective_gradient_seconds':cold_vjp,'warm_forward_seconds':warm_forward,'warm_objective_gradient_seconds':warm_vjp,
                      'counting':'Actual in-process Tesseract apply/VJP calls counted by delegating proxy. Shared _vjp_grad includes primal apply plus VJP internal work. No container/network overhead. IC/setup excluded.'},
             'directions':[]}
        dirs=[('gradient',jnp.asarray(grad)/jnp.linalg.norm(jnp.asarray(grad)))]
        if mode!='reduced':dirs.append(('fixed_random',random))
        for name,d in dirs:
            ad=float(jnp.dot(jnp.asarray(grad).ravel(),d.ravel()))
            direction={'name':name,'sha256':digest(d),'local_ad':ad,'epsilons':[]}
            for eps in ([.03] if mode=='reduced' else EPSILONS):
                absolute=eps*ic_scale;before=dict(counts);started=time.perf_counter()
                item={'epsilon':eps,'absolute_epsilon':absolute,'orders':[]}
                try:
                    plus=jnp.asarray(forward(x+absolute*d));minus=jnp.asarray(forward(x-absolute*d))
                    fd=float(jnp.sum(plus**2-minus**2)/(2*absolute))
                    if not np.isfinite(fd):raise FloatingPointError('nonfinite central FD')
                    fd_seconds=time.perf_counter()-started
                    p=(np.asarray(x+absolute*d,dtype=np.float64)-np.asarray(x,dtype=np.float64))/absolute
                    m=(np.asarray(x,dtype=np.float64)-np.asarray(x-absolute*d,dtype=np.float64))/absolute
                    item.update(central_fd=fd,local_ad_fd=error(ad,fd),fd_seconds=fd_seconds,
                                central_fd_float64_reduction_diagnostic=float(np.sum(np.asarray(plus,dtype=np.float64)**2-np.asarray(minus,dtype=np.float64)**2)/(2*absolute)),
                                precision_variant_note='Float64 reduction changes only diagnostic arithmetic, not shared FD; primary central_fd uses exact shared Float32 expression',
                                realized_plus_direction_norm=float(np.linalg.norm(p)),realized_minus_direction_norm=float(np.linalg.norm(m)))
                except Exception as exc:
                    item['unavailable']={'phase':'central_fd','type':type(exc).__name__,'message':str(exc)}
                    item['api_calls']={key:counts[key]-before[key] for key in counts}
                    direction['epsilons'].append(item)
                    continue
                for order in ORDERS:
                    nodes,weights=np.polynomial.legendre.leggauss(order);slopes=[];records=[];started=time.perf_counter();counts_before=dict(counts)
                    try:
                        for node,weight in zip(nodes,weights,strict=True):
                            offset=float(node)*absolute;point=x+offset*d;g=gradient(point)
                            slope=float(jnp.dot(jnp.asarray(g).ravel(),d.ravel()))
                            if not np.isfinite(slope):raise FloatingPointError('nonfinite directional AD')
                            actual=(np.asarray(point,dtype=np.float64)-np.asarray(x,dtype=np.float64))/offset
                            slopes.append(slope);records.append({'node':float(node),'weight':float(weight),'directional_ad':slope,
                                                                'realized_direction_norm':float(np.linalg.norm(actual))})
                        avg=float(.5*np.dot(weights,slopes))
                        item['orders'].append({'order':order,'average_ad':avg,'path_ad_fd':error(avg,fd),'seconds':time.perf_counter()-started,
                                               'api_calls':{key:counts[key]-counts_before[key] for key in counts},'nodes':records})
                    except Exception as exc:
                        item['orders'].append({'order':order,'unavailable':{'type':type(exc).__name__,'message':str(exc)},
                                               'seconds':time.perf_counter()-started,'nodes':records,
                                               'api_calls':{key:counts[key]-counts_before[key] for key in counts}})
                item['api_calls']={key:counts[key]-before[key] for key in counts}
                direction['epsilons'].append(item)
            row['directions'].append(direction)
        row['total_api_calls']=dict(counts);results.append(row)
        write(out/'native_partial.json',results)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['controls','reduced','full'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'complete':False,'mode':args.mode,'source_commit':'1b47f6213822dd4c4d20ca27682b306b39fe274d','epsilons':EPSILONS,'quadrature_orders':ORDERS,
            'agreement_display_only':{'absolute_tolerance':ABS_TOL,'relative_tolerance':REL_TOL,'not_a_proposed_ci_gate':True},
            'method':'Local AD compares point derivative to finite secant; path-average AD asks a separate interval consistency question. Neither agreement alone proves all derivatives correct.',
            'precision_caveat':'Float32 native casts/rounding make literal numerical maps discontinuous; quadrature agreement is diagnostic, not a fundamental-theorem proof.',
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'analytic_controls':controls()}
    write(args.output/'evaluation.json',report)
    started=time.perf_counter()
    if args.mode!='controls':report['native_cases']=native_cases(args.mode,args.output)
    report['native_seconds']=time.perf_counter()-started
    report['complete']=True;write(args.output/'evaluation.json',report)


if __name__=='__main__':main()
