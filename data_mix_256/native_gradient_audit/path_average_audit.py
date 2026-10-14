"""Prespecified path-average AD diagnostic for the unchanged native Float32 map."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from mosaic.benchmarks.problems import get_config

import native_gradient_audit as audit
from precision_companion import load

EPSILONS=[.001,.03,.1]
ORDERS=[8,16,32]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--saved-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py').resolve()
    native=load(source,'_path_average_native')
    t=Tesseract.from_tesseract_api(str(source))
    cfg=get_config('buoyancy-grid');solver=next(s for s in cfg.solvers if s.key=='jax_cfd')
    metrics=json.loads((args.saved_result/'result.json').read_text())['results'][0]['metrics']
    scales=jnp.asarray(metrics['scales'],dtype=jnp.float32)
    with np.load(args.saved_result/'corrector_fields.npz') as data:
        idx=list(map(str,data['solver_names'])).index(solver.name)
        reference=np.asarray(data[f'reference_{idx}'][72],dtype=np.float32)
    z=jnp.asarray(reference)/scales
    mask=np.ones((64,64,3),dtype=np.float32);mask[-1,:,0]=0;mask[:,-1,1]=0
    def flow(x,wrapped=False):
        s=x*scales*mask
        for _ in range(8):
            inputs=cfg.make_inputs(solver.name,s,nu=.0005,diffusivity=.0005,buoyancy=1.,dt=.005,steps=8,domain_extent=1.)
            s=(apply_tesseract(t,inputs) if wrapped else native.apply_jit(inputs))['result']
        ux=jnp.pad(s[:-1,:,0],((1,1),(0,0)));uy=jnp.pad(s[:,:-1,1],((0,0),(1,1)))
        return jnp.stack(((ux[:-1]+ux[1:])/2,(uy[:,:-1]+uy[:,1:])/2,s[...,2]),axis=-1)/scales
    probe=jnp.asarray(audit.probes()[('scalar','small')][0])
    value,back=jax.vjp(flow,z);gradient=np.asarray(back(probe)[0])
    assert np.array_equal(value,flow(z,True))
    pg=audit.projection(gradient);direction=jnp.asarray(pg/np.linalg.norm(pg),dtype=jnp.float32)
    def scalar(array):
        return float(np.sum(np.asarray(array,dtype=np.float64)*np.asarray(probe,dtype=np.float64)))
    center_ad=float(np.sum(gradient.astype(np.float64)*np.asarray(direction,dtype=np.float64)))
    report={'complete':False,'frame':72,'field':'scalar','band':'small','horizon':8,
            'epsilons':EPSILONS,'quadrature_orders':ORDERS,
            'interpretation':'Additional finite-path consistency diagnostic, not an integral theorem proof: native Float32 casts make the literal map discontinuous while AD differentiates through casts.',
            'formula':'average directional AD = 0.5 * sum(Gauss-Legendre weight * <probe,JVP(z+t*epsilon*d,d)>)',
            'input_sha256':audit.digest(z),'probe_sha256':audit.digest(probe),'direction_sha256':audit.digest(direction),
            'center_ad':center_ad,'forward_native_wrapper_bitwise_equal':True,
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),Path(audit.__file__)]},
            'samples':[]}
    for epsilon in EPSILONS:
        xp=z+epsilon*direction;xm=z-epsilon*direction
        plus=flow(xp,True);minus=flow(xm,True)
        assert np.array_equal(plus,flow(xp)) and np.array_equal(minus,flow(xm))
        central=scalar(np.asarray(plus,dtype=np.float64)-np.asarray(minus,dtype=np.float64))/(2*epsilon)
        row={'epsilon':epsilon,'central_fd':central,'center_ad':center_ad,'center_ad_fd_absolute_error':abs(center_ad-central),
             'center_ad_fd_relative_error':audit.relative(center_ad,central),'orders':[]}
        for order in ORDERS:
            nodes,weights=np.polynomial.legendre.leggauss(order)
            records=[];slopes=[]
            for node,weight in zip(nodes,weights,strict=True):
                distance=float(node)*epsilon
                point=z+distance*direction
                _,jvp=jax.jvp(flow,(point,),(direction,))
                slope=scalar(jvp);assert np.isfinite(slope)
                realized=(np.asarray(point,dtype=np.float64)-np.asarray(z,dtype=np.float64))/distance
                records.append({'node':float(node),'weight':float(weight),'nominal_distance':distance,
                                'directional_ad':slope,'realized_direction_norm':float(np.linalg.norm(realized)),
                                'realized_direction_cosine':float(np.sum(realized*np.asarray(direction,dtype=np.float64))/(np.linalg.norm(realized)*np.linalg.norm(direction))),
                                'changed_fraction':float(np.mean(realized!=0))})
                slopes.append(slope)
            average=float(.5*np.dot(weights,slopes))
            row['orders'].append({'order':order,'average_directional_ad':average,
                                  'average_fd_absolute_error':abs(average-central),'average_fd_relative_error':audit.relative(average,central),
                                  'average_center_absolute_difference':abs(average-center_ad),'nodes':records})
            report['samples']=[*report['samples'][:-1],row] if report['samples'] and report['samples'][-1]['epsilon']==epsilon else [*report['samples'],row]
            audit.save(args.output/'path_average_audit.json',report)
            print('PATH_AVERAGE',epsilon,order,average,central,flush=True)
    report['complete']=True;audit.save(args.output/'path_average_audit.json',report)


if __name__=='__main__':main()
