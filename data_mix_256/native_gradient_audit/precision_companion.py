"""Different-precision diagnostic, never a proof/reference for the native Float32 map."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mosaic.benchmarks.problems import get_config

import native_gradient_audit as audit

LADDER=[1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,.001,.003,.01]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def comparison(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return {'native_l2':float(np.linalg.norm(a)), 'companion_l2':float(np.linalg.norm(b)),
            'difference_l2':float(np.linalg.norm(a-b)), 'difference_relative_native_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-300)),
            'max_absolute_difference':float(np.max(np.abs(a-b)))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--saved-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py').resolve()
    original=source.read_text()
    before='return {"result": result.astype(jnp.float32)}'
    after='return {"result": result}'
    assert original.count(before)==1
    changed=original.replace(before,after)
    companion_source=args.output/'diagnostic_native_float64.py';companion_source.write_text(changed)
    native=load(source,'_precision_native');companion=load(companion_source,'_precision_companion')
    cfg=get_config('buoyancy-grid');solver=next(s for s in cfg.solvers if s.key=='jax_cfd')
    metrics=json.loads((args.saved_result/'result.json').read_text())['results'][0]['metrics']
    scales32=jnp.asarray(metrics['scales'],dtype=jnp.float32)
    with np.load(args.saved_result/'corrector_fields.npz') as data:
        idx=list(map(str,data['solver_names'])).index(solver.name)
        state=np.asarray(data[f'reference_{idx}'][72],dtype=np.float32)
    z32=jnp.asarray(state)/scales32;z64=z32.astype(jnp.float64)
    mask=np.ones((64,64,3),dtype=np.float32);mask[-1,:,0]=0;mask[:,-1,1]=0
    def flow(z,module,dtype):
        scale=scales32.astype(dtype)
        s=z*scale*jnp.asarray(mask,dtype=dtype)
        for _ in range(8):
            inputs=cfg.make_inputs(solver.name,s,nu=.0005,diffusivity=.0005,buoyancy=1.,dt=.005,steps=8,domain_extent=1.)
            # physics.make_inputs explicitly uses Float32 physical scalar arrays.
            assert all(inputs[key].dtype==jnp.float32 for key in ['viscosity','diffusivity','buoyancy','dt'])
            s=module.apply_jit(inputs)['result']
        ux=jnp.pad(s[:-1,:,0],((1,1),(0,0)));uy=jnp.pad(s[:,:-1,1],((0,0),(1,1)))
        return jnp.stack(((ux[:-1]+ux[1:])/2,(uy[:,:-1]+uy[:,1:])/2,s[...,2]),axis=-1)/scale
    native_flow=lambda z:flow(z,native,jnp.float32)
    companion_flow=lambda z:flow(z,companion,jnp.float64)
    probe32=jnp.asarray(audit.probes()[('scalar','small')][0]);probe64=probe32.astype(jnp.float64)
    yn,bn=jax.vjp(native_flow,z32);yc,bc=jax.vjp(companion_flow,z64)
    gn=np.asarray(bn(probe32)[0]);gc=np.asarray(bc(probe64)[0])
    assert yc.dtype==jnp.float64 and yn.dtype==jnp.float32
    assert np.isfinite(gn).all() and np.isfinite(gc).all()
    pn=audit.projection(gn);pc=audit.projection(gc)
    native_direction=jnp.asarray(pn/np.linalg.norm(pn),dtype=jnp.float32).astype(jnp.float64)
    companion_direction=jnp.asarray(pc/np.linalg.norm(pc),dtype=jnp.float64)
    report={'complete':False,'frame':72,'field':'scalar','band':'small','horizon':8,
            'purpose':'Different precision map diagnostic ONLY; not a proof or accuracy reference for the unchanged native Float32 forward map',
            'changes':['sole solver-source edit removes interval-final Float32 output cast',
                       'same nominal normalized native Float32 z values cast toFloat64',
                       'initial z*scale and mask multiplication, centering and output normalization computed inFloat64',
                       'physics coefficient arrays remain identical Float32 values promoted internally by native solver'],
            'source_commit':'a2339b9587b2eccb845c324325cb5288aae6aa04',
            'source_patch':{'before':before,'after':after,'replacement_count':1,
                            'native_sha256':hashlib.sha256(original.encode()).hexdigest(),
                            'companion_sha256':hashlib.sha256(changed.encode()).hexdigest(),
                            'patch_sha256':hashlib.sha256((before+'\n'+after).encode()).hexdigest()},
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_input_sha256':audit.digest(z32),'probe_sha256':audit.digest(probe32),
            'native_companion_physical_initial_comparison':comparison(z32*scales32*mask,z64*scales32.astype(jnp.float64)*jnp.asarray(mask,dtype=jnp.float64)),
            'native_companion_output_comparison':comparison(yn,yc),
            'native_companion_raw_gradient_comparison':comparison(gn,gc),
            'native_companion_projected_gradient_comparison':comparison(pn,pc),
            'epsilon_ladder':LADDER,'directions':{}}
    audit.EPSILONS=LADDER
    for name,direction in [('original_native_projected_gradient',native_direction),('companion_projected_gradient',companion_direction)]:
        row=audit.finite_differences(companion_flow,z64,direction,gc,probe64)
        _,jvp=jax.jvp(companion_flow,(z64,),(direction,))
        left=float(np.sum(np.asarray(jvp)*np.asarray(probe64)));right=float(np.sum(gc*np.asarray(direction)))
        row['same_ad_adjoint']={'jvp_dot':left,'vjp_dot':right,'absolute_error':abs(left-right),'relative_error':audit.relative(left,right)}
        row['direction_sha256']=audit.digest(direction)
        report['directions'][name]=row
        audit.save(args.output/'precision_companion.json',report)
    report['complete']=True;audit.save(args.output/'precision_companion.json',report)


if __name__=='__main__':main()
