"""External diagnostic: sampled scale-restricted adjoints, no new training."""
from pathlib import Path
import os,time,json,hashlib
import numpy as np
from scipy.fft import dctn,idctn
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from mosaic.benchmarks.problems import get_config

START=time.perf_counter()
SOLVER=os.environ['SOLVER_NAME'];SAVED=Path(os.environ['SAVED_RESULT_DIR']);OUT=Path(os.environ['ANALYSIS_OUTPUT_DIR']);OUT.mkdir(parents=True,exist_ok=True)
N=64;RANK=8;H=8;FRAMES=[0,24,48,72];EPSILONS=[.03,.1]
cfg=get_config('buoyancy-grid');spec=next(s for s in cfg.solvers if s.key==SOLVER.replace('-','_'))
t=Tesseract.from_tesseract_api(str(Path(f'mosaic/tesseracts/buoyancy-grid/{SOLVER}/tesseract_api.py').resolve()))
result=json.loads((SAVED/'result.json').read_text());metrics=result['results'][0]['metrics'];scales=jnp.asarray(metrics['scales'],dtype=jnp.float32)
with np.load(SAVED/'corrector_fields.npz',allow_pickle=False) as data:
    idx=list(map(str,data['solver_names'])).index(spec.name);reference=np.asarray(data[f'reference_{idx}'],dtype=np.float32)
assert reference.shape==(101,N,N,3)
mask=np.ones((N,N,3),dtype=np.float32);mask[-1,:,0]=0;mask[:,-1,1]=0
jmask=jnp.asarray(mask)
kx,ky=np.meshgrid(np.arange(N),np.arange(N),indexing='ij');q=np.hypot(kx,ky)
bands={'large':(q>0)&(q<=4),'medium':(q>4)&(q<=16),'small':q>16}
eig=-4*(np.sin(np.pi*kx/(2*N))**2+np.sin(np.pi*ky/(2*N))**2);eig[0,0]=1

def divergence(v):
    x=np.pad(v[:-1,:,0],((1,1),(0,0)));y=np.pad(v[:,:-1,1],((0,0),(1,1)))
    return np.diff(x,axis=0)+np.diff(y,axis=1)

def tangent(g):
    """Euclidean orthogonal projection onto zero-normal solenoidal velocities; scalar free."""
    g=np.asarray(g,dtype=np.float64)*mask
    rhs=dctn(divergence(g),type=2,norm='ortho');rhs[0,0]=0
    p=idctn(rhs/eig,type=2,norm='ortho')
    out=g.copy();out[:-1,:,0]-=np.diff(p,axis=0);out[:,:-1,1]-=np.diff(p,axis=1)
    assert np.max(np.abs(divergence(out)))<1e-10*max(1.,np.linalg.norm(out))
    return out

# Validate projection algebra independently of a solver.
rng=np.random.default_rng(8821);a=rng.normal(size=(N,N,3));b=rng.normal(size=a.shape)
pa=tangent(a);pb=tangent(b)
np.testing.assert_allclose(tangent(pa),pa,rtol=1e-10,atol=1e-12)
np.testing.assert_allclose(np.sum(a*pb),np.sum(pa*b),rtol=1e-10,atol=1e-10)
assert abs(np.sum(pa*(a-pa)))<1e-9

def center(s):
    ux=jnp.pad(s[:-1,:,0],((1,1),(0,0)));uy=jnp.pad(s[:,:-1,1],((0,0),(1,1)))
    return jnp.stack(((ux[:-1]+ux[1:])/2,(uy[:,:-1]+uy[:,1:])/2,s[...,2]),axis=-1)

def step(s):
    inputs=cfg.make_inputs(spec.name,s,nu=.0005,diffusivity=.0005,buoyancy=1.,dt=.005,steps=8,domain_extent=1.)
    return apply_tesseract(t,inputs)['result']

def flow(z,horizon=H):
    s=z*scales*jmask
    for _ in range(horizon):s=step(s)
    return center(jnp.asarray(s))/scales

# Same random orthonormal output directions for every backend and reference time.
probes={};basis_hashes={}
for fi,field in enumerate(['scalar','velocity']):
    channels=[2] if field=='scalar' else [0,1]
    for bi,(band,selected) in enumerate(bands.items()):
        locations=np.argwhere(selected);dim=len(locations)*len(channels)
        coefficients=np.random.default_rng(4700+100*fi+bi).normal(size=(dim,RANK))
        coeff,_=np.linalg.qr(coefficients,mode='reduced')
        arrays=[]
        for r in range(RANK):
            modes=np.zeros((N,N,3));offset=0
            for channel in channels:
                modes[locations[:,0],locations[:,1],channel]=coeff[offset:offset+len(locations),r];offset+=len(locations)
            arrays.append(idctn(modes,type=2,norm='ortho',axes=(0,1)))
        arrays=np.stack(arrays)
        np.testing.assert_allclose(arrays.reshape(RANK,-1)@arrays.reshape(RANK,-1).T,np.eye(RANK),rtol=1e-10,atol=1e-12)
        probes[(field,band)]=arrays.astype(np.float32)
        basis_hashes[f'{field}/{band}']=hashlib.sha256(probes[(field,band)].tobytes()).hexdigest()

report={'solver':spec.name,'source_commit':'a2339b9587b2eccb845c324325cb5288aae6aa04','slurm_job':os.environ.get('SLURM_JOB_ID'),'rank':RANK,'frames':FRAMES,'times':[f*.04 for f in FRAMES],'horizon_intervals':H,'horizon_duration':.32,'scales':metrics['scales'],'selected_corrector_update':metrics['solver_in_loop']['selected_update'],'fd_epsilons':EPSILONS,'basis_hashes':basis_hashes,'source_hashes':{name:hashlib.sha256((SAVED/name).read_bytes()).hexdigest() for name in ['corrector_fields.npz','result.json']},'checks':{'projection_idempotent_symmetric_orthogonal':True,'orthonormal_probes':True},'samples':[]}

def spectrum(g):
    sv=np.linalg.svd(np.asarray(g).reshape(RANK,-1),compute_uv=False)
    # Float32 public gradients: do not publish a finite condition when the smallest
    # singular value lies below a conservative precision-resolution threshold.
    resolved=bool(sv[-1]>1e-6*sv[0] and sv[0]>0)
    return {'singular_values':sv.tolist(),'rms_gain':float(np.sqrt(np.mean(sv**2))),'condition':float(sv[0]/sv[-1]) if resolved else None,'rank_resolved':resolved,'smallest_relative_singular_value':float(sv[-1]/sv[0]) if sv[0]>0 else None}

# H0 accounts for output centering/projection geometry without a solver evolution.
z0=jnp.asarray(reference[0])/scales
_,back0=jax.vjp(lambda z:flow(z,0),z0)
initial={key:spectrum([tangent(back0(jnp.asarray(probe))[0]) for probe in basis]) for key,basis in probes.items()}
for frame in FRAMES:
    z=jnp.asarray(reference[frame])/scales
    value,back=jax.vjp(flow,z);assert np.isfinite(np.asarray(value)).all()
    for (field,band),basis in probes.items():
        began=time.perf_counter();gradients=[]
        for probe in basis:
            g=tangent(back(jnp.asarray(probe))[0]);assert np.isfinite(g).all();gradients.append(g)
        local=spectrum(gradients);identity=initial[(field,band)]
        local['gain_relative_H0']=local['rms_gain']/identity['rms_gain']
        local['condition_relative_H0']=local['condition']/identity['condition'] if local['condition'] is not None and identity['condition'] is not None else None
        # Fixed first probe, both prespecified perturbation sizes, every frame/band.
        # Direction lies in the admissible input tangent; no best-epsilon selection.
        norm=float(np.linalg.norm(gradients[0]));fd=[]
        if norm>0:
            direction=jnp.asarray(gradients[0]/norm,dtype=jnp.float32);probe=jnp.asarray(basis[0])
            for epsilon in EPSILONS:
                plus=np.asarray(flow(z+epsilon*direction),dtype=np.float64)
                minus=np.asarray(flow(z-epsilon*direction),dtype=np.float64)
                finite=float(np.sum((plus-minus)*np.asarray(probe,dtype=np.float64))/(2*epsilon))
                fd.append({'epsilon':epsilon,'autodiff':norm,'finite_difference':finite,'absolute_error':abs(norm-finite),'relative_error':abs(norm-finite)/max(abs(norm),abs(finite),1e-12)})
        row={'frame':frame,'time':frame*.04,'field':field,'band':band,'H8':local,'H0':identity,'finite_differences':fd,'seconds':time.perf_counter()-began}
        report['samples'].append(row)
        (OUT/'gradient_scales_partial.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('PROGRESS',frame,field,band,'gain/H0',local['gain_relative_H0'],'condition',local['condition'],'fd',[r['relative_error'] for r in fd],flush=True)
    del back,value
report['seconds']=time.perf_counter()-START;report['complete']=True
report['method']='Eight orthonormal DCT-II output cotangents per field/band. Pull back through eight uncorrected solver intervals in training-RMS-normalized coordinates. Project input velocity gradients onto divergence-free zero-normal-face tangent; scalar input gradients remain free. SVD of eight resulting gradients gives sampled adjoint gain and conditioning.'
report['limitations']=['Sampled eight-dimensional restriction, not full Jacobian condition or neural-parameter conditioning.','Input tangent includes scalar variations; CNN itself changes only velocity.','State samples follow each backend restricted fine reference, not learned model or training trajectory.','Frame0 is at zero velocity where discretizations can have nonsmooth derivatives; all fixed-epsilon discrepancies are retained.','Float32 API; singular values below1e-6 of largest are marked unresolved, not replaced by an epsilon floor.','Descriptive pairing with full-rollout spectral error changes; no causal attribution.']
(OUT/'gradient_scales.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('GRADIENT_SCALES_COMPLETE',report['seconds'],flush=True)
