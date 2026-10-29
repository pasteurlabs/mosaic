"""External matched-window JAX-CFD stability pilot. Run only on Slurm."""
import argparse,copy,hashlib,importlib.util,inspect,json,time,traceback
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.fft import dctn
from mosaic.benchmarks.problems.buoyancy_grid.config import problem
from mosaic.benchmarks.problems.buoyancy_grid import training
from mosaic.benchmarks.problems.buoyancy_grid.corrector import apply_corrector

p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
recipe=json.loads(args.recipe.read_text())
receipt={'complete':False,'arm':args.arm,'backend':'direct native apply_jit, no HTTP','model_seed':recipe['model_seed'],'window_seed':0,'batch_size':recipe['batch'],'clip_global_norm':1.0,'validation_window_evaluations':list(range(0,257,16))}
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
def write(): (out/'receipt.json').write_text(json.dumps(clean(receipt),indent=2,allow_nan=False)+'\n')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save_params(name,params):
    leaves,tree=jax.tree.flatten(params);np.savez(out/(name+'.npz'),**{f'leaf_{i}':np.asarray(x) for i,x in enumerate(leaves)});receipt[name]={'tree':str(tree),'leaf_shapes':[list(x.shape) for x in leaves],'sha256':[ah(x) for x in leaves]}
path=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py');spec=importlib.util.spec_from_file_location('native_buoyancy',path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
training.apply_tesseract=lambda t,inputs:native.apply_jit(inputs)
kwargs=copy.deepcopy(inspect.signature(problem.experiments['optimization/solver_in_loop'].fn).parameters['_kw'].default);run=kwargs['runs'][0];run['training'].update(learning_rates=[recipe['lr']],validation_interval=16,model_seed=recipe['model_seed'],hidden_channels=recipe['width'],correct_temperature=recipe['correct_temperature'])
receipt['sweep_recipe']=recipe
solver=next(s for s in problem.solvers if s.key=='jax_cfd');ctx=SimpleNamespace(cfg=problem,name=solver.name,phys=run['physics'],run=run,domain_extent=problem.domain_extent,make_inputs=problem.make_inputs)
receipt.update(recipe=run,devices=[str(d) for d in jax.devices()],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),path,Path(training.__file__)]})
captured={}
def fit(params,loss,validate,updates,lr,validation_interval):
    closure=inspect.getclosurevars(loss).nonlocals;vc=inspect.getclosurevars(validate).nonlocals
    data=closure['training_jax'];scales=closure['scales'];step=closure['step'];starts=closure['starts'];cases=closure['case_ids'];horizon=closure['horizon']
    receipt['data_hashes']={'train':ah(data),'validation':ah(vc['validation']),'scales':ah(scales)};receipt['schedule']={'starts':starts,'cases':cases};save_params('initial_params',params)
    np.savez(out/'training_data.npz',training=np.asarray(data),validation=vc['validation'],scales=np.asarray(scales))
    vg=jax.jit(jax.value_and_grad(lambda pp,target:training.window_loss(pp,target,step,scales)))
    from optimizers import build,check_muon
    if recipe['optimizer']=='muon':receipt['muon_partition_check']=check_muon(params)
    receipt['parameter_count']=sum(x.size for x in jax.tree.leaves(params))
    optimizer=build(params,recipe['optimizer'],lr,updates//receipt['batch_size'],recipe.get('cosine',False));state=optimizer.init(params)
    batch=receipt['batch_size'];history=[];checks=[];best=float('inf');selected=params;selected_at=0;train_s=val_s=0.0
    def evaluate(update):
        nonlocal best,selected,selected_at,val_s
        start=time.perf_counter();score=float(validate(params));val_s+=time.perf_counter()-start
        if not np.isfinite(score):raise FloatingPointError('nonfinite validation')
        checks.append({'update':update,'window_evaluations':update,'optimizer_steps':update//batch,'loss':score})
        if score<best:best=score;selected=params;selected_at=update;save_params('selected_params',selected)
        receipt.update(validation_checkpoints=checks,selected_window_evaluations=selected_at,training_seconds=train_s,validation_seconds=val_s);write();print(args.arm,update,score,flush=True)
    evaluate(0)
    for start_index in range(0,updates,batch):
        started=time.perf_counter();gradients=[];values=[]
        for i in range(start_index,start_index+batch):
            target=data[cases[i],starts[i]:starts[i]+horizon+1]
            value,gradient=vg(params,target);value=float(value)
            if not np.isfinite(value) or any(not np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(gradient)):raise FloatingPointError(f'nonfinite training window {i}')
            values.append(value);gradients.append(gradient)
        gradient=jax.tree.map(lambda *gs:sum(gs)/batch,*gradients)
        delta,state=optimizer.update(gradient,state,params);params=optax.apply_updates(params,delta)
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):raise FloatingPointError('nonfinite parameters')
        history.extend(values);train_s+=time.perf_counter()-started
        if (start_index+batch)%16==0:evaluate(start_index+batch)
    save_params('final_params',params);captured.update(params=selected,step=step,scales=scales)
    receipt.update(optimizer_steps=updates//batch,completed_window_evaluations=updates,training_loss=history);write()
    return selected,{'learning_rate':lr,'training_loss':history,'completed_updates':updates,'actual_optimizer_steps':updates//batch,'validation_loss':best,'selected_update':selected_at,'validation_checkpoints':checks,'training_seconds':train_s,'validation_seconds':val_s}
training.fit=fit

def diagnostics(reference):
    params,step,scales=captured['params'],captured['step'],captured['scales'];rows=[];arrays={};n=reference.shape[1]
    mask=np.ones((n,n,3),dtype=np.float32);mask[-1,:,0]=0;mask[:,-1,1]=0
    def S(z):return step(z*scales)/scales
    def C(z):return apply_corrector(params,z*scales,scales)/scales
    k=np.arange(n);qrad=np.sqrt(k[:,None]**2+k[None,:]**2)
    def stats(g):
        a=np.asarray(g,dtype=np.float64);power=dctn(a,axes=(0,1),norm='ortho')**2
        return {'norm':float(np.linalg.norm(a)),'canonical_dct_energy':{label:float(power[select].sum()) for label,select in [('dc',qrad==0),('large',(qrad>0)&(qrad<=4)),('medium',(qrad>4)&(qrad<=16)),('small',qrad>16)]}}
    def cosine(a,b):return float(np.vdot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    for frame in [25,50,75]:
        z=jnp.asarray(reference[frame])/scales;y,sp=jax.vjp(S,z);_,cp=jax.vjp(C,y);_,fp=jax.vjp(lambda zz:C(S(zz)),z)
        for seed in [3401,3402,3403]:
            q=np.random.default_rng(seed).normal(size=z.shape).astype(np.float32)*mask;q=jnp.asarray(q/np.linalg.norm(q))
            sg=sp(q)[0];cg=cp(q)[0];composed=sp(cg)[0];direct=fp(q)[0];residual=cg-q;effect=composed-sg
            grads={'solver':sg,'corrector':cg,'composed':composed,'corrector_residual':residual,'composed_effect':effect}
            row={'frame':frame,'probe_seed':seed,'q_sha256':ah(q),'state_sha256':ah(z),'statistics':{key:stats(g) for key,g in grads.items()},'cosine_solver_composed':cosine(sg,composed),'cosine_q_corrector':cosine(q,cg),'chainrule_relative_difference':float(np.linalg.norm(np.asarray(composed-direct))/max(np.linalg.norm(np.asarray(direct)),1e-30)),'finite_differences':[]}
            for label,F,point,g in [('solver',S,z,sg),('corrector',C,y,cg),('composed',lambda zz:C(S(zz)),z,composed)]:
                direction=g*jnp.asarray(mask);direction=direction/jnp.maximum(jnp.linalg.norm(direction),1e-30);ad=float(jnp.vdot(g,direction))
                for eps in [.003,.03]:
                    plus=np.asarray(F(point+eps*direction),dtype=np.float64);minus=np.asarray(F(point-eps*direction),dtype=np.float64);fd=float(np.sum((plus-minus)*np.asarray(q,dtype=np.float64))/(2*eps))
                    row['finite_differences'].append({'map':label,'epsilon':eps,'ad':ad,'fd':fd,'relative_error':abs(ad-fd)/max(abs(ad),abs(fd),1e-30)})
            for key,g in grads.items():arrays[f'f{frame}_seed{seed}_{key}']=np.asarray(g)
            rows.append(row)
    np.savez(out/'state_gradients.npz',**arrays);(out/'gradient_decomposition.json').write_text(json.dumps(clean({'coordinates':'canonical MAC packed state / per-channel training RMS; not cell-centered; unprojected canonical input sensitivities','probes':'three matched normalized random terminal cotangents, wall padding zero; no full-Jacobian conditioning claim','rows':rows}),indent=2)+'\n')
try:
    write();result=training.solver_in_loop(None,ctx);receipt['metrics']=result['metrics'];np.savez(out/'heldout_fields.npz',**result['snapshots']);receipt['training_complete']=True;write()
    if recipe.get('diagnostics',False):diagnostics(result['snapshots']['reference'])
    receipt['complete']=True;write()
except Exception as exc:
    receipt['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};write();raise
