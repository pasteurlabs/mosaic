"""Fresh references only. Never imports or evaluates learned models."""
import hashlib,json,time,traceback,importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from mosaic.mosaic_shared.problems.buoyancy_grid import make_interacting_plumes_ic,restrict_state
from mosaic.benchmarks.problems.buoyancy_grid.training import rollout_errors,temperature_diagnostics,divergence_rms
root=Path(__file__).parent;out=root/'references';out.mkdir(exist_ok=True)
protocol=json.loads((root/'PROTOCOL.json').read_text());inventory=json.loads((root/'inventory.json').read_text());item=inventory['solvers']['jax-cfd']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
begun=time.monotonic()
record={'complete':False,'scope':'Fresh native solver reference/admission only. No corrector imports, no model evaluation. Models remain sealed until choice-freeze receipt.','seeds':protocol['seeds'],'rows':[],'devices':[str(d) for d in jax.devices()],'protocol_sha256':sha(root/'PROTOCOL.json'),'driver_sha256':sha(__file__)}
def save():
 record['elapsed_s']=time.monotonic()-begun;(out/'outcome.json').write_text(json.dumps(record,indent=2,allow_nan=False))
try:
 for entry in item['files'].values():assert sha(entry['path'])==entry['sha256']
 path=Path('mosaic/tesseracts/buoyancy-grid/jax-cfd/tesseract_api.py')
 assert sha(path)==next(v for k,v in item['source_sha256'].items() if k.endswith('/tesseract_api.py'))
 record['native_sha256']=sha(path)
 spec=importlib.util.spec_from_file_location('native_api',path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
 data=np.load(item['files'][item['data_file']]['path']);scales=data['scales'];assert digest(scales)==item['data_hashes']['scales']
 record['scales']=scales.tolist();record['scales_sha256']=digest(scales)
 def step(subdivision):
  fixed={'viscosity':jnp.asarray([.0005],jnp.float32),'diffusivity':jnp.asarray([.0005],jnp.float32),'buoyancy':jnp.asarray([1.],jnp.float32),'dt':jnp.asarray([.005/subdivision],jnp.float32),'steps':8*subdivision,'domain_extent':1.}
  return jax.jit(lambda x:api.apply_jit({**fixed,'state0':x})['result'])
 steps={v:step(v) for v in [1,2,4]}
 def trajectory(initial,subdivision):
  states=[np.asarray(initial)];x=jnp.asarray(initial)
  for _ in range(100):x=steps[subdivision](x);states.append(np.asarray(x))
  states=np.stack(states);assert np.isfinite(states).all()
  return states
 def phys(xs):
  td=temperature_diagnostics(xs)
  return {'temperature_min':float(xs[...,2].min()),'temperature_max':float(xs[...,2].max()),'final_heat_drift_relative':td['relative_integral_drift'][-1],'max_heat_drift_relative':max(abs(v) for v in td['relative_integral_drift']),'max_divergence_rms':max(divergence_rms(xs,1.))}
 for seed in protocol['generation_order']:
  t=time.monotonic();initial=make_interacting_plumes_ic(N=128,seed=seed)
  fine=trajectory(initial,2);reference=np.stack([restrict_state(x,2) for x in fine]);audit_native=trajectory(initial,4);audit=np.stack([restrict_state(x,2) for x in audit_native]);baseline=trajectory(reference[0],1)
  refinement_error=float(np.sqrt(np.mean(rollout_errors(audit,reference,scales)['loss'][1:])))
  coarse_gap=float(np.sqrt(np.mean(rollout_errors(baseline,reference,scales)['loss'][1:])))
  ratio=refinement_error/max(coarse_gap,1e-12)
  row={'seed':seed,'role':'validation' if seed in protocol['validation_seeds'] else 'sealed_fresh_test','initial_fine_sha256':digest(initial),'reference_sha256':digest(reference),'audit_sha256':digest(audit),'baseline_sha256':digest(baseline),'fine_sha256':digest(fine),'time_refinement_error':refinement_error,'coarse_gap':coarse_gap,'audit_gap_ratio':ratio,'admitted':bool(refinement_error<=.05 and np.isfinite(ratio) and ratio<=.25),'reference_physics':phys(reference),'baseline_physics':phys(baseline),'seconds':time.monotonic()-t}
  filename=out/f'ic-{seed}.npz';np.savez_compressed(filename,reference=reference,audit=audit,uncorrected=baseline,scales=scales,fine_initial=initial,fine_reference=fine)
  row['file']=str(filename);row['file_sha256']=sha(filename);record['rows'].append(row);save()
  valrows=[r for r in record['rows'] if r['role']=='validation']
  (root/'validation-manifest.json').write_text(json.dumps({'complete':len(valrows)==2 and all(r['admitted'] for r in valrows),'seeds':[5,6],'references':valrows,'reference_key':'reference','reference_shape':[101,64,64,3],'scales_sha256':record['scales_sha256'],'native_sha256':record['native_sha256'],'protocol_sha256':record['protocol_sha256']},indent=2))
  print(json.dumps(row),flush=True)
  assert row['admitted'],f'Fresh reference {seed} rejected by unchanged criteria'
 record['complete']=True;save()
except Exception as e:
 record['failure']={'message':str(e),'traceback':traceback.format_exc()};save();raise
