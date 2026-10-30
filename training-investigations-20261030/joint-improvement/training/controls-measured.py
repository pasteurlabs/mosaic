"""Standalone correctness controls on training states only."""
from pathlib import Path
import json,hashlib
import jax
import jax.numpy as jnp
import numpy as np
from helper import apply_variant,effective_params
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector,apply_corrector,normalized_loss
from mosaic.benchmarks.problems.buoyancy_grid.training import window_loss
from gpu_adapter import GPUAdapter
root=Path(__file__).parent;item=json.loads((root/'inventory.json').read_text())['solvers']['jax-cfd'];data=np.load(item['files'][item['data_file']]['path']);D=jnp.asarray(data['scales']);state=jnp.asarray(data['training'][0,50]);q=jax.random.normal(jax.random.PRNGKey(1984),state.shape,dtype=jnp.float32);q/=jnp.linalg.norm(q)
adapter=GPUAdapter("jax-cfd",np.asarray(D),item["recipe"]["physics"]);step=jax.jit(lambda x:adapter.api.apply_jit({**adapter.fixed,"state0":x})["result"])
rows=[]
for joint in [False,True]:
 for center in [False,True]:
  for zero in ([False,True] if joint else [False]):
   recipe={'joint':joint,'center_temperature':center,'zero_mean_temperature':zero,'temperature_head_scale':.2139727633353161 if joint else 1}
   p=init_corrector(0,8,joint);assert np.array_equal(np.asarray(apply_variant(p,state,D,recipe)),np.asarray(state))
   probe=tuple({**x} for x in p);probe[-1]['w']=.01*jax.random.normal(jax.random.PRNGKey(17),probe[-1]['w'].shape,dtype=jnp.float32);probe[-1]['b']=jax.random.normal(jax.random.PRNGKey(42),probe[-1]['b'].shape,dtype=jnp.float32)*.01
   result,cd=apply_variant(probe,state,D,recipe,True);row={'recipe':recipe,'zero_head_exact_identity':True,'predicted_thermal_mean':float(cd[0]),'actual_thermal_mean':float(cd[2])}
   heatq=jnp.zeros_like(state).at[...,2].set(1/64**2);_,pb=jax.vjp(lambda x:apply_variant(probe,x,D,recipe),state);hg=pb(heatq)[0]
   row['heat_mean_pullback_relative_defect']=float(jnp.linalg.norm(hg-heatq)/jnp.linalg.norm(heatq))
   if zero or not joint:assert row['heat_mean_pullback_relative_defect']<1e-5
   if zero:
    assert abs(float(cd[0]))<1e-9
    gp=jax.grad(lambda pp:jnp.vdot(apply_variant(pp,state,D,recipe),q))(probe);row['thermal_uniform_bias_gradient']=float(gp[-1]['b'][1]);assert abs(row['thermal_uniform_bias_gradient'])<1e-5
   if center:
    offset=.1*float(D[2]);shift=state.at[...,2].add(offset);shifted=apply_variant(probe,shift,D,recipe);expected=result.at[...,2].add(offset)
    row['offset_covariance_max_defect']=float(jnp.max(abs(shifted-expected)));assert row['offset_covariance_max_defect']<1e-6
    _,rp=jax.vjp(lambda x:apply_variant(probe,x,D,recipe)-x,state);rg=rp(q)[0];row['residual_temperature_covector_sum']=float(jnp.sum(rg[...,2]));assert abs(row['residual_temperature_covector_sum'])<1e-5
   if not center and not zero:
    ep=effective_params(probe,recipe);source=apply_corrector(ep,state,D);row['source_forward_bitwise_equal']=np.array_equal(np.asarray(source),np.asarray(result));assert row['source_forward_bitwise_equal']
    g1=jax.grad(lambda pp:jnp.vdot(apply_variant(pp,state,D,recipe),q))(probe);g2=jax.grad(lambda pp:jnp.vdot(apply_corrector(effective_params(pp,recipe),state,D),q))(probe)
    row['source_parameter_gradient_max_difference']=max(float(jnp.max(abs(a-b))) for a,b in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)));assert row['source_parameter_gradient_max_difference']==0
    def own_loss(pp):
     x=jnp.asarray(data['training'][0,0]);value=jnp.asarray(0.0)
     for target in jnp.asarray(data['training'][0,1:9]):
      x=apply_variant(pp,step(x),D,recipe);value=value+normalized_loss(x,target,D)
     return value/8
    def source_loss(pp):return window_loss(effective_params(pp,recipe),jnp.asarray(data['training'][0,:9]),step,D)
    v1,gr1=jax.value_and_grad(own_loss)(probe);v2,gr2=jax.value_and_grad(source_loss)(probe)
    row['source_H8_forward_equal']=np.array_equal(np.asarray(v1),np.asarray(v2));assert row['source_H8_forward_equal']
    row['source_H8_gradient_max_difference']=max(float(jnp.max(abs(a-b))) for a,b in zip(jax.tree.leaves(gr1),jax.tree.leaves(gr2)))
    row['source_H8_gradient_relative_difference']=float(jnp.sqrt(sum(jnp.sum((a-b)**2) for a,b in zip(jax.tree.leaves(gr1),jax.tree.leaves(gr2))))/jnp.maximum(jnp.sqrt(sum(jnp.sum(a*a) for a in jax.tree.leaves(gr2))),1e-30))
    print('MeasuredH8gradientParity',recipe,row['source_H8_gradient_max_difference'],row['source_H8_gradient_relative_difference'],flush=True)

   rows.append(row)
out=root/'controls';out.mkdir(exist_ok=True);(out/'controls-measured.json').write_text(json.dumps({'complete':True,'rows':rows,'training_state_sha256':hashlib.sha256(np.asarray(state).tobytes()).hexdigest(),'helper_sha256':hashlib.sha256((root/'helper.py').read_bytes()).hexdigest(),'devices':[str(x) for x in jax.devices()]},indent=2));print(json.dumps(rows,indent=2))
