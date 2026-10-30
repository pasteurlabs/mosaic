from pathlib import Path
import json,importlib.util,hashlib
import jax
import jax.numpy as jnp
import numpy as np
from mosaic.benchmarks.problems.buoyancy_grid.corrector import init_corrector,apply_corrector,normalized_loss
from mosaic.benchmarks.problems.buoyancy_grid.training import window_loss
root=Path(__file__).parent
old=Path('/data/personal/andrinr/runner/results/mosaic/buoyancy-joint-improvement-20261030')
spec=importlib.util.spec_from_file_location('external_helper',old/'helper.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
item=json.loads((old/'inventory.json').read_text())['solvers']['jax-cfd']
data=np.load(item['files'][item['data_file']]['path']);D=jnp.asarray(data['scales'])
z=np.load(old/'train/joint-c0-m0-t0.1-lr0.001-s0/selected_params.npz')
template=init_corrector(0,8,True);params=jax.tree.unflatten(jax.tree.structure(template),[jnp.asarray(z[f'leaf_{i}']) for i in range(len(z.files))])
recipe={'joint':True,'temperature_head_scale':.1,'center_temperature':False,'zero_mean_temperature':False}
def production(pp,x):return apply_corrector(pp,x,D,temperature_head_scale=.1)
def external(pp,x):return helper.apply_variant(pp,x,D,recipe)
q=jax.random.normal(jax.random.PRNGKey(1984),(64,64,3),dtype=jnp.float32);rows=[]
for frame in [0,50,100]:
 x=jnp.asarray(data['training'][0,frame]);p=production(params,x);e=external(params,x)
 assert np.array_equal(np.asarray(p),np.asarray(e))
 a=jax.grad(lambda pp:jnp.vdot(production(pp,x),q))(params);b=jax.grad(lambda pp:jnp.vdot(external(pp,x),q))(params)
 diff=max(float(jnp.max(abs(aa-bb))) for aa,bb in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True))
 assert diff==0
 rows.append({'training_frame':frame,'forward_bitwise_equal':True,'parameter_pullback_max_abs_difference':diff})
target=jnp.asarray(data['training'][0,:9])
def reference_loss(pp):
 x=target[0];value=jnp.asarray(0.0)
 for t in target[1:]:
  x=external(pp,.99*x);value+=normalized_loss(x,t,D)
 return value/8
p,pg=jax.value_and_grad(lambda pp:window_loss(pp,target,lambda x:.99*x,D,temperature_head_scale=.1))(params)
e,eg=jax.value_and_grad(reference_loss)(params)
delta=max(float(jnp.max(abs(aa-bb))) for aa,bb in zip(jax.tree.leaves(pg),jax.tree.leaves(eg),strict=True))
assert float(p)==float(e) and delta==0
report={'complete':True,'rows':rows,'H8_scalar_loss_bitwise_equal':True,'H8_parameter_gradient_max_abs_difference':delta,'scope':'Same trained parameters/three training states and fixed linear recurrence; no solver or heldout score selected. Production optional scale reproduces external research helper.'}
(root/'parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
