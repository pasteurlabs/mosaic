"""Explicit convolution-aware optimizers for this external comparison."""
import jax
import optax


def build(params, name, lr, steps, cosine=False):
    rate=optax.cosine_decay_schedule(lr,steps,alpha=.1) if cosine else lr
    if name=='adam': base=optax.adam(rate)
    elif name=='adamw':
        base=optax.adamw(rate,weight_decay=1e-4,mask=jax.tree.map(lambda x:x.ndim==4,params))
    elif name=='muon':
        dimensions=jax.tree.map(lambda x:optax.contrib.MuonDimensionNumbers(reduction_axis=(0,1,2),output_axis=(3,)) if x.ndim==4 else None,params)
        base=optax.contrib.muon(rate,weight_decay=0,adam_weight_decay=0,adam_learning_rate=.001,muon_weight_dimension_numbers=dimensions)
    else: raise ValueError(name)
    return optax.chain(optax.clip_by_global_norm(1.0),base)


def check_muon(params):
    import numpy as np
    opt=build(params,'muon',.005,256);state=opt.init(params)
    partition=state[1].inner_states
    muon_shapes=[list(x.shape) for x in jax.tree.leaves(partition['muon']) if getattr(x,'ndim',0)==4]
    adam_shapes=[list(x.shape) for x in jax.tree.leaves(partition['adam']) if getattr(x,'ndim',0)==4]
    expected=[list(x.shape) for x in jax.tree.leaves(params) if x.ndim==4]
    assert sorted(muon_shapes)==sorted(expected) and not adam_shapes,(muon_shapes,adam_shapes,expected)
    keys=iter(jax.random.split(jax.random.PRNGKey(9001),len(jax.tree.leaves(params))))
    gradient=jax.tree.map(lambda x:jax.random.normal(next(keys),x.shape,dtype=x.dtype),params)
    delta,state=opt.update(gradient,state,params)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(delta))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state))
    assert any(np.linalg.norm(np.asarray(x))>0 for x in jax.tree.leaves(delta) if x.ndim==4)
    return {'passed':True,'muon_conv_shapes':muon_shapes,'adam_conv_shapes':adam_shapes,'dimensions':'HWIO reduction(0,1,2),output(3)','random_gradient_update_finite':True,'bias_adam_learning_rate':.001}
