"""Unmodified native API forward/VJP with normalized-coordinate wrappers."""
import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

class GPUAdapter:
    def __init__(self,solver,scales,physics):
        path=Path('mosaic/tesseracts/buoyancy-grid')/solver/'tesseract_api.py'
        spec=importlib.util.spec_from_file_location('native_api',path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
        self.api=api;self.scales=jnp.asarray(scales)
        self.fixed={'viscosity':jnp.asarray([physics['nu']],dtype=jnp.float32),'diffusivity':jnp.asarray([physics['diffusivity']],dtype=jnp.float32),'buoyancy':jnp.asarray([physics['buoyancy']],dtype=jnp.float32),'dt':jnp.asarray([physics['dt']],dtype=jnp.float32),'steps':int(physics['steps']),'domain_extent':1.0}
        self.inputs=lambda z:{**self.fixed,'state0':z*self.scales}
        self.forward_jax=jax.jit(lambda z:api.apply_jit(self.inputs(z))['result']/self.scales)
        self.single_vjp=jax.jit(lambda z,q:self.scales*api.vjp_jit(self.inputs(z),('state0',),('result',),{'result':q/self.scales})['state0'])
        self.batched_vjp=jax.jit(jax.vmap(self.single_vjp,in_axes=(None,0)))
        self.batched_forward=jax.jit(jax.vmap(self.forward_jax))
        self.use_batch=False
    def forward(self,z):return np.asarray(self.forward_jax(jnp.asarray(z)))
    def forward_many(self,Z):
        if self.use_batch:return np.asarray(self.batched_forward(jnp.asarray(Z)))
        return np.stack([self.forward(z) for z in Z])
    def vjp(self,z,Q):
        if self.use_batch:return np.asarray(self.batched_vjp(jnp.asarray(z),jnp.asarray(Q)))
        return np.stack([np.asarray(self.single_vjp(jnp.asarray(z),jnp.asarray(q))) for q in Q])
    def close(self):pass
