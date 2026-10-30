"""Fixed head reparameterization; physical reporting loss remains unchanged."""
import jax.numpy as jnp
from mosaic.benchmarks.problems.buoyancy_grid.corrector import apply_corrector
def scaled_parameters(params,head_scales):
    if params[-1]['w'].shape[-1]!=len(head_scales):raise ValueError('readout/head scale mismatch')
    factors=jnp.asarray(head_scales,dtype=params[-1]['w'].dtype)
    return (*params[:-1],{**params[-1],'w':params[-1]['w']*factors,'b':params[-1]['b']*factors})
def apply_scaled_corrector(params,state,scales,head_scales):
    return apply_corrector(scaled_parameters(params,head_scales),state,scales)
