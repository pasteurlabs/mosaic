"""Orthogonal physical closed-MAC Hodge projection, matching the CNN curl K.

Packed ux[-1,:] and uy[:,-1] are unused zero wall faces. Unpacked left/bottom
wall faces are also zero. psi has (N-1)^2 interior corners and zero boundary.
P=K(K.T K)^-1 K.T is Euclidean orthogonal; uniform cell-volume weighting
multiplies the metric by a scalar, leaving P unchanged. Do not impose heat-DC
removal: heat gradients generally have a physically meaningful constant mode.
"""
import numpy as np
from scipy.fft import dstn, idstn


def curl(psi):
    """K without 1/dx, exactly the corrector's corner-to-face difference."""
    psi=np.asarray(psi); p=np.pad(psi,[(0,0)]*(psi.ndim-2)+[(1,1),(1,1)])
    return np.stack((np.diff(p,axis=-1)[...,1:,:],-np.diff(p,axis=-2)[...,:,1:]),axis=-1)


def curl_transpose(g):
    g=np.asarray(g);u,v=g[...,0],g[...,1]
    return u[...,:-1,:-1]-u[...,:-1,1:]+v[...,1:,:-1]-v[...,:-1,:-1]


def project_velocity(g):
    """Accept (...,N,N,2) physical covectors; return same dtype and shape."""
    g=np.asarray(g); n=g.shape[-3]; assert g.shape[-2:]==(n,2)
    k=np.arange(1,n);e=4*np.sin(np.pi*k/(2*n))**2
    rhs=curl_transpose(g).astype(np.float64)
    psi=idstn(dstn(rhs,type=1,norm='ortho',axes=(-2,-1))/(e[:,None]+e[None,:]),type=1,norm='ortho',axes=(-2,-1))
    return curl(psi).astype(g.dtype)


def project_covector_normalized(g,scales):
    """g_z=D*g_x -> D*P(g_z/D), leaving heat covector unchanged."""
    g=np.asarray(g);d=np.asarray(scales);physical=g/d
    return np.concatenate((project_velocity(physical[...,:2]),physical[...,2:]),axis=-1)*d


def divergence(g,extent=1.):
    g=np.asarray(g);n=g.shape[-3];u,v=g[...,0],g[...,1]
    # Canonical last faces are padding, not physical flux. Report their energy
    # separately instead of pretending they are actual boundary velocities.
    u=np.concatenate((np.zeros_like(u[...,:1,:]),u[...,:-1,:],np.zeros_like(u[...,:1,:])),axis=-2)
    v=np.concatenate((np.zeros_like(v[...,:,:1]),v[...,:,:-1],np.zeros_like(v[...,:,:1])),axis=-1)
    return (np.diff(u,axis=-2)+np.diff(v,axis=-1))*n/extent


def physical_metrics(g_z,scales,extent=1.):
    """Unprojected physical coordinate covectors, not asserted fluid velocities."""
    g=np.asarray(g_z,dtype=np.float64)/np.asarray(scales,dtype=np.float64)
    v=g[...,:2];p=project_velocity(v);r=v-p;heat=g[...,2]
    e=float(np.sum(v*v));te=float(np.sum(heat*heat));n=g.shape[-3]
    padding=float(np.sum(v[...,-1,:,0]**2)+np.sum(v[...,:,-1,1]**2))
    dc=np.mean(heat,axis=(-2,-1),keepdims=True)
    return {'velocity_covector_norm':float(np.sqrt(e)), 'heat_covector_norm':float(np.sqrt(te)),
            'solenoidal_velocity_energy_fraction':float(np.sum(p*p)/e) if e else None,
            'orthogonal_complement_energy_fraction':float(np.sum(r*r)/e) if e else None,
            'padding_energy_fraction':padding/e if e else None,
            'divergence_rms':float(np.sqrt(np.mean(divergence(v,extent)**2))),
            'curl_rms':float(np.sqrt(np.mean((curl_transpose(v)*n/extent)**2))),
            'heat_dc_energy_fraction':float(np.sum(dc*dc)*n*n/te) if te else None,
            'heat_mean':float(np.mean(heat)),
            'heat_neumann_edge_difference_rms':float(np.sqrt(np.mean(np.concatenate([(heat[...,1,:]-heat[...,0,:]).ravel(),(heat[...,-1,:]-heat[...,-2,:]).ravel(),(heat[...,:,1]-heat[...,:,0]).ravel(),(heat[...,:,-1]-heat[...,:,-2]).ravel()])**2))),
            'decomposition_energy_relative_residual':abs(e-float(np.sum(p*p)+np.sum(r*r)))/e if e else 0.}


def project_velocity_jax(g):
    import jax.numpy as jnp
    n=g.shape[-3];u,v=g[...,0],g[...,1]
    rhs=u[...,:-1,:-1]-u[...,:-1,1:]+v[...,1:,:-1]-v[...,:-1,:-1]
    def dst(x,axis):
        x=jnp.moveaxis(x,axis,-1);z=jnp.zeros_like(x[...,:1]);odd=jnp.concatenate((z,x,z,-x[...,::-1]),axis=-1)
        return jnp.moveaxis(-jnp.fft.fft(odd,axis=-1).imag[...,1:n]/jnp.sqrt(2*n),-1,axis)
    k=jnp.arange(1,n,dtype=g.dtype);e=4*jnp.sin(jnp.pi*k/(2*n))**2
    psi=dst(dst(dst(dst(rhs,-1),-2)/(e[:,None]+e[None,:]),-1),-2)
    p=jnp.pad(psi,[(0,0)]*(psi.ndim-2)+[(1,1),(1,1)])
    return jnp.stack((jnp.diff(p,axis=-1)[...,1:,:],-jnp.diff(p,axis=-2)[...,:,1:]),axis=-1)


def project_covector_normalized_jax(g,scales):
    import jax.numpy as jnp
    physical=g/scales
    return jnp.concatenate((project_velocity_jax(physical[...,:2]),physical[...,2:]),axis=-1)*scales
