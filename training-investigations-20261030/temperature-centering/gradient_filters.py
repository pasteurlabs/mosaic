"""Optional surrogate reverse filters; these are NOT derivatives of forward map.

ell_cells is a dimensionless smoothing radius measured in grid cells. Filtering
is a proposed optimization intervention, not a correction to valid native AD.
"""
import jax.numpy as jnp


def velocity_h1(g,ell_cells=1.):
    """K [L(I+ell_cells²L)]^-1 K.T g, L=K.T K with zero-wall psi."""
    n=g.shape[-3];u,v=g[...,0],g[...,1]
    rhs=u[...,:-1,:-1]-u[...,:-1,1:]+v[...,1:,:-1]-v[...,:-1,:-1]
    def dst(x,axis):
        x=jnp.moveaxis(x,axis,-1);z=jnp.zeros_like(x[...,:1]);odd=jnp.concatenate((z,x,z,-x[...,::-1]),axis=-1)
        return jnp.moveaxis(-jnp.fft.fft(odd,axis=-1).imag[...,1:n]/jnp.sqrt(2*n),-1,axis)
    k=jnp.arange(1,n,dtype=g.dtype);e=4*jnp.sin(jnp.pi*k/(2*n))**2;lam=e[:,None]+e[None,:]
    psi=dst(dst(dst(dst(rhs,-1),-2)/(lam*(1+ell_cells**2*lam)),-1),-2)
    p=jnp.pad(psi,[(0,0)]*(psi.ndim-2)+[(1,1),(1,1)])
    return jnp.stack((jnp.diff(p,axis=-1)[...,1:,:],-jnp.diff(p,axis=-2)[...,:,1:]),axis=-1)


def heat_h1(g,ell_cells=1.):
    """Cell-centered Neumann inverse(I+ell_cells²L), preserving mean/DC."""
    n=g.shape[-1];k=jnp.arange(n,dtype=g.dtype);j=jnp.arange(n,dtype=g.dtype)
    c=jnp.cos(jnp.pi*k[:,None]*(j[None,:]+.5)/n)*jnp.sqrt(2/n)
    c=c.at[0].multiply(1/jnp.sqrt(2.))
    coeff=jnp.einsum('ki,...ij,lj->...kl',c,g,c)
    e=4*jnp.sin(jnp.pi*k/(2*n))**2
    coeff=coeff/(1+ell_cells**2*(e[:,None]+e[None,:]))
    return jnp.einsum('ki,...kl,lj->...ij',c,coeff,c)


def physical_h1(g,ell_cells=1.):
    return jnp.concatenate((velocity_h1(g[...,:2],ell_cells),heat_h1(g[...,2],ell_cells)[...,None]),axis=-1)


def normalized_h1(g,scales,ell_cells=1.):
    return scales*physical_h1(g/scales,ell_cells)
