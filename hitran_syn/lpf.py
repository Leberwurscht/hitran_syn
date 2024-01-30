"""
Complex voigt profile (cvoigt), with GPU acceleration provided by the JAX library.

This file is "stolen" and adapted from the exojax library (MIT license).

Copyright (c) 2024 Maximilian Högner
Copyright (c) 2021 Hajime Kawahara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Custom JVP version of the line profile functions used in exospectral
analysis."""

from jax import jit, vmap
import jax.numpy as jnp
from .faddeeva import rewofz, imwofz
from .faddeeva import asymptotic_wofz
from jax import custom_jvp

import warnings

@jit
def ljert(x, a):
    """ljert function, consisting of a combination of imwofz and imag(asymptiotic wofz).

    Args:
        x:
        a:

    Returns:
        L(x,a) or Imag(wofz(x+ia))

    Note:
        ljert provides a L(x,a) function. This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.
    """
    r2 = x * x + a * a
    return jnp.where(r2 < 111., imwofz(x, a), jnp.imag(asymptotic_wofz(x, a)))


@custom_jvp
def hjert(x, a):
    """custom JVP version of the Voigt-Hjerting function, consisting of a
    combination of rewofz and real(asymptotic wofz).

    Args:
        x: 
        a:

    Returns:
        H(x,a) or Real(wofz(x+ia))

    Examples:

       hjert provides a Voigt-Hjerting function w/ custom JVP. 

       >>> hjert(1.0,1.0)
          DeviceArray(0.30474418, dtype=float32)

       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.

       >>> from jax import vmap
       >>> x=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,None),0)(x,1.0)
          DeviceArray([0.42758358, 0.42568347, 0.4200511 , 0.41088563, 0.39850432,0.3833214 , 0.3658225 , 0.34653533, 0.32600054, 0.3047442 ],dtype=float32)
       >>> a=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,0),0)(x,a)
          DeviceArray([1.        , 0.8764037 , 0.7615196 , 0.6596299 , 0.5718791 ,0.49766064, 0.43553388, 0.3837772 , 0.34069115, 0.3047442 ],dtype=float32)
    """
    r2 = x * x + a * a
    return jnp.where(r2 < 111., rewofz(x, a), jnp.real(asymptotic_wofz(x, a)))


@hjert.defjvp
def hjert_jvp(primals, tangents):
    x, a = primals
    ux, ua = tangents
    dHdx = 2.0 * a * ljert(x, a) - 2.0 * x * hjert(x, a)
    dHda = 2.0 * x * ljert(x, a) + 2.0 * a * hjert(x, a) - 2.0 / jnp.sqrt(
        jnp.pi)
    primal_out = hjert(x, a)
    tangent_out = dHdx * ux + dHda * ua
    return primal_out, tangent_out

#@custom_jvp # TODO
def imhjert(x, a):
    """custom JVP version of the Voigt-Hjerting function, consisting of a
    combination of imwofz and imag(asymptotic wofz).

    Args:
        x: 
        a:

    Returns:
        H(x,a) or Real(wofz(x+ia))

    Examples:

       hjert provides a Voigt-Hjerting function w/ custom JVP. 

       >>> hjert(1.0,1.0)
          DeviceArray(0.30474418, dtype=float32)

       This function accepts a scalar value as an input. Use jax.vmap to use a vector as an input.

       >>> from jax import vmap
       >>> x=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,None),0)(x,1.0)
          DeviceArray([0.42758358, 0.42568347, 0.4200511 , 0.41088563, 0.39850432,0.3833214 , 0.3658225 , 0.34653533, 0.32600054, 0.3047442 ],dtype=float32)
       >>> a=jnp.linspace(0.0,1.0,10)
       >>> vmap(hjert,(0,0),0)(x,a)
          DeviceArray([1.        , 0.8764037 , 0.7615196 , 0.6596299 , 0.5718791 ,0.49766064, 0.43553388, 0.3837772 , 0.34069115, 0.3047442 ],dtype=float32)
    """
    r2 = x * x + a * a
    return jnp.where(r2 < 111., imwofz(x, a), jnp.imag(asymptotic_wofz(x, a)))

@jit
def voigtone(nu, sigmaD, gammaL):
    """Custom JVP version of (non-vmapped) Voigt function using Voigt-Hjerting
    function.

    Args:
       nu: wavenumber
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt funtion
    """

    sfac = 1.0 / (jnp.sqrt(2) * sigmaD)
    v = sfac * hjert(sfac * nu, sfac * gammaL) / jnp.sqrt(jnp.pi)
    return v


@jit
def voigt(nuvector, sigmaD, gammaL):
    """Custom JVP version of Voigt profile using Voigt-Hjerting function.

    Args:
       nu: wavenumber array
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt profile
    """

    sfac = 1.0 / (jnp.sqrt(2.0) * sigmaD)
    vhjert = vmap(hjert, (0, None), 0)
    v = sfac * vhjert(sfac * nuvector, sfac * gammaL) / jnp.sqrt(jnp.pi)
    return v

@jit
def imvoigt(nuvector, sigmaD, gammaL):
    """Custom JVP version of Voigt profile using Voigt-Hjerting function.

    Args:
       nu: wavenumber array
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt profile
    """

    sfac = 1.0 / (jnp.sqrt(2.0) * sigmaD)
    imvhjert = vmap(imhjert, (0, None), 0)
    v = sfac * imvhjert(sfac * nuvector, sfac * gammaL) / jnp.sqrt(jnp.pi)
    return v

@jit
def cvoigt(nuvector, sigmaD, gammaL):
    """Custom JVP version of Voigt profile using Voigt-Hjerting function.

    Args:
       nu: wavenumber array
       sigmaD: sigma parameter in Doppler profile
       gammaL: broadening coefficient in Lorentz profile

    Returns:
       v: Voigt profile
    """

    return voigt(nuvector, sigmaD, gammaL) + 1j*imvoigt(nuvector, sigmaD, gammaL)


@jit
def vvoigt(numatrix, sigmaD, gammas):
    """Custom JVP version of vmaped voigt profile.

    Args:
       numatrix: wavenumber matrix in R^(Nline x Nwav)
       sigmaD: doppler sigma vector in R^Nline
       gammaL: gamma factor vector in R^Nline

    Return:
       Voigt profile vector in R^Nwav
    """
    vmap_voigt = vmap(voigt, (0, 0, 0), 0)
    return vmap_voigt(numatrix, sigmaD, gammas)
