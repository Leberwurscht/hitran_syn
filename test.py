import itertools

import numpy as np
import matplotlib.pyplot as plt

import jax.config
jax.config.update("jax_enable_x64", True)

import fourioso, fourioso.jax

def maxrelerror(data, data_reference):
  return np.max(abs(data-data_reference))/np.max(abs(data_reference))

def gaussian(t, alpha, t0=0, additional_linear_phase=0):
  return np.exp(-alpha * (t-t0)**2) * np.exp(1j*additional_linear_phase*(t-t0))

def transformed_gaussian(f, alpha, t0=0, additional_linear_phase=0):
  f0 = additional_linear_phase/2/np.pi
  return np.sqrt(np.pi/alpha) * np.exp(-(np.pi*(f-f0))**2/alpha) * np.exp(-1j*2*np.pi*f*t0) # http://en.wikipedia.org/wiki/Fourier_transform (206)

for backend, parity in itertools.product([fourioso, fourioso.jax], [0,1]):
  # TO DO: test additional_linear_phase

  n, t_spacing = backend.n_spacing(.05, 30)
  n += parity
  t = backend.get_axis(n, t_spacing)

  f_spacing = 1/t_spacing/n
  f_reference = backend.get_axis(n, f_spacing)

  # test transform(gaussian) vs analytical solution
  alpha = .2
  t0 = .3
  t = backend.get_axis(n, t_spacing)
  y = gaussian(t, alpha, t0)
  f, Y = backend.transform(t, y)
  assert maxrelerror(f, f_reference)<1e-10
  Y_reference = transformed_gaussian(f, alpha, t0)
  assert maxrelerror(Y, Y_reference)<1e-10

  # test transform(gaussian, -1) vs analytical solution
  alpha = .2
  t0 = .3
  Y = transformed_gaussian(f, alpha, t0)
  t_, y = backend.transform(f, Y, -1)
  assert maxrelerror(t_, t)<1e-10
  y_reference = gaussian(t, alpha, t0)
  assert maxrelerror(Y, Y_reference)<1e-10

  # test transform(gaussian, 2) vs analytical solution
  alpha = .2
  t0 = .3
  Y_reference = gaussian(-t, alpha, t0)
  t_, Y = backend.transform(t, y, 2)
  assert maxrelerror(t_, t)<1e-10
  assert maxrelerror(Y, Y_reference)<1e-10

  # test transform(gaussian, 0) vs analytical solution
  alpha = .2
  t0 = .3
  y_reference = gaussian(t, alpha, t0)
  t_, y = backend.transform(t, y_reference, 0)
  assert maxrelerror(t_, t)<1e-10
  assert maxrelerror(y, y_reference)<1e-10
