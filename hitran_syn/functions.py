import os, functools

import numpy as np
from scipy import constants

import hapi

from hapi import absorptionCoefficient_Generic, absorptionCoefficient_Priority, absorptionCoefficient_HT, absorptionCoefficient_SDVoigt, absorptionCoefficient_Voigt, absorptionCoefficient_Lorentz, absorptionCoefficient_Doppler

_default_db_path = os.path.join(os.path.expanduser("~"), "hitran-database")

available_molecules = [i[2] for i in hapi.ISO_ID.values()]

def download_hitran_data(molecule_name, db_path=_default_db_path):
  hapi.db_begin(db_path)
  if molecule_name in hapi.getTableList(): return

  try: table_entry, = filter(lambda i: i[2]==molecule_name, hapi.ISO_ID.values())
  except ValueError: raise Exception("invalid molecule name (check `available_molecules`)")
  M, I, _, _, _, _ = table_entry

  hapi.fetch(molecule_name,M,I,0,1e10)

def profile_ComplexLorentz(Nu,Gamma0,Delta0,WnGrid,YRosen=0.0,Sw=1.0):
  if not YRosen==0: raise NotImplementedError()

  difference = Nu-WnGrid-Delta0
  return Sw * (Gamma0+1j*difference)/np.pi / (difference**2 + Gamma0**2)

def profile_TwosidedComplexLorentz(Nu,Gamma0,Delta0,WnGrid,YRosen=0.0,Sw=1.0):
  return profile_ComplexLorentz(Nu,Gamma0,Delta0,WnGrid,YRosen,Sw) + profile_ComplexLorentz(-Nu,Gamma0,Delta0,WnGrid,YRosen,Sw)

def profile_ComplexVoigt(Nu,GammaD,Gamma0,Delta0,WnGrid,YRosen=0.0,Sw=1.0):
    re, im = hapi.pcqsdhc(Nu,GammaD,Gamma0,0j,Delta0,0j,0j,0j,WnGrid,YRosen)
    return Sw*(re - 1j*im)

def profile_TwosidedComplexVoigt(Nu,GammaD,Gamma0,Delta0,WnGrid,YRosen=0.0,Sw=1.0):
  return profile_ComplexVoigt(Nu,GammaD,Gamma0,Delta0,WnGrid,YRosen,Sw) + profile_ComplexVoigt(-Nu,GammaD,Gamma0,Delta0,WnGrid,YRosen,Sw)

def absorptionCoefficient_ComplexLorentz(*args, WavenumberGrid=None, **kwargs):
  return absorptionCoefficient_Generic(*args, **kwargs,
    profile=profile_ComplexLorentz,
    calcpars=hapi.calculateProfileParametersLorentz,
    WavenumberGrid=WavenumberGrid,
    initial_Xsect=np.zeros(WavenumberGrid.size, complex))

def absorptionCoefficient_TwosidedComplexLorentz(*args, WavenumberGrid=None, **kwargs):
  return absorptionCoefficient_Generic(*args, **kwargs,
    profile=profile_TwosidedComplexLorentz,
    calcpars=hapi.calculateProfileParametersLorentz,
    WavenumberGrid=WavenumberGrid,
    initial_Xsect=np.zeros(WavenumberGrid.size, complex))

def absorptionCoefficient_ComplexVoigt(*args, WavenumberGrid=None, **kwargs):
  return absorptionCoefficient_Generic(*args, **kwargs,
    profile=profile_ComplexVoigt,
    calcpars=hapi.calculateProfileParametersVoigt,
    WavenumberGrid=WavenumberGrid,
    initial_Xsect=np.zeros(WavenumberGrid.size, complex))

def absorptionCoefficient_TwosidedComplexVoigt(*args, WavenumberGrid=None, **kwargs):
  return absorptionCoefficient_Generic(*args, **kwargs,
    profile=profile_TwosidedComplexVoigt,
    calcpars=hapi.calculateProfileParametersVoigt,
    WavenumberGrid=WavenumberGrid,
    initial_Xsect=np.zeros(WavenumberGrid.size, complex))

def intensity_absorption_coefficient(molecule_name, nu, partial_pressure=101325, total_pressure=101325, temperature=293.15, backend=absorptionCoefficient_Lorentz, db_path=_default_db_path, wings_approximation=False):
  """
    Returns the real-valued quantity mu in I_out = exp(-mu*z) I_in, where I_out/I_in are the intensity spectra.

    molecule_name (string): an entry from `available_molecules`
    nu (1d ndarray): optical frequency grid, in Hz
    partial_pressure of species specified by `molecule name`: total pressure in Pascal (defaults to 101325 Pa = 1 atm)
    total_pressure: total pressure in Pascal (defaults to 101325 Pa = 1 atm)
    temperature: temperature in Kelvin (defaults to 293.15, i.e., 20°C)
  """

  if molecule_name not in available_molecules: raise ValueError("molecule_name {} not available, see `available_molecules` for a list".format(molecule_name))

  download_hitran_data(molecule_name, db_path=db_path)

  fraction = partial_pressure/total_pressure

  nutilde = nu/constants.c

  _, mu = backend(
    SourceTables=molecule_name,
    Environment={'p':total_pressure/1e5, 'T':temperature},
    Diluent={'self':fraction, 'air': 1-fraction},
    HITRAN_units=False,
    WavenumberGrid=nutilde/1e-2**-1,
    WavenumberWing=2*(nutilde.max()-nutilde.min())/1e-2**-1 if not wings_approximation else None
  )

  mu *= 1e-2**-1 * fraction # HAPI does not seem to multiply fraction automatically, so we do it

  return mu

def propagation_coefficient(*args, backend=absorptionCoefficient_TwosidedComplexLorentz, **kwargs):
  """
    Returns the complex-valued quantity gamma in A_out = exp(-gamma*z) A_in, where A_out/A_in are the complex amplitude spectra. Also known as "propagation constant".
    Takes the same arguments as `intensity_absorption_coefficient`.
  """

  mu = intensity_absorption_coefficient(*args, backend=backend, **kwargs)

  gamma = mu/2

  return gamma

# trick to get line data from hapi.absorptionCoefficient_Generic:
#  - set profile to a function that records line data
#  - set initialize_Xsect to an object that mimics ndarray in a limited way, but records line strengths as they are added
class ProfileParams(dict):
  def __rmul__(self, factor):
    return factor, dict(self)

class XsectList(object):
  def __init__(self):
    self.lines = []

  def __iadd__(self, value):
    self.lines.append(value)
    return self

  def __getitem__(self, *args): return self

  def __setitem__(self, *args): pass

  def __imul__(self, other):
    for i in range(len(self.lines)):
      factor, params = self.lines[i]
      self.lines[i] = factor*other, params
    return self

  def __repr__(self):
    return "XSectList(lines={})".format(repr(self.lines))

def linedata_backend(*args, WavenumberGrid=None, calpars=hapi.calculateProfileParametersLorentz, **kwargs):
  return absorptionCoefficient_Generic(*args, **kwargs,
    profile=ProfileParams,
    calcpars=calpars,
    WavenumberGrid=WavenumberGrid,
    initial_Xsect=XsectList())

def line_data(*args, calpars=hapi.calculateProfileParametersLorentz, raw=False, **kwargs):
  """
    returns line data, in a format compatible with `compose_lorentzians`
  """

  xsectlist_object = intensity_absorption_coefficient(*args, backend=functools.partial(linedata_backend,calpars=calpars), **kwargs)
  lines = xsectlist_object.lines

  if raw: return lines

  strengths = np.array([i[0]*i[1]['Sw'] for i in lines]) * constants.c*1e-2**-1
  nu0s = np.array([i[1]['Nu']*constants.c*1e-2**-1 for i in lines])
  widths = np.array([i[1]['Gamma0']*constants.c*1e-2**-1 for i in lines])

  return strengths, nu0s, widths

def reallorentz_frequencydomain(nu,nu0,width):
  return width/np.pi / ((nu-nu0)**2 + width**2 )

def imaglorentz_frequencydomain(nu,nu0,width):
  return -(nu-nu0)/np.pi / ((nu-nu0)**2 + width**2 )

def complexlorentz_frequencydomain(nu,nu0,width):
  return (width-1j*(nu-nu0))/np.pi / ((nu-nu0)**2 + width**2 )

def reallorentz_timedomain(t,nu0,width):
  return np.exp(-2*np.pi*width*abs(t)) * np.exp(1j*2*np.pi*nu0*t)

def complexlorentz_timedomain(t,nu0,width):
  return 2*np.exp(-2*np.pi*width*t) * np.exp(1j*2*np.pi*nu0*t) * (t>=0)

def twosided_reallorentz_frequencydomain(nu,nu0,width):
  return reallorentz_frequencydomain(nu,nu0,width) + reallorentz_frequencydomain(-nu,nu0,width)

def twosided_imaglorentz_frequencydomain(nu,nu0,width):
  return imaglorentz_frequencydomain(nu,nu0,width) - imaglorentz_frequencydomain(-nu,nu0,width)

def twosided_complexlorentz_frequencydomain(nu,nu0,width):
  return complexlorentz_frequencydomain(nu,nu0,width) + complexlorentz_frequencydomain(-nu,nu0,width).conj()

def twosided_reallorentz_timedomain(t,nu0,width):
  return 2*np.exp(-2*np.pi*width*abs(t)) * np.cos(2*np.pi*nu0*t)

def twosided_complexlorentz_timedomain(t,nu0,width):
  return 4*np.exp(-2*np.pi*width*t) * np.cos(2*np.pi*nu0*t) * (t>=0)

def compose_lorentzians(axis, strengths, nu0s, widths, backend=reallorentz_frequencydomain):
  ret = 0
  for strength, nu0, width in zip(strengths, nu0s, widths):
    ret = ret + strength*backend(axis, nu0, width)
  return ret
