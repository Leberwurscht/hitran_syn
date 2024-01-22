Allows to synthesize complex-valued absorption spectra from the hitran database, using HAPI.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

import joblib
memory = joblib.Memory("cache")

import hitran_syn # pip install git+https://gitlab.com/leberwurscht/hitran_syn.git

### list available molecules from HITRAN
print("available molecules:")
for i in hitran_syn.available_molecules:
  print("\t", i)
print()

### compute propagation coefficient
@memory.cache # takes some time (~1 min on my computer), so cache the result with joblib.Memory
def get_data(nu_min, nu_max):
  nu = np.linspace(nu_min,nu_max,25000)

  pressure = 101325 # 1 atm in Torr
  temperature = 273+20 # 20°C in Kelvin
  fraction = 100e-6 # 100 ppm
  molecule_name = "(12C)H3(16O)H" # ... of methanol

  gamma = hitran_syn.propagation_coefficient(molecule_name,nu,total_pressure=pressure,partial_pressure=fraction*pressure,temperature=temperature)

  return nu, gamma

### plot
nu, gamma = get_data(20e12, 40e12)
sample_length = 1 # in meters
transmission_coefficient = np.exp(-gamma*sample_length)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,1)
ax = fig.add_subplot(gs[0,0])
ax.plot(nu/1e12, abs(transmission_coefficient)**2 /  1e-2)
ax.set_xlabel("optical frequency (THz)")
ax.set_ylabel(r"PSD$_\mathrm{out}$/PSD$_\mathrm{in}$ (%)")
ax = fig.add_subplot(gs[1,0])
ax.plot(nu/1e12, np.angle(transmission_coefficient))
ax.set_xlabel("optical frequency (THz)")
ax.set_ylabel("spectral phase change (rad)")

fig.suptitle("transmission through {} meters of 100 ppm methanol at 1 atm, 20°C".format(sample_length))
plt.savefig("output.png")
plt.show()
```

![Output](output.png)
