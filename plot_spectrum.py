import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')

import utils

transit_depth = 6797 # ppm
delta_j = 5.5

# Compute spectrum of target:
target_wavelength, target_flux = utils.get_stellar_model(teff=6671, logg=4.187, feh=0.11, jmag=9.094)
companion_wavelength, companion_flux = utils.get_stellar_model(teff=3200, logg=5.0, feh=0.11, jmag=9.094 + delta_j)

# Plot dilution spectrum:
factor = 1. / (1. + companion_flux/target_flux)
plt.plot(target_wavelength, transit_depth - (transit_depth * factor))
plt.show()

# Save (dilution) spectrum:
wavelength = target_wavelength.value
tspec = transit_depth * factor.value * 1e-6
fout = open('dilution_tspec.dat', 'w')

for i in range(len(target_wavelength)):

    if not np.isnan(tspec[i]):

        fout.write('{0:.5f} {1:.10f}\n'.format(wavelength[i], tspec[i]))

fout.close()
