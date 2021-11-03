import numpy as np
from scipy.interpolate import interp1d
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

import utils

input_spectra = 'dilution_tspec.dat'
simulation_folder = 'pandexo_simulations'

files = ['lrs.p', 'niriss.p']
names = ['MIRI/LRS', 'NIRISS/SOSS']
nbins = [4, 25]
colors = ['orangered', 'royalblue']

# Read in input transit spectrum:
w, d = np.loadtxt(input_spectra, unpack=True)

# Create interpolator:
f = interp1d(w, d * 1e6)

plt.figure(figsize = (10,3))

plt.plot(w, d * 1e6, color = 'black', lw = 3, label = 'HAT-P-14b (diluted) spectrum')

# Go through instruments, creating mock datasets:
for i in range(len(files)):

    filename, name, nbin = files[i], names[i], nbins[i]

    simulation = pickle.load(open(simulation_folder+'/'+filename, 'rb'))

    sim_w, sim_error = simulation['FinalSpectrum']['wave'], \
                       simulation['FinalSpectrum']['error_w_floor']

    sim_model = f(sim_w) 
    sim_data = np.zeros(len(sim_model))

    for j in range(len(sim_w)):

        sim_data[j] = np.random.normal(sim_model[j], sim_error[j] * 1e6, 1) 

    # Plot it:
    plt.errorbar(sim_w, sim_data, sim_error * 1e6, fmt = '.', ms = 1, elinewidth = 1, label = None, color = 'grey', alpha = 0.2, rasterized=True)

    # Plot binned data:
    xbin, ybin, ybinerr = utils.bin_data(sim_w, sim_data, sim_error * 1e6, nbin)
    plt.errorbar(xbin, ybin, ybinerr, fmt = 'o', ms = 4, label = name, mfc = 'white', elinewidth=1.5, mec = colors[i], ecolor = colors[i])

plt.xlabel('Wavelength (microns)', fontsize = 13)
plt.ylabel('Transit depth (ppm)', fontsize = 13)
plt.xlim(0.835, 10)
plt.ylim(6750-500, 6750+500)
plt.xscale('log')

ax=plt.gca()
fmt = mpl.ticker.StrMethodFormatter("{x:g}")
ax.xaxis.set_major_formatter(fmt)
ax.xaxis.set_minor_formatter(fmt)

#ax.tick_params(axis='both', labelsize = 14)

plt.legend()
plt.tight_layout()
plt.savefig('hat-p-14-prediction.pdf')
#plt.show()
