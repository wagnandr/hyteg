from matplotlib import pyplot as plt
import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular


with open('run_height_single_tetrahedron.json') as f:
    data = json.loads(f.read())
    data = data['results']
    gs = 'cell_gs'
    ilu = 'inplace_ldlt'

    fig, axes = plt.subplots(1,3, sharex=True)

    gs_rates = [d['rate'] for d in data if d['smoother'] == gs]
    gs_height = [d['height'] for d in data if d['smoother'] == gs]
    ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if d['smoother'] == ilu]

    axes[0].plot(gs_height, gs_rates, '-x', label=r'SGS')
    axes[0].plot(ilu_height, ilu_rates, '-o', label=r'ILU')

    axes[1].semilogy(gs_height, gs_rates, '-x', label=r'SGS')
    axes[1].semilogy(ilu_height, ilu_rates, '-o', label=r'ILU')

    axes[2].semilogy(gs_height, np.log(ilu_rates)/np.log(gs_rates), '-s', color='black', label=r'SGS')


    axes[0].set_xlabel(r'$h$')
    axes[1].set_xlabel(r'$h$')
    axes[2].set_xlabel(r'$h$')
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    axes[0].set_ylabel(r'$\rho$')
    axes[1].set_ylabel(r'$\rho$')
    axes[2].set_ylabel(r'$\log(\rho_{ILU}) \,/\, \log(\rho_{SGS})$')

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    plt.tight_layout()
    plt.show()

