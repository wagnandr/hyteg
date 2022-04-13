from matplotlib import pyplot as plt
import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular


with open('run_jump_with_height_mg.json')  as f:
    data = json.loads(f.read())
    data = data['results']
    gs = 'cell_gs'
    ilu = 'inplace_ldlt'

    gs_rates = [d['rate'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == gs]
    gs_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == gs]
    ilu_rates = [d['rate'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == ilu]

    plt.plot(gs_height, gs_rates, ':x', label=r'SGS $\kappa_{upper} = 10^{-5}$')
    plt.plot(ilu_height, ilu_rates, ':o', label=r'ILU $\kappa_{upper} = 10^{-5}$')

    gs_rates = [d['rate'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == gs]
    gs_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == gs]
    ilu_rates = [d['rate'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == ilu]

    plt.plot(gs_height, gs_rates, '-x', label=r'SGS $\kappa_{upper} = 1$')
    plt.plot(ilu_height, ilu_rates, '-o', label=r'ILU $\kappa_{upper} = 1$')

    gs_rates = [d['rate'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == gs]
    gs_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == gs]
    ilu_rates = [d['rate'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == ilu]

    plt.plot(gs_height, gs_rates, '--x', label=r'SGS $\kappa_{upper} = 10^5$')
    plt.plot(ilu_height, ilu_rates, '--o', label=r'ILU $\kappa_{upper} = 10^5$')

    plt.legend()
    plt.grid(True)

    plt.ylabel(r'$\rho$')
    plt.xlabel(r'$h_{upper}$')
    plt.show()

