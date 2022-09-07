from matplotlib import pyplot as plt
import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular


with open('run_jump_with_height_cg.json')  as f:
    data = json.loads(f.read())
    data = data['results']
    gs = 'cell_gs'
    ilu = 'inplace_ldlt'

    fig, axes = plt.subplots(2, 1, sharex=True)

    gs_convergence_numbers = [d['convergence_number'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == gs]
    gs_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == gs]
    ilu_convergence_numbers = [d['convergence_number'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e-5) and d['smoother'] == ilu]

    axes[0].semilogy(gs_height, gs_convergence_numbers, ':x', label=r'SGS $\kappa_{upper} = 10^{-5}$')
    axes[0].semilogy(ilu_height, ilu_convergence_numbers, ':o', label=r'ILU $\kappa_{upper} = 10^{-5}$')

    axes[1].plot(ilu_height, np.array(gs_convergence_numbers)/np.array(ilu_convergence_numbers), ':', color='black', label=r'$\kappa_{upper} = 10^{-5}$')

    gs_convergence_numbers = [d['convergence_number'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == gs]
    gs_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == gs]
    ilu_convergence_numbers = [d['convergence_number'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1.0) and d['smoother'] == ilu]

    axes[0].semilogy(gs_height, gs_convergence_numbers, '-x', label=r'SGS $\kappa_{upper} = 1$')
    axes[0].semilogy(ilu_height, ilu_convergence_numbers, '-o', label=r'ILU $\kappa_{upper} = 1$')
    
    axes[1].plot(ilu_height, np.array(gs_convergence_numbers)/np.array(ilu_convergence_numbers), '-', color='black', label=r'$\kappa_{upper} = 1$')

    gs_convergence_numbers = [d['convergence_number'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == gs]
    gs_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == gs]
    ilu_convergence_numbers = [d['convergence_number'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if np.isclose(d['kappa_upper'], 1e5) and d['smoother'] == ilu]

    axes[0].semilogy(gs_height, gs_convergence_numbers, '--x', label=r'SGS $\kappa_{upper} = 10^5$')
    axes[0].semilogy(ilu_height, ilu_convergence_numbers, '--o', label=r'ILU $\kappa_{upper} = 10^5$')

    axes[1].plot(ilu_height, np.array(gs_convergence_numbers)/np.array(ilu_convergence_numbers), '--', color='black', label=r'$\kappa_{upper} = 10^{5}$')

    axes[0].legend()
    axes[0].grid(True)

    #axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim((1,1.4))


    axes[0].set_ylabel(r'iterations')
    axes[1].set_ylabel(r'$iter_{SGS} \,/\, iter_{ILU}$')

    axes[1].set_aspect(0.25)

    plt.tight_layout()

    plt.xlabel(r'$h_{lower}$')
    plt.show()

