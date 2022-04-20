from matplotlib import pyplot as plt
import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate, kappa_type

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

with open('run_surrogate_degree_isotropic.json') as f:
    data = json.loads(f.read())
    data = data['results']

    ilu_corr = [[d for d in data if d['smoother'] == ilu_surrogate and d['kappa_type'] == kappa_type[i] and d['boundary_correction'] == True] for i in range(len(kappa_type))]
    ilu_boundary = [[d for d in data if d['smoother'] == ilu_surrogate and d['kappa_type'] == kappa_type[i] and d['boundary_correction'] == False] for i in range(len(kappa_type))]

    surr_ilu_rates_corr = [[d['rate'] for d in kd] for kd in ilu_corr]
    surr_ilu_degrees_corr = [[d['ilu_deg'][0] for d in kd] for kd in ilu_corr]

    surr_ilu_rates_boundary = [[d['rate'] for d in kd] for kd in ilu_boundary]
    surr_ilu_degrees_boundary = [[d['ilu_deg'][0] for d in kd] for kd in ilu_boundary]

    basic_ilu_rates = [[d['rate'] for d in data if d['smoother'] == ilu_basic and d['kappa_type'] == kappa_type[i]] for i in range(len(kappa_type))]

    fig, ax = plt.subplots(1,2, sharey=True)

    # with boundary stencil:
    for i in range(len(surr_ilu_rates_boundary)):
        label = r'matrix' if i == 0 else None
        ax[0].semilogy([min(surr_ilu_degrees_boundary[i]), max(surr_ilu_degrees_boundary[i])], basic_ilu_rates[i]*2, '--', color='tab:gray', label=label)

    for i in range(len(surr_ilu_rates_boundary)):
        ax[0].semilogy(surr_ilu_degrees_boundary[i], surr_ilu_rates_boundary[i], '-o', color=colors[i], label=r'surrogate $\kappa_' + f'{i}$')
        #plt.semilogy([min(surr_ilu_degrees[i]), max(surr_ilu_degrees[i])], basic_ilu_rates[i]*2, '--o', color=colors[i], label=r'matrix $deg(\kappa) = ' + f'{i} $')


    ax[0].set_ylabel(r'$\rho$')
    ax[0].set_xlabel(r'$dg_x = dg_y = dg_z$')
    ax[0].set_ylim([10**(-2), 1])
    ax[0].set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
    ax[0].grid(True)

    # with boundary stencil:
    for i in range(len(surr_ilu_rates_boundary)):
        label = r'matrix' if i == 0 else None
        ax[1].semilogy([min(surr_ilu_degrees_corr[i]), max(surr_ilu_degrees_corr[i])], basic_ilu_rates[i]*2, '--', color='tab:gray', label=label)

    for i in range(len(surr_ilu_rates_corr)):
        ax[1].semilogy(surr_ilu_degrees_corr[i], surr_ilu_rates_corr[i], '-o', color=colors[i], label=r'surrogate $\kappa_' + f'{i}$')
        #plt.semilogy([min(surr_ilu_degrees[i]), max(surr_ilu_degrees[i])], basic_ilu_rates[i]*2, '--o', color=colors[i], label=r'matrix $deg(\kappa) = ' + f'{i} $')


    #ax[1].set_ylabel(r'$\rho$')
    ax[1].set_xlabel(r'$dg_x = dg_y = dg_z$')
    ax[1].set_ylim([10**(-2), 1])
    ax[1].set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
    ax[1].grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

    '''
    # with boundary correction
    for i in range(len(surr_ilu_rates_corr)):
        label = r'matrix' if i == 0 else None
        plt.semilogy([min(surr_ilu_degrees_corr[i]), max(surr_ilu_degrees_corr[i])], basic_ilu_rates[i]*2, '--', color='tab:gray', label=label)

    for i in range(len(surr_ilu_rates_corr)):
        plt.semilogy(surr_ilu_degrees_corr[i], surr_ilu_rates_corr[i], '-o', color=colors[i], label=r'surrogate $\kappa_' + f'{i}$')
        #plt.semilogy([min(surr_ilu_degrees[i]), max(surr_ilu_degrees[i])], basic_ilu_rates[i]*2, '--o', color=colors[i], label=r'matrix $deg(\kappa) = ' + f'{i} $')

    plt.legend()
    plt.grid(True)

    ax[1].ylabel(r'$\rho$')
    ax[1].xlabel(r'$dg_x = dg_y = dg_z$')
    a.ylim([10**(-2), 1])
    plt.xticks([i for i in range(len(surr_ilu_rates_corr))])
    '''
