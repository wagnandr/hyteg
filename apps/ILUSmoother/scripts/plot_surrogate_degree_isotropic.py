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

    surr_ilu_rates = [[d['rate'] for d in data if d['smoother'] == ilu_surrogate and d['kappa_type'] == kappa_type[i]] for i in range(len(kappa_type))]
    surr_ilu_degrees = [[d['ilu_deg'][0] for d in data if d['smoother'] == ilu_surrogate and d['kappa_type'] == kappa_type[i]] for i in range(len(kappa_type))]
    basic_ilu_rates = [[d['rate'] for d in data if d['smoother'] == ilu_basic and d['kappa_type'] == kappa_type[i]] for i in range(len(kappa_type))]

    for i in range(len(surr_ilu_rates)):
        label = r'matrix' if i == 0 else None
        plt.semilogy([min(surr_ilu_degrees[i]), max(surr_ilu_degrees[i])], basic_ilu_rates[i]*2, '--', color='tab:gray', label=label)

    for i in range(len(surr_ilu_rates)):
        plt.semilogy(surr_ilu_degrees[i], surr_ilu_rates[i], '-o', color=colors[i], label=r'surrogate $\kappa_' + f'{i}$')
        #plt.semilogy([min(surr_ilu_degrees[i]), max(surr_ilu_degrees[i])], basic_ilu_rates[i]*2, '--o', color=colors[i], label=r'matrix $deg(\kappa) = ' + f'{i} $')


    plt.legend()
    plt.grid(True)

    plt.ylabel(r'$\rho$')
    plt.xlabel(r'$dg_x = dg_y = dg_z$')
    plt.ylim([10**(-2), 1])
    plt.xticks([i for i in range(len(surr_ilu_rates))])
    plt.show()

