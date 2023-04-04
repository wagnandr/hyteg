from matplotlib import pyplot as plt
import json
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate, kappa_type

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
symbols = ['o', 'x', '.']

with open('run_surrogate_degree_isotropic.json') as f:
    data = json.loads(f.read())
    data = data['results']

    ilu_corr = [[d for d in data if d['smoother'] == ilu_surrogate and d['kappa_type'] == kappa_type[i] and d['boundary_correction'] == True] for i in range(len(kappa_type))]

    surr_ilu_rates_corr = [[d['rate'] for d in kd] for kd in ilu_corr]
    surr_ilu_degrees_corr = [[d['ilu_deg'][0] for d in kd] for kd in ilu_corr]

    basic_ilu_rates = [[d['rate'] for d in data if d['smoother'] == ilu_basic and d['kappa_type'] == kappa_type[i]] for i in range(len(kappa_type))]

    fig, axes = plt.subplots(1,1, sharey=True)

    ax = axes

    # with boundary stencil:
    for i in range(len(surr_ilu_rates_corr)):
        label = r'matrix' if i == 0 else None
        ax.semilogy([min(surr_ilu_degrees_corr[i]), max(surr_ilu_degrees_corr[i])], basic_ilu_rates[i]*2, '--', color='tab:gray', label=label)

    for i in range(len(surr_ilu_rates_corr)):
        ax.semilogy(surr_ilu_degrees_corr[i], surr_ilu_rates_corr[i], '-o', color=colors[i], label=r'surrogate $\kappa_' + f'{i}$')
        #plt.semilogy([min(surr_ilu_degrees[i]), max(surr_ilu_degrees[i])], basic_ilu_rates[i]*2, '--o', color=colors[i], label=r'matrix $deg(\kappa) = ' + f'{i} $')


    ax.set_ylabel(r'$\rho$')
    ax.set_xlabel(r'$dg_x = dg_y = dg_z$')
    ax.set_ylim([10**(-2), 1])
    ax.set_xticks([i for i in range(len(surr_ilu_rates_corr))])
    ax.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()
