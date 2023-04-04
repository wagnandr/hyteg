from matplotlib import pyplot as plt
import json
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
symbols = ['o', 'x', '.']

fig, ax = plt.subplots(1, 1, sharey=True,)

for lidx, level in enumerate([6, 7]):
    try:
        with open(f'run_surrogate_lvl{level}_degree_blending_uniform.json') as f:
            data = json.loads(f.read())
    except:
        continue

    data = data['results']

    ilu_corr = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == True]

    surr_ilu_rates_corr = [d['rate'] for d in ilu_corr]
    surr_ilu_degrees_corr = [d['ilu_deg'][0] for d in ilu_corr]

    basic_ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu_basic]

    ax.semilogy(
        [min(surr_ilu_degrees_corr), max(surr_ilu_degrees_corr)],
        [basic_ilu_rates]*2,
        '--',
        color=colors[lidx],
        #label='matrix'
    )
    ax.semilogy(
        surr_ilu_degrees_corr,
        surr_ilu_rates_corr,
        '-'+symbols[lidx],
        color=colors[lidx],
        # label=r'surrogate'
        label=f'level {level}'
    )

    ax.set_xlabel(r'$dg_x = dg_y = dg_z$')
    ax.set_ylabel(r'$\rho$')
    #ax.set_ylim([8e-3, 3e-1])
    ax.set_ylim([4e-4, 1])
    ax.set_xticks([i for i in range(len(surr_ilu_rates_corr))])
    ax.grid(True)

plt.legend()
plt.tight_layout()
plt.show()
