from matplotlib import pyplot as plt
import json
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate, kappa_type

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
symbols = ['o', 'x', '.']

fig, axes = plt.subplots(1, 3, sharey=True)

for aidx, type in enumerate([
    'uniform',
    'only_z_0',
    'only_z_2'
]):
    ax = axes[aidx]
    for lidx, level in enumerate([6, 7, 8]):
        try:
            with open(f'run_surrogate_lvl{level}_degree_squished_{type}.json') as f:
                data = json.loads(f.read())
        except:
            continue

        data = data['results']

        ilu_corr = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == True]

        surr_ilu_rates_corr = [d['rate'] for d in ilu_corr]
        surr_ilu_degrees_corr = [d['ilu_deg'][2] for d in ilu_corr]

        basic_ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu_basic]

        level = int(data[0]["maxLevel"])
        ax.semilogy([min(surr_ilu_degrees_corr), max(surr_ilu_degrees_corr)],
                    [basic_ilu_rates]*2,
                    '--',
                    color=colors[lidx],
                    #label=f'matrix-based, level {level}'
        )
        ax.semilogy(surr_ilu_degrees_corr,
                    surr_ilu_rates_corr,
                    '-'+symbols[lidx],
                    color=colors[lidx],
                    #label=f'surrogated, level {level}'
                    label=f'level {level}'
        )

    if type == 'uniform':
        ax.set_xlabel(r'$dg_z$')
        ax.set_title(r'$dg_x = dg_y = dg_z$')
    elif type == 'only_z_0':
        ax.set_xlabel(r'$dg_z$')
        ax.set_title(r'$dg_x = dg_y = 0$')
    else:
        ax.set_xlabel(r'$dg_z$')
        ax.set_title(r'$dg_x = dg_y = 2$')

    ax.set_ylim([10**(-3), 1])
    ax.set_xticks([i for i in range(len(surr_ilu_rates_corr)) if i % 2 == 0])
    ax.grid(True)
    #ax.set(aspect=12./3.)

#ratio = 1.0
#x_left, x_right = ax.get_xlim()
#y_low, y_high = ax.get_ylim()
#ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
#ax.set_aspect(12.)
#ax.set_aspect(1.)

plt.tight_layout()
plt.legend()
plt.show()
