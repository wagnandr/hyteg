from matplotlib import pyplot as plt
import json
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate, kappa_type

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

with open('run_single_squished_tetrahedron.json') as f:
    data = json.loads(f.read())

data = data['results']

# stencils = ['w', 's', 'se', 'bnw', 'bn', 'bc', 'be', 'c']
stencils = ['bc', 'bnw', 'bn', 'be', 'c', 'se', 's', 'w']

def extract(data, error_type, stencil, level):
    data = [d for d in data if d['level'] == level]
    error = [d[error_type][stencil] for d in data]
    degree = [d['degree'][0] for d in data]
    return degree, error

fig, ax = plt.subplots(2, 4, sharey=False, sharex=True)
'''
for s in stencils:
    deg, error = extract(data, 'l2_error_per_dof', s, 5)
    ax.semilogy(deg, error, '-o', label=f'{s} (level {5})')
'''

for x in range(2):
    for y in range(4):
        for level, symbol in zip([6, 7], ['-o', '-x']):
            stencil = stencils[4*x + y]
            #deg, error = extract(data, 'l2_error_per_dof', stencil, level)
            deg, error = extract(data, 'L2_error_global', stencil, level)
            label = None
            if x == 1 and y == 3:
                label = f'level {level}'
            ax[x][y].set_title(stencil)
            ax[x][y].semilogy(deg, error, symbol, label=label)
            ax[x][y].grid(True)

handles, labels = ax[-1][-1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0,-0.1,1,1))
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .1),)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()


'''
fig, ax = plt.subplots(1, 1, sharey=True,)

ax.semilogy([min(surr_ilu_degrees_boundary), max(surr_ilu_degrees_boundary)], [basic_ilu_rates]*2, '--', color='tab:gray', label='matrix')
ax.semilogy(surr_ilu_degrees_boundary, surr_ilu_rates_boundary, '-o', label=r'surrogate with boundary')
ax.semilogy(surr_ilu_degrees_corr, surr_ilu_rates_corr, '-o', label=r'surrogate without boundary')

ax.set_xlabel(r'$dg_x = dg_y = dg_z$')
ax.set_ylabel(r'$\rho$')
ax.set_ylim([10**(-3), 1])
ax.set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
ax.grid(True)


plt.legend()
plt.tight_layout()
plt.show()
'''
