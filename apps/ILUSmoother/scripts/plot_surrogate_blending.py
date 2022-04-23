from matplotlib import pyplot as plt
import json
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate


with open('run_surrogate_degree_blending_uniform.json') as f:
    data = json.loads(f.read())

data = data['results']

ilu_corr = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == True]
ilu_boundary = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == False]

surr_ilu_rates_corr = [d['rate'] for d in ilu_corr]
surr_ilu_degrees_corr = [d['ilu_deg'][0] for d in ilu_corr]

surr_ilu_rates_boundary = [d['rate'] for d in ilu_boundary]
surr_ilu_degrees_boundary = [d['ilu_deg'][0] for d in ilu_boundary]

basic_ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu_basic]

fig, ax = plt.subplots(1, 1, sharey=True,)

ax.semilogy([min(surr_ilu_degrees_boundary), max(surr_ilu_degrees_boundary)], [basic_ilu_rates]*2, '--', color='tab:gray', label='matrix')
ax.semilogy(surr_ilu_degrees_boundary, surr_ilu_rates_boundary, '-o', label=r'surrogate with boundary')
ax.semilogy(surr_ilu_degrees_corr, surr_ilu_rates_corr, '-o', label=r'surrogate without boundary')

ax.set_xlabel(r'$dg_x = dg_y = dg_z$')
ax.set_ylabel(r'$\rho$')
#ax.set_ylim([8e-3, 3e-1])
ax.set_ylim([1e-1, 1])
ax.set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
ax.grid(True)

plt.legend()
plt.tight_layout()
plt.show()
