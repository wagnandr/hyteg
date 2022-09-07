from matplotlib import pyplot as plt
import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular
from run_surrogate_degree_isotropic import ilu_basic, ilu_surrogate, kappa_type

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

#with open('run_surrogate_degree_squished_uniform.json') as f:
with open('run_surrogate_degree_squished_only_z_2.json') as f:
    data = json.loads(f.read())

data = data['results']

ilu_corr = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == True]
ilu_boundary = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == False]

surr_ilu_rates_corr = [d['rate'] for d in ilu_corr]
surr_ilu_degrees_corr = [d['ilu_deg'][0] for d in ilu_corr]

surr_ilu_rates_boundary = [d['rate'] for d in ilu_boundary]
surr_ilu_degrees_boundary = [d['ilu_deg'][0] for d in ilu_boundary]

basic_ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu_basic]

fig, ax = plt.subplots(1, 2, sharey=True,)

ax[0].semilogy([min(surr_ilu_degrees_boundary), max(surr_ilu_degrees_boundary)], [basic_ilu_rates]*2, '--', color='tab:gray', label='matrix')
ax[0].semilogy(surr_ilu_degrees_boundary, surr_ilu_rates_boundary, '-o', label=r'surrogate with boundary')
ax[0].semilogy(surr_ilu_degrees_corr, surr_ilu_rates_corr, '-o', label=r'surrogate without boundary')

ax[0].set_xlabel(r'$dg_x = dg_y = dg_z$')
ax[0].set_ylabel(r'$\rho$')
ax[0].set_ylim([10**(-3), 1])
ax[0].set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
ax[0].grid(True)

'''
with open('run_surrogate_degree_squished_only_z_0.json') as f:
    data = json.loads(f.read())

data = data['results']

ilu_corr = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == True]
ilu_boundary = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == False]

surr_ilu_rates_corr = [d['rate'] for d in ilu_corr]
surr_ilu_degrees_corr = [d['ilu_deg'][2] for d in ilu_corr]

surr_ilu_rates_boundary = [d['rate'] for d in ilu_boundary]
surr_ilu_degrees_boundary = [d['ilu_deg'][2] for d in ilu_boundary]

basic_ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu_basic]

ax[1].semilogy([min(surr_ilu_degrees_boundary), max(surr_ilu_degrees_boundary)], [basic_ilu_rates]*2, '--', color='tab:gray', label='matrix')
ax[1].semilogy(surr_ilu_degrees_boundary, surr_ilu_rates_boundary, '-o', label=r'surrogate V1')
ax[1].semilogy(surr_ilu_degrees_corr, surr_ilu_rates_corr, '-o', label=r'surrogate V2')

ax[1].set_xlabel(r'$dg_z$')
ax[1].set_ylim([10**(-3), 1])
ax[1].set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
ax[1].grid(True)
'''

'''
with open('run_surrogate_degree_squished_only_z_2.json') as f:
    data = json.loads(f.read())

data = data['results']

ilu_corr = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == True]
ilu_boundary = [d for d in data if d['smoother'] == ilu_surrogate and d['boundary_correction'] == False]

surr_ilu_rates_corr = [d['rate'] for d in ilu_corr]
surr_ilu_degrees_corr = [d['ilu_deg'][2] for d in ilu_corr]

surr_ilu_rates_boundary = [d['rate'] for d in ilu_boundary]
surr_ilu_degrees_boundary = [d['ilu_deg'][2] for d in ilu_boundary]

basic_ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu_basic]

ax[2].semilogy([min(surr_ilu_degrees_boundary), max(surr_ilu_degrees_boundary)], [basic_ilu_rates]*2, '--', color='tab:gray', label='matrix')
ax[2].semilogy(surr_ilu_degrees_boundary, surr_ilu_rates_boundary, '-o', label=r'surrogate V1')
ax[2].semilogy(surr_ilu_degrees_corr, surr_ilu_rates_corr, '-o', label=r'surrogate V2')

ax[2].set_xlabel(r'$dg_z$')
ax[2].set_ylim([10**(-3), 1])
ax[2].set_xticks([i for i in range(len(surr_ilu_rates_boundary))])
ax[2].grid(True)
'''


plt.legend()
plt.tight_layout()
plt.show()
