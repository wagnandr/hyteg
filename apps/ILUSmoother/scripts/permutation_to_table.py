import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular



reverse_order = False 


with open('run_tetrahedron_permutation_results.json')  as f:
    data = json.loads(f.read())
    data = data['results']
    gs = 'cell_gs'
    gs_rates_spindle = [d['rate'] for d in data if d['domain'] == spindle and d['smoother'] == gs]
    gs_rates_cap = [d['rate'] for d in data if d['domain'] == cap and d['smoother'] == gs]
    gs_rates_spade = [d['rate'] for d in data if d['domain'] == spade and d['smoother'] == gs]
    gs_rates_regular = [d['rate'] for d in data if d['domain'] == regular and d['smoother'] == gs]

    ilu_rates_spindle = [d['rate'] for d in data if d['domain'] == spindle and d['smoother'] == 'inplace_ldlt']
    ilu_rates_cap = [d['rate'] for d in data if d['domain'] == cap and d['smoother'] == 'inplace_ldlt']
    ilu_rates_spade = [d['rate'] for d in data if d['domain'] == spade and d['smoother'] == 'inplace_ldlt']
    ilu_rates_regular = [d['rate'] for d in data if d['domain'] == regular and d['smoother'] == 'inplace_ldlt']

    permutation_list = list(permutations([0,1,2,3]))
    permutation_list = [tuple(x) for x in permutation_list]
    permutation_list_labels = [''.join([f'{x+1}' for x in tx]) for tx in permutation_list]

    if reverse_order:
        permutation_list_reversed = [tuple(reversed(x)) for x in permutation_list]
        unpermute = [permutation_list.index(it) for it in permutation_list_reversed]
        print(permutation_list_reversed)

        gs_rates_spindle = np.array(gs_rates_spindle)[unpermute]
        gs_rates_cap = np.array(gs_rates_cap)[unpermute]
        gs_rates_spade = np.array(gs_rates_spade)[unpermute]
        gs_rates_regular = np.array(gs_rates_regular)[unpermute]

        ilu_rates_spindle = np.array(ilu_rates_spindle)[unpermute]
        ilu_rates_cap = np.array(ilu_rates_cap)[unpermute]
        ilu_rates_spade = np.array(ilu_rates_spade)[unpermute]
        ilu_rates_regular = np.array(ilu_rates_regular)[unpermute]

        permutation_list_labels = [permutation_list_labels[x]  for x in unpermute]

    print('spindle')
    print(' '.join([f'{l}'.rjust(8) for l in permutation_list_labels]))
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_spindle)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_spindle]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_spindle]))
    print()

    print('cap')
    print(' '.join([f'{l}'.rjust(8) for l in permutation_list_labels]))
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_cap)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_cap]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_cap]))
    print()

    print('spade')
    print(' '.join([f'{l}'.rjust(8) for l in permutation_list_labels]))
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_spade)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_spade]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_spade]))
    print()

    print('regular')
    print(' '.join([f'{l}'.rjust(8) for l in permutation_list_labels]))
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_regular)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_regular]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_regular]))
    print()

