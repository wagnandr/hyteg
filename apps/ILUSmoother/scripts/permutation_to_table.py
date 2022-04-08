import json
from _shared import spade, spindle, cap, regular



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

    print('spindle')
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_spindle)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_spindle]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_spindle]))
    print()

    print('cap')
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_cap)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_cap]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_cap]))
    print()

    print('spade')
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_spade)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_spade]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_spade]))
    print()

    print('regular')
    print(' '.join([f'{i}'.rjust(8) for i,x in enumerate(ilu_rates_regular)]))
    print(' '.join([f'{x:.6f}' for x in gs_rates_regular]))
    print(' '.join([f'{x:.6f}' for x in ilu_rates_regular]))
    print()

