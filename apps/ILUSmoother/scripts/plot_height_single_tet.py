from matplotlib import pyplot as plt
import numpy as np
import json
from itertools import permutations
from _shared import spade, spindle, cap, regular


with open('run_height_single_tetrahedron.json') as f:
    data = json.loads(f.read())
    data = data['results']
    gs = 'cell_gs'
    ilu = 'inplace_ldlt'

    gs_rates = [d['rate'] for d in data if d['smoother'] == gs]
    gs_height = [d['height'] for d in data if d['smoother'] == gs]
    ilu_rates = [d['rate'] for d in data if d['smoother'] == ilu]
    ilu_height = [d['height'] for d in data if d['smoother'] == ilu]

    plt.plot(gs_height, gs_rates, '-x', label=r'SGS')
    plt.plot(ilu_height, ilu_rates, '-o', label=r'ILU')

    plt.legend()
    plt.grid(True)

    plt.ylabel(r'$\rho$')
    plt.xlabel(r'$h$')
    plt.show()

