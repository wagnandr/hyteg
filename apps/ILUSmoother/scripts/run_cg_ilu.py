import os
import subprocess
import re


def run(parameters=[], np=1, main_file_='../ILUSmoother3D'):
    os.makedirs('./output', exist_ok=True)
    params = ['mpirun', '-np', f'{np}', main_file_, './run_cg_ilu.prm'] + parameters
    output = subprocess.check_output(params)
    output = output.decode('utf-8')
    return output

def run_simulation(level, solver_type, smoother_type, deg_x, deg_y, deg_z, correction=True):
    return run([f'-Parameters.minLevel={level}',
                f'-Parameters.maxLevel={level}',
                f'-Parameters.solver_type={solver_type}',
                f'-Parameters.smoother_type={smoother_type}',
                f'-Parameters.ilu_surrogate_degree_x={deg_x}',
                f'-Parameters.ilu_surrogate_degree_y={deg_y}',
                f'-Parameters.ilu_use_boundary_correction={correction}',
                f'-Parameters.ilu_surrogate_degree_z={deg_z}',], np=1)


def extract(output):
    regex = r".* \[CG\] converged after (.*) iterations$"
    match = re.search(regex, output, re.MULTILINE)
    if match is not None:
        return int(match.group(1))
    return -1


level = 6


s = run_simulation(level, 'cg_ilu', 'cell_gs', 0, 0, 0)
it_cell_gs = extract(s)
print(it_cell_gs)

#s = run_simulation(level, 'cg_none', 'cell_gs', 0, 0, 0)
#it_none = extract(s)
#print(it_none)

it_ilu_corr = []
for deg in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    s = run_simulation(level, 'cg_ilu', 'surrogate_ldlt', deg, deg, deg, True)
    it = extract(s)
    it_ilu_corr.append(it)
    print(it_ilu_corr)

it_ilu_nocorr = []
for deg in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    s = run_simulation(level, 'cg_ilu', 'surrogate_ldlt', deg, deg, deg, False)
    it = extract(s)
    it_ilu_nocorr.append(it)
    print(it_ilu_nocorr)

s = run_simulation(level, 'cg_ilu', 'inplace_ldlt', 0, 0, 0)
it_ilu_inplace = extract(s)
print(it_ilu_inplace)


print(f'it_cell_gs = {it_cell_gs}')
#print(f'it_none = {it_none}')
print(f'it_ilu_inplace = {it_ilu_inplace}')
print(f'it_ilu_nocorr = {it_ilu_nocorr}')
print(f'it_ilu_corr = {it_ilu_corr}')