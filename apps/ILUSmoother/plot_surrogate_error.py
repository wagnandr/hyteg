import matplotlib.pyplot as plt
import subprocess
import re


directions = ['W', 'S', 'SE', 'BNW', 'BN', 'BC', 'BE', 'C']


def run(degree):
    try:
        params = [
            'mpirun', '-np', '1',
            './CompareSurrogateILUStencils', 'CompareSurrogateILUStencils.prm',
            '-Parameters.vtk_output={}'.format(False),
            '-Parameters.degree={}'.format(degree),
        ]
        print(params)
        output = subprocess.check_output(params)
    except:
        print('some error :(')
        print(output)
        return float('nan'), False

    output = output.decode('utf-8')

    results = {}

    for d in directions:
        regex = r"l2 error VERTEX_{} (.*)$".format(d)

        match = re.search(regex, output, re.MULTILINE)

        if match == None:
            print('Error: Incomplete output')
            val = float('nan')
        else:
            val = float(match.group(1))

        results[d] = val

    return results


degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

results = {}
for dir in directions:
    results[dir] = []

for deg in degrees:
    r = run(deg)
    for dir in directions:
        results[dir].append(r[dir])

print(results)

plt.grid(True)
for dir in directions:
    plt.semilogy(degrees, results[dir], label=dir)
plt.ylabel(r'$l_2$-error')
plt.xlabel('degree')
plt.legend()
plt.show()


if False:
    # degrees = [9, 10, 11]
    #degrees = [7, 8, 9]
    degrees = [6, 7]
    level = 8
    skip_levels = range(1, level+1)
    skip_distance = [2**level / 2**skip_level - 1 for skip_level in skip_levels]

    plt.hlines(gs_rate, xmin=skip_levels[0], xmax=skip_levels[-1], colors='r', linestyles='dashed', label='GS')
    plt.hlines(ilu_rate, xmin=skip_levels[0], xmax=skip_levels[-1], colors='g', linestyles='dashed', label='ILU')


    for degree in degrees:
        rate_surrogate_ilu = []
        for skip_level in skip_levels:
            rate = run('surrogate_ldlt', degree=degree, skip_level=skip_level)
            rate_surrogate_ilu.append(rate)
        print(rate_surrogate_ilu)
        plt.semilogy(skip_levels, rate_surrogate_ilu, label='surrogate $LDL^T$, degree={}'.format(degree))

    plt.grid(True)
    plt.ylim((ilu_rate*0.9, 1))
    #plt.xlim((0, level))
    plt.ylabel(r'$\rho$')
    plt.xlabel('skip level')
    plt.legend()
    plt.show()
