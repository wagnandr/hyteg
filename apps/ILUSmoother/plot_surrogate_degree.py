import matplotlib.pyplot as plt
import subprocess
import re


def run(
    smoother_type,
    surrogate_degree=None,
    surrogate_skip_level=None
):

    print(smoother_type,
          surrogate_degree,
          surrogate_skip_level)

    regex = r"Final eigenvalue: (.*)$"

    try:
        params = [
            'mpirun', '-np', '1',
            './ILUSmoother3D', 'ILUSmoother3D.prm',
            '-Parameters.smoother_type={}'.format(smoother_type),
        ]

        params.append('-Parameters.smoother_type={}'.format(smoother_type))

        if surrogate_degree:
            params.append('-Parameters.surrogate_degree_x={}'.format(surrogate_degree))
            params.append('-Parameters.surrogate_degree_y={}'.format(surrogate_degree))
            params.append('-Parameters.surrogate_degree_z={}'.format(surrogate_degree))
        if surrogate_skip_level:
            params.append('-Parameters.surrogate_skip_level={}'.format(surrogate_skip_level))

        print(params)

        output = subprocess.check_output(params)

    except:
        print('some error :(')
        print(output)
        return float('nan'), False

    output = output.decode('utf-8')

    print(output)

    match = re.search(regex, output, re.MULTILINE)

    if match == None:
        print('Error: Incomplete output')
        val = float('nan')
    else:
        val = float(match.group(1))

    return val


gs_rate = run('gs')
ilu_rate = run('inplace_ldlt')

if True:
    degrees = list(range(1,13,1))
    rate_surrogate_ilu = []

    for degree in degrees:
        rate = run('surrogate_ldlt', surrogate_degree=degree, surrogate_skip_level=20)
        rate_surrogate_ilu.append(rate)

    plt.hlines(gs_rate, xmin=degrees[0], xmax=degrees[-1], colors='r', linestyles='dashed', label='GS')
    plt.hlines(ilu_rate, xmin=degrees[0], xmax=degrees[-1], colors='g', linestyles='dashed', label='ILU')
    #plt.plot(degrees, rate_surrogate_ilu, label='surrogate $LDL^T$')
    plt.semilogy(degrees, rate_surrogate_ilu, 'x-', label='surrogate $LDL^T$')
    plt.grid(True)
    plt.ylim((0, 1))
    plt.ylabel(r'$\rho$')
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
