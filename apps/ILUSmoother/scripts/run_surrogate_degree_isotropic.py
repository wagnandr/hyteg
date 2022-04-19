from _shared import run, extract_number, extract_vector_int, to_json
from _shared import tetrahedron_types
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ResultsMGSurrogate:
    maxLevel: int
    smoother: str
    ilu_deg: Tuple[int, int, int]
    op_deg: Tuple[int, int, int]
    rate: float
    kappa_type: str

def run_height_mg(smoother, ilu_deg, op_deg, kappa_type):
    output = run('./run_surrogate_degree_isotropic.prm', [
        f'-Parameters.smoother_type={smoother}',
        f'-Parameters.domain=tetrahedron',
        f'-Parameters.ilu_surrogate_degree_x={ilu_deg[0]}',
        f'-Parameters.ilu_surrogate_degree_y={ilu_deg[1]}',
        f'-Parameters.ilu_surrogate_degree_z={ilu_deg[2]}',
        f'-Parameters.op_surrogate_degree_x={op_deg[0]}',
        f'-Parameters.op_surrogate_degree_y={op_deg[1]}',
        f'-Parameters.op_surrogate_degree_z={op_deg[2]}',
        f'-Parameters.kappa_type={kappa_type}',
    ], 1)
    print(output)
    return ResultsMGSurrogate(
        maxLevel=extract_number('maxLevel', output),
        smoother=smoother,
        rate=extract_number('eigenvalue', output),
        ilu_deg=ilu_deg,
        op_deg=op_deg,
        kappa_type=kappa_type,
    )


kappa_type = [
    'constant',
    'linear',
    'quadratic',
    'cubic',
]

ilu_basic = 'inplace_ldlt'
ilu_surrogate = 'surrogate_ldlt'


if __name__ == '__main__':
    output = []
    degrees = [0, 1, 2, 3, 4]
    op_deg = (5, 5, 5)
    for coefficient_degree in range(4):
        for deg in degrees:
            ilu_deg = (deg, deg, deg)
            #op_deg = (deg, deg, deg)
            output.append(run_height_mg(ilu_surrogate, ilu_deg, op_deg, kappa_type[coefficient_degree]))
            print(to_json(output))

        output.append(run_height_mg(ilu_basic, [0, 0, 0], op_deg, kappa_type[coefficient_degree]))

    with open('run_surrogate_degree_isotropic.json', 'w') as f:
        f.write(to_json({
            'script': 'run_surrogate_degree_isotropic',
            'results': output
        }))
