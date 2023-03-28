from _shared import run, extract_number, extract_vector_int, to_json
from _shared import tetrahedron_types
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ResultsMGSurrogate:
    maxLevel: int
    smoother: str
    ilu_deg: Tuple[int, int, int]
    boundary_correction: bool
    op_deg: Tuple[int, int, int]
    rate: float
    height: float

def run_height_mg(smoother, ilu_deg, op_deg, boundary_correction, height, maxLevel):
    output = run('./run_surrogate_degree_squished.prm', [
        f'-Parameters.maxLevel={maxLevel}',
        f'-Parameters.smoother_type={smoother}',
        f'-Parameters.domain=tetrahedron',
        f'-Parameters.tetrahedron_height={height}',
        f'-Parameters.ilu_surrogate_degree_x={ilu_deg[0]}',
        f'-Parameters.ilu_surrogate_degree_y={ilu_deg[1]}',
        f'-Parameters.ilu_surrogate_degree_z={ilu_deg[2]}',
        f'-Parameters.ilu_use_boundary_correction={boundary_correction}',
        f'-Parameters.op_surrogate_degree_x={op_deg[0]}',
        f'-Parameters.op_surrogate_degree_y={op_deg[1]}',
        f'-Parameters.op_surrogate_degree_z={op_deg[2]}',
    ], 1)
    print(output)
    return ResultsMGSurrogate(
        maxLevel=extract_number('maxLevel', output),
        smoother=smoother,
        rate=extract_number('eigenvalue', output),
        ilu_deg=ilu_deg,
        boundary_correction=boundary_correction,
        op_deg=op_deg,
        height=height,
    )


ilu_basic = 'inplace_ldlt'
ilu_surrogate = 'surrogate_ldlt'


if __name__ == '__main__':
    output = []
    degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    op_deg = (0, 0, 0)
    height = 0.1
    uniform = False
    deg_xy = 2
    maxLevel = 8
    for deg in degrees:
        for boundary_correction in [True]:
            if uniform:
                ilu_deg = (deg, deg, deg)
            else:
                ilu_deg = (deg_xy, deg_xy, deg)
            #op_deg = (deg, deg, deg)
            output.append(run_height_mg(ilu_surrogate, ilu_deg, op_deg, boundary_correction, height, maxLevel))
            print(to_json(output))

    output.append(run_height_mg(ilu_basic, [0, 0, 0], op_deg, True, height, maxLevel))

    level = int(output[0].maxLevel)

    with open(f'run_surrogate_lvl{level}_degree_squished_{"uniform" if uniform else f"only_z_{deg_xy}"}.json', 'w') as f:
        f.write(to_json({
            'script': 'run_surrogate_degree_squished',
            'results': output
        }))
