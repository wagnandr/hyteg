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
    rate: float

def run_height_mg(smoother, ilu_deg, boundary_correction, maxLevel):
    output = run('./run_surrogate_degree_blending.prm', [
        f'-Parameters.maxLevel={maxLevel}',
        f'-Parameters.smoother_type={smoother}',
        f'-Parameters.ilu_surrogate_degree_x={ilu_deg[0]}',
        f'-Parameters.ilu_surrogate_degree_y={ilu_deg[1]}',
        f'-Parameters.ilu_surrogate_degree_z={ilu_deg[2]}',
        f'-Parameters.ilu_use_boundary_correction={boundary_correction}',
    ], 1)
    print(output)
    return ResultsMGSurrogate(
        maxLevel=extract_number('maxLevel', output),
        smoother=smoother,
        rate=extract_number('eigenvalue', output),
        ilu_deg=ilu_deg,
        boundary_correction=boundary_correction,
    )


ilu_basic = 'inplace_ldlt'
ilu_surrogate = 'surrogate_ldlt'


if __name__ == '__main__':
    output = []
    degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    op_deg = (0, 0, 0)
    uniform = True
    deg_xy = 2
    maxLevel = 8
    for deg in degrees:
        for boundary_correction in [True]:
            if uniform:
                ilu_deg = (deg, deg, deg)
            else:
                ilu_deg = (deg_xy, deg_xy, deg)
            #op_deg = (deg, deg, deg)
            output.append(run_height_mg(ilu_surrogate, ilu_deg, boundary_correction, maxLevel))
            print(to_json(output))

    output.append(run_height_mg(ilu_basic, [0, 0, 0], True, maxLevel))

    with open(f'run_surrogate_lvl{maxLevel}_degree_blending_{"uniform" if uniform else f"only_z_{deg_xy}"}.json', 'w') as f:
        f.write(to_json({
            'script': 'run_surrogate_degree_blending',
            'results': output
        }))
