from _shared import run, extract_number, extract_vector_int, to_json
from _shared import tetrahedron_types
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class StencilValues:
    def __init__(self, list):
        self.w = list[0]
        self.s = list[1]
        self.se = list[2]
        self.bnw = list[3]
        self.bn = list[4]
        self.bc = list[5]
        self.be = list[6]
        self.c = list[7]

    w: float
    s: float
    se: float
    bnw: float
    bn: float
    bc: float
    be: float
    c: float


@dataclass
class ResultsMGSurrogate:
    degree: Tuple[int,int,int]
    level: int
    l2_error_global: StencilValues
    l2_error_per_dof: StencilValues
    L2_error_global: StencilValues


directions = ['W', 'S', 'SE', 'BNW', 'BN', 'BC', 'BE', 'C']


def run_height_mg(ilu_deg, height, level):
    output = run('./run_single_squished_tetrahedron.prm', [
        f'-Parameters.level={level}',
        f'-Parameters.tetrahedron_height={height}',
        f'-Parameters.degreeX={ilu_deg[0]}',
        f'-Parameters.degreeY={ilu_deg[1]}',
        f'-Parameters.degreeZ={ilu_deg[2]}',
    ], 1, main_file_='../CompareSurrogateILUStencils')
    print(output)
    return ResultsMGSurrogate(
        degree=ilu_deg,
        level=level,
        l2_error_global=StencilValues([extract_number(f'l2 error global VERTEX_{s} ', output) for s in directions]),
        l2_error_per_dof=StencilValues([extract_number(f'l2 error per dof VERTEX_{s} ', output) for s in directions]),
        L2_error_global=StencilValues([extract_number(f'L2 error global VERTEX_{s} ', output) for s in directions]),
    )


ilu_basic = 'inplace_ldlt'
ilu_surrogate = 'surrogate_ldlt'


if __name__ == '__main__':
    output = []
    #degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    #degrees = [0, 1, 2, 3, 4, 5]
    degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    op_deg = (0, 0, 0)
    height = 0.1
    uniform = True
    deg_xy = 0
    for maxLevel in [5, 6, 7, 8]:
        for deg in degrees:
            if uniform:
                ilu_deg = (deg, deg, deg)
            else:
                ilu_deg = (deg_xy, deg_xy, deg)
            output.append(run_height_mg(ilu_deg, height, maxLevel))
            print(to_json(output))

    with open(f'run_single_squished_tetrahedron.json', 'w') as f:
        f.write(to_json({
            'script': 'run_single_squished_tetrahedron',
            'results': output
        }))
