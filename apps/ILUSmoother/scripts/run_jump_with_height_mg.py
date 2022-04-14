from _shared import run, extract_number, extract_vector_int, to_json
from _shared import tetrahedron_types
from typing import List
from dataclasses import dataclass


@dataclass
class ResultsMGJump:
    maxLevel: int
    smoother: str
    rate: float
    kappa_lower: float
    kappa_upper: float
    height: float

def run_height_mg(smoother, kappa_lower, kappa_upper, height):
    output = run('./run_jump_with_height_mg.prm', [
        f'-Parameters.smoother_type={smoother}',
        f'-Parameters.kappa_lower={kappa_lower}',
        f'-Parameters.kappa_upper={kappa_upper}',
        f'-Parameters.tetrahedron_height={height}',
    ], 4)
    print(output)
    return ResultsMGJump(
        maxLevel=extract_number('maxLevel', output),
        smoother=smoother,
        rate=extract_number('eigenvalue', output),
        kappa_lower=kappa_lower,
        kappa_upper=kappa_upper,
        height=height
    )


gs = 'cell_gs'
ilu_basic = 'inplace_ldlt'

smoother_types = [gs, ilu_basic]


if __name__ == '__main__':
    output = []
    kappa_lower = 1.
    kappa_uppers = [1e0, 1e-5, 1e5]
    heights = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.0125]
    for kappa_upper in kappa_uppers:
        for smoother in smoother_types:
            for height in heights:
                output.append(run_height_mg(smoother, kappa_lower, kappa_upper, height))
                print(to_json(output))

    with open('run_jump_with_height_mg.json', 'w') as f:
        f.write(to_json({
            'script': 'run_jump_with_height_mg',
            'results': output
        }))
