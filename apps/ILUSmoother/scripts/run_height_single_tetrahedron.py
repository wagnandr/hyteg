from _shared import run, extract_number, to_json
from dataclasses import dataclass


@dataclass
class ResultsMG:
    maxLevel: int
    smoother: str
    rate: float
    height: float

def run_height_mg(smoother, height):
    output = run('./run_height_single_tetrahedron.prm', [
        f'-Parameters.smoother_type={smoother}',
        f'-Parameters.tetrahedron_height={height}',
    ], 1)
    print(output)
    return ResultsMG(
        maxLevel=extract_number('maxLevel', output),
        smoother=smoother,
        rate=extract_number('eigenvalue', output),
        height=height
    )


gs = 'cell_gs'
ilu_basic = 'inplace_ldlt'

smoother_types = [gs, ilu_basic]


if __name__ == '__main__':
    output = []
    kappa_lower = 1.
    heights = [1.0, 0.75, 0.5, 0.2, 0.1, 0.05, 0.025, 0.0125]
    for smoother in smoother_types:
        for height in heights:
            output.append(run_height_mg(smoother, height))
            print(to_json(output))

    with open('run_height_single_tetrahedron.json', 'w') as f:
        f.write(to_json({
            'script': 'run_height_single_tetrahedron.py',
            'results': output
        }))
