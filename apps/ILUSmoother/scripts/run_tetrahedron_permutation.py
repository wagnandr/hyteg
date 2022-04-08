from _shared import run, extract_number, extract_vector_int, to_json
from _shared import tetrahedron_types
from typing import List
from dataclasses import dataclass


@dataclass
class ResultsTetrahedronPermutation:
    permutation_number: int
    permutation: List[int]
    domain: str
    maxLevel: int
    smoother: str
    rate: float

def run_permutation(domain, smoother, permutation_number):
    output = run('./run_tetrahedron_permutation.prm', [
        f'-Parameters.tetrahedron_permutation={permutation_number}',
        f'-Parameters.domain={domain}',
        f'-Parameters.smoother_type={smoother}'
    ])
    print(output)
    return ResultsTetrahedronPermutation(
        domain=domain,
        permutation_number=permutation_number,
        permutation=extract_vector_int('permutation', output),
        maxLevel=extract_number('maxLevel', output),
        smoother=smoother,
        rate=extract_number('eigenvalue', output)
    )


gs = 'cell_gs'
ilu_basic = 'inplace_ldlt'

smoother_types = [gs, ilu_basic]


if __name__ == '__main__':
    output = []
    for smoother in smoother_types:
        for domain in tetrahedron_types:
            for p in range(24):
                output.append(run_permutation(domain, smoother, p))
                print(to_json(output))

    with open('run_tetrahedron_permutation_results.json', 'w') as f:
        f.write(to_json({
            'script': 'run_tetrahedron_permutation_results',
            'results': output
        }))
