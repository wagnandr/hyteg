import os
import re
import subprocess
import json
import dataclasses

main_file = '../ILUSmoother3D'

def run(config_file, parameters=[], np=1):
    os.makedirs('./output', exist_ok=True)
    params = ['mpirun', '-np', f'{np}', main_file, config_file] + parameters
    output = subprocess.check_output(params)
    output = output.decode('utf-8')
    return output

def extract_number(key, output):
    regex = key + r": (.*)$"
    match = re.search(regex, output, re.MULTILINE)
    if match is not None:
        return float(match.group(1))

    regex = key + r" = (.*)$"
    match = re.search(regex, output, re.MULTILINE)
    if match is not None:
        return float(match.group(1))

    regex = f" Key = '{key}' , Value = " + r"'(.*)'$"
    match = re.search(regex, output, re.MULTILINE)
    if match is not None:
        return float(match.group(1))

    print('Error: Incomplete output')
    return float('nan')


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def to_json(o):
    return json.dumps(o, cls=EnhancedJSONEncoder, indent=1)


def extract_vector_int(key, output):
    regex = f'{key}: ' + r'(.*) (.*) (.*) (.*)'
    match = re.search(regex, output, re.MULTILINE)
    if match is not None:
        return [int(match.group(d+1)) for d in range(4)]

    print('Error: Incomplete output')
    return [-1 for d in range(4)]


spindle = 'tetrahedron_spindle'
cap = 'tetrahedron_cap'
spade = 'tetrahedron_spade'
regular = 'tetrahedron_regular'

tetrahedron_types = [
    spindle,
    cap,
    spade,
    regular
]
