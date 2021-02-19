#!/usr/bin/env python3

import numpy as np
import scipy.special
from sympy import *
from enum import Enum
import sys
import itertools


if __name__ == '__main__':

    DEGREE = 12
    if len(sys.argv) > 1:
        DEGREE = eval(sys.argv[1])

    X = [x,y,z] = symbols('x y z')

    monomials = [Mul(*base, evaluate=False) for d in range(DEGREE+1) for base in itertools.combinations_with_replacement(X, d)]

    # Generate code
    x.name = 'x[0]'
    y.name = 'x[1]'
    z.name = 'x[2]'

    name = 'MonomialBasis3D'

    print('#pragma once\n')
    print('// This file was generated by the monomial_basis_3d.py Python script')
    print('// Do not edit it by hand\n')
    print('namespace hyteg {\n')

    print('class {} {{'.format(name))
    print(' public:')

    print('   static real_t eval(uint_t basis, const Point3D &x) {')

    print('      switch(basis) {')

    for i, poly in enumerate(monomials):
        print('         case {}:'.format(i))

        print('            return {};'.format(ccode(poly)))

    print('         default:')
    print('            WALBERLA_ABORT("Polynomial basis " << basis << " was not generated");')

    print('      }')
    print('   }')

    print('};\n')

    print('}')
