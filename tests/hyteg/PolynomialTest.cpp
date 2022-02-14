//
// Created by wagneran on 14.02.22.
//
/*
 * Copyright (c) 2020 Andreas Wagner
 *
 * This file is part of HyTeG
 * (see https://i10git.cs.fau.de/hyteg/hyteg).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/math/Constants.h"
#include "core/mpi/MPIManager.h"

#include "hyteg/polynomial/QuadrilateralPolynomialEvaluator.hpp"

using walberla::real_t;
using walberla::math::pi;

using namespace hyteg;

int main( int argc, char** argv )
{
   walberla::Environment env( argc, argv );
   walberla::mpi::MPIManager::instance()->useWorldComm();

   for ( uint_t degree = 0; degree < 13; degree += 1 )
   {
      Polynomial3D< MonomialBasis3D > polynomial( degree );

      for ( uint_t i = 0; i < Polynomial3D< MonomialBasis3D >::getNumCoefficients( degree ); i += 1 )
         polynomial.setCoefficient( i, static_cast< real_t >( i + 1 ) );

      Polynomial3DEvaluator polynomialEvaluator( degree );
      polynomialEvaluator.setPolynomial( polynomial );

      const real_t x = 1.2;
      const real_t y = 0.5;
      const real_t z = 3.8;
      const real_t h = 0.1;

      Point3D p( { x, y, z } );

      polynomialEvaluator.setZ( z );
      polynomialEvaluator.setY( y );

      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( p ), polynomialEvaluator.evalX( p[0] ) );
      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( Point3D( { x, y, z } ) ), polynomialEvaluator.setStartX( x, h ) );
      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( Point3D( { x + h, y, z } ) ), polynomialEvaluator.incrementEval() );
      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( Point3D( { x + 2 * h, y, z } ) ), polynomialEvaluator.incrementEval() );
   }
}
