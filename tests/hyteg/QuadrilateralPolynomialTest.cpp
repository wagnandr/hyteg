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

   {
      QuadrilateralBasis2D basis( 2, 3 );

      Point2D p( { 1, 0.5 } );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 0, p ), std::pow( p[0], 0 ) * pow( p[1], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 1, p ), std::pow( p[0], 0 ) * pow( p[1], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 2, p ), std::pow( p[0], 0 ) * pow( p[1], 2 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 3, p ), std::pow( p[0], 0 ) * pow( p[1], 3 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 4, p ), std::pow( p[0], 1 ) * pow( p[1], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 5, p ), std::pow( p[0], 1 ) * pow( p[1], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 6, p ), std::pow( p[0], 1 ) * pow( p[1], 2 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 7, p ), std::pow( p[0], 1 ) * pow( p[1], 3 ) );
   }

   {
      QuadrilateralBasis3D basis( 2, 3, 1 );

      Point3D p( { 1, 0.5, 0.2 } );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 0, p ), std::pow( p[0], 0 ) * pow( p[1], 0 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 1, p ), std::pow( p[0], 0 ) * pow( p[1], 0 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 2, p ), std::pow( p[0], 0 ) * pow( p[1], 1 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 3, p ), std::pow( p[0], 0 ) * pow( p[1], 1 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 4, p ), std::pow( p[0], 0 ) * pow( p[1], 2 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 5, p ), std::pow( p[0], 0 ) * pow( p[1], 2 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 6, p ), std::pow( p[0], 0 ) * pow( p[1], 3 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 7, p ), std::pow( p[0], 0 ) * pow( p[1], 3 ) * pow( p[2], 1 ) );

      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 8, p ), std::pow( p[0], 1 ) * pow( p[1], 0 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 9, p ), std::pow( p[0], 1 ) * pow( p[1], 0 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 10, p ), std::pow( p[0], 1 ) * pow( p[1], 1 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 11, p ), std::pow( p[0], 1 ) * pow( p[1], 1 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 12, p ), std::pow( p[0], 1 ) * pow( p[1], 2 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 13, p ), std::pow( p[0], 1 ) * pow( p[1], 2 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 14, p ), std::pow( p[0], 1 ) * pow( p[1], 3 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 15, p ), std::pow( p[0], 1 ) * pow( p[1], 3 ) * pow( p[2], 1 ) );

      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 16, p ), std::pow( p[0], 2 ) * pow( p[1], 0 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 17, p ), std::pow( p[0], 2 ) * pow( p[1], 0 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 18, p ), std::pow( p[0], 2 ) * pow( p[1], 1 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 19, p ), std::pow( p[0], 2 ) * pow( p[1], 1 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 20, p ), std::pow( p[0], 2 ) * pow( p[1], 2 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 21, p ), std::pow( p[0], 2 ) * pow( p[1], 2 ) * pow( p[2], 1 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 22, p ), std::pow( p[0], 2 ) * pow( p[1], 3 ) * pow( p[2], 0 ) );
      WALBERLA_CHECK_FLOAT_EQUAL( basis.eval( 23, p ), std::pow( p[0], 2 ) * pow( p[1], 3 ) * pow( p[2], 1 ) );
   }

   {
      QuadrilateralBasis2D      basis( 2, 3 );
      QuadrilateralPolynomial2D polynomial( basis );
      for ( uint_t i = 0; i < 3 * 4; i += 1 )
         polynomial.setCoefficient( i, static_cast< real_t >( i + 1 ) );

      QuadrilateralPolynomial2DEvaluator polynomialEvaluator( polynomial );

      const real_t x = 1.2;
      const real_t y = 0.5;
      const real_t h = 0.1;

      Point2D p( { x, y } );

      polynomialEvaluator.setY( y );

      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( p ), polynomialEvaluator.evalX( p[0] ) );
      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( Point2D( { x, y } ) ), polynomialEvaluator.setStartX( x, h ) );
      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( Point2D( { x + h, y } ) ), polynomialEvaluator.incrementEval() );
      WALBERLA_CHECK_FLOAT_EQUAL( polynomial.eval( Point2D( { x + 2 * h, y } ) ), polynomialEvaluator.incrementEval() );
   }

   {
      QuadrilateralBasis3D      basis( 2, 3, 4 );
      QuadrilateralPolynomial3D polynomial( basis );
      for ( uint_t i = 0; i < 3 * 4 * 5; i += 1 )
         polynomial.setCoefficient( i, static_cast< real_t >( i + 1 ) );

      QuadrilateralPolynomial3DEvaluator polynomialEvaluator( polynomial );

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
