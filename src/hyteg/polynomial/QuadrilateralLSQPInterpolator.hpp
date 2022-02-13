/*
* Copyright (c) 2022 Andreas Wagner.
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
#pragma once

#include "core/DataTypes.h"

#include "hyteg/eigen/EigenWrapper.hpp"

#include "Polynomial.hpp"

namespace hyteg {

using walberla::real_t;
using walberla::uint_t;

template < typename Basis, typename Polynomial, typename Point >
class VariableQuadrilateralLSQPInterpolator
{
 public:
   VariableQuadrilateralLSQPInterpolator( const Basis& basis )
   : basis_( basis )
   {}

   void addInterpolationPoint( const Point& x, real_t value )
   {
      points.push_back( x );
      values.push_back( value );
   }

   void interpolate( Polynomial& poly )
   {
      if ( points.size() != values.size() )
         WALBERLA_ABORT( "point and value sizes are different" );

      // build system of equations:
      auto num_coefficients = basis_.numBasisFunctions();
      if ( num_coefficients > points.size() )
         WALBERLA_LOG_WARNING( "Polynomial interpolation may have poor quality since there are less interpolation points "
                               "than coefficients. Please try to increase the interpolation level to fix this." );

      // if no data is present, we make sure that the polynomials is unusable.
      if ( points.empty() )
      {
         for ( uint_t i = 0; i < num_coefficients; ++i )
         {
            poly.setCoefficient( i, NAN );
         }
         return;
      }

      Eigen::Matrix< real_t, Eigen::Dynamic, Eigen::Dynamic > A( values.size(), num_coefficients );
      Eigen::Matrix< real_t, Eigen::Dynamic, Eigen::Dynamic > rhs( values.size(), 1 );
      for ( int i = 0; i < values.size(); ++i )
      {
         auto x     = points.at( i );
         auto value = values.at( i );
         for ( int k = 0; k < num_coefficients; ++k )
         {
            A( i, k ) = basis_.eval( k, x );
         }
         rhs( i ) = value;
      }

      Eigen::Matrix< real_t, Eigen::Dynamic, Eigen::Dynamic > coeffs =
          A.bdcSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( rhs );

      for ( uint_t i = 0; i < num_coefficients; ++i )
      {
         poly.setCoefficient( i, coeffs( i ) );
      }
   }

 private:
   Basis                 basis_;
   std::vector< Point >  points;
   std::vector< double > values;
};

} // namespace hyteg