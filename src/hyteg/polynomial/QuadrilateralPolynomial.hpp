/*
* Copyright (c) 2021 Benjamin Mann
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
#include "core/mpi/RecvBuffer.h"

#include "hyteg/polynomial/QuadrilateralBasis.hpp"
#include "hyteg/types/pointnd.hpp"

namespace hyteg {

using walberla::real_t;
using walberla::uint_t;

template < uint_t Dim, class Point, class Basis >
class QuadrilateralPolynomial
{
 public:
   uint_t getNumCoefficients() const { return basis_.numBasisFunctions(); }

   explicit QuadrilateralPolynomial( const Basis& basis )
   : basis_( basis )
   , coeffs_( basis_.numBasisFunctions() )
   {}

   inline std::array< uint_t, Dim > getDegree() const { return basis_.getDegrees(); }

   inline real_t eval( const Point& x ) const
   {
      real_t eval = coeffs_[0] * basis_.eval( 0, x );

      for ( uint_t c = 1; c < getNumCoefficients(); ++c )
      {
         eval = std::fma( coeffs_[c], basis_.eval( c, x ), eval );
      }

      return eval;
   }

   inline void setCoefficient( uint_t idx, real_t value )
   {
      WALBERLA_ASSERT( idx < getNumCoefficients() );
      coeffs_[idx] = value;
   }

   [[nodiscard]] inline real_t getCoefficient( uint_t idx ) const
   {
      WALBERLA_ASSERT( idx < getNumCoefficients() );
      return coeffs_[idx];
   }

   void addToCoefficient( uint_t idx, real_t value ) { coeffs_[idx] += value; }

   inline void setZero() { std::memset( coeffs_.data(), 0, getNumCoefficients() * sizeof( real_t ) ); }

   /// Serializes the allocated data to a send buffer
   inline void serialize( walberla::mpi::SendBuffer& sendBuffer ) const { WALBERLA_ABORT( "Not implemented" ); }

   /// Deserializes data from a recv buffer (clears all already allocated data and replaces it with the recv buffer's content)
   inline void deserialize( walberla::mpi::RecvBuffer& recvBuffer ) { WALBERLA_ABORT( "Not implemented" ); }

 private:
   Basis                 basis_;
   std::vector< real_t > coeffs_;
};

using QuadrilateralPolynomial2D = QuadrilateralPolynomial< 2, Point2D, QuadrilateralBasis2D >;
using QuadrilateralPolynomial3D = QuadrilateralPolynomial< 3, Point3D, QuadrilateralBasis3D >;

} // namespace hyteg
