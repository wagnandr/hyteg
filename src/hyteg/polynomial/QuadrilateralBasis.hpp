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

#include "hyteg/types/pointnd.hpp"

namespace hyteg {

using walberla::real_t;
using walberla::uint_t;

class QuadrilateralBasis3D
{
 public:
   QuadrilateralBasis3D( uint_t degreeX, uint_t degreeY, uint_t degreeZ )
   : degreeX_( degreeX )
   , degreeY_( degreeY )
   , degreeZ_( degreeZ )
   {}

   QuadrilateralBasis3D( const std::array< uint_t, 3 >& degrees )
   : QuadrilateralBasis3D( degrees[0], degrees[1], degrees[2] )
   {}

   [[nodiscard]] real_t eval( uint_t basis, const Point3D& p ) const
   {
      const uint_t numMonomialsY = degreeY_ + 1;
      const uint_t numMonomialsZ = degreeZ_ + 1;
      const uint_t powerX        = basis / ( numMonomialsY * numMonomialsZ );
      basis -= powerX * numMonomialsY * numMonomialsZ;
      const uint_t powerY = basis / numMonomialsZ;
      basis -= powerY * numMonomialsZ;
      const uint_t powerZ = basis;
      WALBERLA_ASSERT_LESS_EQUAL( powerX, degreeX_ );
      WALBERLA_ASSERT_LESS_EQUAL( powerY, degreeY_ );
      WALBERLA_ASSERT_LESS_EQUAL( powerZ, degreeZ_ );
      return std::pow( p[0], powerX ) * std::pow( p[1], powerY ) * std::pow( p[2], powerZ );
   }

   [[nodiscard]] std::array< uint_t, 3 > getDegrees() const { return { degreeX_, degreeY_, degreeZ_ }; }

   [[nodiscard]] uint_t numBasisFunctions() const { return ( degreeX_ + 1 ) * ( degreeY_ + 1 ) * ( degreeZ_ + 1 ); }

 private:
   uint_t degreeX_;
   uint_t degreeY_;
   uint_t degreeZ_;
};

class QuadrilateralBasis2D
{
 public:
   QuadrilateralBasis2D( uint_t degreeX, uint_t degreeY )
   : degreeX_( degreeX )
   , degreeY_( degreeY )
   {}

   [[nodiscard]] real_t eval( uint_t basis, const Point2D& p ) const
   {
      const uint_t numMonomialsY = degreeY_ + 1;
      const uint_t powerX        = basis / numMonomialsY;
      basis -= powerX * numMonomialsY;
      const uint_t powerY = basis;
      WALBERLA_ASSERT_LESS_EQUAL( powerX, degreeX_ );
      WALBERLA_ASSERT_LESS_EQUAL( powerY, degreeY_ );
      return std::pow( p[0], powerX ) * std::pow( p[1], powerY );
   }

   [[nodiscard]] std::array< uint_t, 2 > getDegrees() const { return { degreeX_, degreeY_ }; }

   [[nodiscard]] uint_t numBasisFunctions() const { return ( degreeX_ + 1 ) * ( degreeY_ + 1 ); }

 private:
   uint_t degreeX_;
   uint_t degreeY_;
};

} // namespace hyteg
