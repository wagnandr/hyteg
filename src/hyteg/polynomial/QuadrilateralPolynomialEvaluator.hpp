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

#include "hyteg/types/pointnd.hpp"
#include "hyteg/polynomial/QuadrilateralBasis.hpp"
#include "hyteg/polynomial/QuadrilateralPolynomial.hpp"
#include "hyteg/polynomial/PolynomialEvaluator.hpp"

namespace hyteg {

using walberla::real_t;
using walberla::uint_t;

class QuadrilateralPolynomial2DEvaluator
{
 public:
   explicit QuadrilateralPolynomial2DEvaluator( const QuadrilateralPolynomial2D& poly )
   : poly2_( &poly )
   , poly1_( poly.getDegree()[0] )
   , degreeX_( poly1_.getDegree() )
   , deltas_( poly.getDegree()[0] + 1 )
   {}

   [[nodiscard]] real_t eval( const Point2D& x ) const { return poly2_->eval( x ); }

   void setPolynomial( const QuadrilateralPolynomial2D& poly )
   {
      WALBERLA_ASSERT( poly.getDegree()[0] == degreeX_ )
      poly2_ = &poly;
   }

   void setY(real_t y) {
      for (uint_t degree = 0; degree <= degreeX_; ++degree) {
         poly1_.setCoefficient(degree, 0.0);
      }

      const uint_t degreeY = poly2_->getDegree()[1];

      uint_t start = 0;
      uint_t coeff = 0;

      for (uint_t xIdx = 0; xIdx <= degreeX_; ++xIdx)
      {
         auto yPower = real_t(1.0);
         for (uint_t yIdx = 0; yIdx <= degreeY; ++yIdx)
         {
            poly1_.addToCoefficient(xIdx, poly2_->getCoefficient(coeff) * yPower);

            yPower *= y;
            coeff += 1;
         }
      }
   }

   [[nodiscard]] real_t evalX(real_t x) const {
      return poly1_.eval(x);
   }

   real_t setStartX( real_t x, real_t h )
   {
      switch ( degreeX_ )
      {
      case 0:
         return polynomialevaluator::setStartX< 0 >( x, h, poly1_, deltas_ );
      case 1:
         return polynomialevaluator::setStartX< 1 >( x, h, poly1_, deltas_ );
      case 2:
         return polynomialevaluator::setStartX< 2 >( x, h, poly1_, deltas_ );
      case 3:
         return polynomialevaluator::setStartX< 3 >( x, h, poly1_, deltas_ );
      case 4:
         return polynomialevaluator::setStartX< 4 >( x, h, poly1_, deltas_ );
      case 5:
         return polynomialevaluator::setStartX< 5 >( x, h, poly1_, deltas_ );
      case 6:
         return polynomialevaluator::setStartX< 6 >( x, h, poly1_, deltas_ );
      case 7:
         return polynomialevaluator::setStartX< 7 >( x, h, poly1_, deltas_ );
      case 8:
         return polynomialevaluator::setStartX< 8 >( x, h, poly1_, deltas_ );
      case 9:
         return polynomialevaluator::setStartX< 9 >( x, h, poly1_, deltas_ );
      case 10:
         return polynomialevaluator::setStartX< 10 >( x, h, poly1_, deltas_ );
      case 11:
         return polynomialevaluator::setStartX< 11 >( x, h, poly1_, deltas_ );
      case 12:
         return polynomialevaluator::setStartX< 12 >( x, h, poly1_, deltas_ );
      default:
         return 0;
      }
   }

   real_t incrementEval()
   {
      switch ( degreeX_ )
      {
      case 0:
         return polynomialevaluator::incrementEval< 0 >( deltas_ );
      case 1:
         return polynomialevaluator::incrementEval< 1 >( deltas_ );
      case 2:
         return polynomialevaluator::incrementEval< 2 >( deltas_ );
      case 3:
         return polynomialevaluator::incrementEval< 3 >( deltas_ );
      case 4:
         return polynomialevaluator::incrementEval< 4 >( deltas_ );
      case 5:
         return polynomialevaluator::incrementEval< 5 >( deltas_ );
      case 6:
         return polynomialevaluator::incrementEval< 6 >( deltas_ );
      case 7:
         return polynomialevaluator::incrementEval< 7 >( deltas_ );
      case 8:
         return polynomialevaluator::incrementEval< 8 >( deltas_ );
      case 9:
         return polynomialevaluator::incrementEval< 9 >( deltas_ );
      case 10:
         return polynomialevaluator::incrementEval< 10 >( deltas_ );
      case 11:
         return polynomialevaluator::incrementEval< 11 >( deltas_ );
      case 12:
         return polynomialevaluator::incrementEval< 12 >( deltas_ );
      default:
         return 0;
      }
   }

   [[nodiscard]] const Polynomial1D< MonomialBasis1D >& getPolynomial1D() const { return poly1_; }

 private:
   const QuadrilateralPolynomial2D* poly2_;

   Polynomial1D< MonomialBasis1D > poly1_;

   uint_t degreeX_;

   std::vector< real_t > deltas_;
};

} // namespace hyteg
