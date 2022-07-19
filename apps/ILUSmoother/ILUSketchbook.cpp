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
#include <hyteg/forms/form_hyteg_generated/p1/p1_div_k_grad_blending_q3.hpp>
#include <hyteg/p1functionspace/P1Elements.hpp>
#include <unordered_map>

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/math/Constants.h"
#include "core/mpi/MPIManager.h"

#include "hyteg/StencilDirections.hpp"
#include "hyteg/indexing/MacroCellIndexing.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p1functionspace/VertexDoFMacroCell.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"

#include "hyteg/polynomial/LSQPInterpolator.hpp"
#include "hyteg/polynomial/QuadrilateralBasis.hpp"
#include "hyteg/polynomial/QuadrilateralLSQPInterpolator.hpp"
#include "hyteg/polynomial/QuadrilateralPolynomial.hpp"
#include "hyteg/polynomial/QuadrilateralPolynomialEvaluator.hpp"

#include "hyteg/LikwidWrapper.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;
using walberla::math::pi;

namespace hyteg {

namespace ldlt {
namespace p1 {
namespace dim3 {

using SD = stencilDirection;

constexpr std::array< SD, 7 > lowerDirections =
    { SD::VERTEX_W, SD::VERTEX_S, SD::VERTEX_SE, SD::VERTEX_BNW, SD::VERTEX_BN, SD::VERTEX_BC, SD::VERTEX_BE };

constexpr std::array< SD, 8 > lowerDirectionsAndCenter =
    { SD::VERTEX_W, SD::VERTEX_S, SD::VERTEX_SE, SD::VERTEX_BNW, SD::VERTEX_BN, SD::VERTEX_BC, SD::VERTEX_BE, SD::VERTEX_C };

constexpr std::array< SD, 7 > upperDirections =
    { SD::VERTEX_E, SD::VERTEX_N, SD::VERTEX_NW, SD::VERTEX_TSE, SD::VERTEX_TS, SD::VERTEX_TC, SD::VERTEX_TW };

constexpr std::array< SD, 15 > allDirections = { SD::VERTEX_C,
                                                 SD::VERTEX_W,
                                                 SD::VERTEX_S,
                                                 SD::VERTEX_SE,
                                                 SD::VERTEX_BNW,
                                                 SD::VERTEX_BN,
                                                 SD::VERTEX_BC,
                                                 SD::VERTEX_BE,
                                                 SD::VERTEX_E,
                                                 SD::VERTEX_N,
                                                 SD::VERTEX_NW,
                                                 SD::VERTEX_TSE,
                                                 SD::VERTEX_TS,
                                                 SD::VERTEX_TC,
                                                 SD::VERTEX_TW };

indexing::IndexIncrement toIndex( SD dir )
{
   switch ( dir )
   {
   case SD::VERTEX_W:
      return { -1, 0, 0 };
   case SD::VERTEX_S:
      return { 0, -1, 0 };
   case SD::VERTEX_SE:
      return { +1, -1, 0 };
   case SD::VERTEX_BNW:
      return { -1, +1, -1 };
   case SD::VERTEX_BN:
      return { 0, +1, -1 };
   case SD::VERTEX_BC:
      return { 0, 0, -1 };
   case SD::VERTEX_BE:
      return { +1, 0, -1 };
   case SD::VERTEX_C:
      return { 0, 0, 0 };
   case SD::VERTEX_E:
      return { +1, 0, 0 };
   case SD::VERTEX_N:
      return { 0, +1, 0 };
   case SD::VERTEX_NW:
      return { -1, +1, 0 };
   case SD::VERTEX_TSE:
      return { +1, -1, +1 };
   case SD::VERTEX_TS:
      return { 0, -1, +1 };
   case SD::VERTEX_TC:
      return { 0, 0, 1 };
   case SD::VERTEX_TW:
      return { -1, 0, +1 };
   default:
      break;
   };
   WALBERLA_ABORT( "not implemented" );
}

SD opposite( SD dir )
{
   switch ( dir )
   {
   case SD::VERTEX_W:
      return SD::VERTEX_E;
   case SD::VERTEX_S:
      return SD::VERTEX_N;
   case SD::VERTEX_SE:
      return SD::VERTEX_NW;
   case SD::VERTEX_BNW:
      return SD::VERTEX_TSE;
   case SD::VERTEX_BN:
      return SD::VERTEX_TS;
   case SD::VERTEX_BC:
      return SD::VERTEX_TC;
   case SD::VERTEX_BE:
      return SD::VERTEX_TW;
   case SD::VERTEX_C:
      return SD::VERTEX_C;
   case SD::VERTEX_E:
      return SD::VERTEX_W;
   case SD::VERTEX_N:
      return SD::VERTEX_S;
   case SD::VERTEX_NW:
      return SD::VERTEX_SE;
   case SD::VERTEX_TSE:
      return SD::VERTEX_BNW;
   case SD::VERTEX_TS:
      return SD::VERTEX_BN;
   case SD::VERTEX_TC:
      return SD::VERTEX_BC;
   case SD::VERTEX_TW:
      return SD::VERTEX_BE;
   default:
      break;
   };
   WALBERLA_ABORT( "not implemented" );
}

void print_stencil( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil )
{
   for ( auto d : ldlt::p1::dim3::lowerDirectionsAndCenter )
      WALBERLA_LOG_INFO( "x " << x << " y " << y << " z " << z << " " << stencilDirectionToStr[d] << " " << stencil[d] );
}

bool on_west_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   WALBERLA_UNUSED( y );
   WALBERLA_UNUSED( z );
   WALBERLA_UNUSED( N );
   return x == 1;
}

bool on_diagonal_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   return x + y + z == N - 2;
}

bool on_south_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   WALBERLA_UNUSED( x );
   WALBERLA_UNUSED( z );
   WALBERLA_UNUSED( N );
   return y == 1;
}

bool on_bottom_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   WALBERLA_UNUSED( x );
   WALBERLA_UNUSED( y );
   WALBERLA_UNUSED( N );
   return z == 1;
}

void apply_boundary_corrections( uint_t x, uint_t y, uint_t z, uint_t N, std::map< SD, real_t >& stencil )
{
   if ( x == 1 )
   {
      stencil[stencilDirection::VERTEX_NW]  = 0;
      stencil[stencilDirection::VERTEX_W]   = 0;
      stencil[stencilDirection::VERTEX_TW]  = 0;
      stencil[stencilDirection::VERTEX_BNW] = 0;
   }
   if ( x + y + z == N - 2 )
   {
      stencil[stencilDirection::VERTEX_E]   = 0;
      stencil[stencilDirection::VERTEX_N]   = 0;
      stencil[stencilDirection::VERTEX_TC]  = 0;
      stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( y == 1 )
   {
      stencil[stencilDirection::VERTEX_S]   = 0;
      stencil[stencilDirection::VERTEX_SE]  = 0;
      stencil[stencilDirection::VERTEX_TS]  = 0;
      stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( z == 1 )
   {
      stencil[stencilDirection::VERTEX_BC]  = 0;
      stencil[stencilDirection::VERTEX_BE]  = 0;
      stencil[stencilDirection::VERTEX_BN]  = 0;
      stencil[stencilDirection::VERTEX_BNW] = 0;
   }
}

bool on_cell_boundary( uint_t x, uint_t y, uint_t z, uint_t size_boundary, uint_t num_dof_on_edge )
{
   const bool on_east_boundary     = x < size_boundary;
   const bool on_south_boundary    = y < size_boundary;
   const bool on_bottom_boundary   = z < size_boundary;
   const bool on_diagonal_boundary = x + y + z > num_dof_on_edge - 1 - size_boundary;
   return on_east_boundary || on_south_boundary || on_bottom_boundary || on_diagonal_boundary;
}

real_t apply_boundary_corrections_to_scalar( uint_t x, uint_t y, uint_t z, uint_t N, SD d, real_t value )
{
   // if we are on the outmost boundary we return the unit stencil:
   if ( on_cell_boundary( x, y, z, 1, N ) )
   {
      if ( d == SD::VERTEX_C )
         return 1.;
      return 0.;
   }

   const bool on_west_boundary     = x == 1;
   const bool on_south_boundary    = y == 1;
   const bool on_bottom_boundary   = z == 1;
   const bool on_diagonal_boundary = x + y + z == N - 2;

   const bool is_western_stencil = ( d == stencilDirection::VERTEX_NW ) || ( d == stencilDirection::VERTEX_W ) ||
                                   ( d == stencilDirection::VERTEX_TW ) || ( d == stencilDirection::VERTEX_BNW );

   const bool is_southern_stencil = ( d == stencilDirection::VERTEX_S ) || ( d == stencilDirection::VERTEX_SE ) ||
                                    ( d == stencilDirection::VERTEX_TS ) || ( d == stencilDirection::VERTEX_TSE );

   const bool is_bottom_stencil = ( d == stencilDirection::VERTEX_BC ) || ( d == stencilDirection::VERTEX_BE ) ||
                                  ( d == stencilDirection::VERTEX_BN ) || ( d == stencilDirection::VERTEX_BNW );

   const bool is_diagonal_stencil = ( d == stencilDirection::VERTEX_E ) || ( d == stencilDirection::VERTEX_N ) ||
                                    ( d == stencilDirection::VERTEX_TC ) || ( d == stencilDirection::VERTEX_TSE );

   if ( on_west_boundary && is_western_stencil )
      return 0;

   if ( on_south_boundary && is_southern_stencil )
      return 0;

   if ( on_bottom_boundary && is_bottom_stencil )
      return 0;

   if ( on_diagonal_boundary && is_diagonal_stencil )
      return 0;

   // if we do not belong to any boundary, we return the value
   return value;
}

void apply_boundary_corrections_on_backward_stencil( uint_t x, uint_t y, uint_t z, uint_t N, std::map< SD, real_t >& stencil )
{
   stencil[SD::VERTEX_W]  = apply_boundary_corrections_to_scalar( x + 1, y, z, N, SD::VERTEX_W, stencil[SD::VERTEX_W] );
   stencil[SD::VERTEX_S]  = apply_boundary_corrections_to_scalar( x, y + 1, z, N, SD::VERTEX_S, stencil[SD::VERTEX_S] );
   stencil[SD::VERTEX_SE] = apply_boundary_corrections_to_scalar( x - 1, y + 1, z, N, SD::VERTEX_SE, stencil[SD::VERTEX_SE] );
   stencil[SD::VERTEX_BNW] =
       apply_boundary_corrections_to_scalar( x + 1, y - 1, z + 1, N, SD::VERTEX_BNW, stencil[SD::VERTEX_BNW] );
   stencil[SD::VERTEX_BN] = apply_boundary_corrections_to_scalar( x, y - 1, z + 1, N, SD::VERTEX_BN, stencil[SD::VERTEX_BN] );
   stencil[SD::VERTEX_BC] = apply_boundary_corrections_to_scalar( x, y, z + 1, N, SD::VERTEX_BC, stencil[SD::VERTEX_BC] );
   stencil[SD::VERTEX_BE] = apply_boundary_corrections_to_scalar( x - 1, y, z + 1, N, SD::VERTEX_BE, stencil[SD::VERTEX_BE] );
}

class LDLTPolynomials
{
 public:
   using Basis      = QuadrilateralBasis3D;
   using Polynomial = QuadrilateralPolynomial3D;

   template < typename CollectionType >
   LDLTPolynomials( const std::array< uint_t, 3 >& degrees, const CollectionType& directions )
   : basis_( degrees )
   {
      for ( auto d : directions )
         polynomials_.emplace( d, basis_ );
   }

   inline Polynomial& getPolynomial( stencilDirection direction ) { return polynomials_.at( direction ); }

   inline const Polynomial& operator[]( stencilDirection direction ) const { return polynomials_.at( direction ); }

   inline bool contains( stencilDirection direction ) const { return polynomials_.count( direction ); }

   [[nodiscard]] inline std::array< uint_t, 3 > getDegrees() const { return basis_.getDegrees(); }

   void print()
   {
      for ( auto it : polynomials_ )
      {
         std::stringstream buf;
         buf << stencilDirectionToStr[it.first];
         buf << " = [";
         for ( uint_t i = 0; i < it.second.getNumCoefficients(); i += 1 )
         {
            buf << it.second.getCoefficient( i ) << ",";
         }
         buf << "]";

         WALBERLA_LOG_INFO_ON_ROOT( buf.str() );
      }
   }

 private:
   Basis basis_;

   mutable std::map< SD, Polynomial > polynomials_;
};

class LDLTHierachicalPolynomials
{
 public:
   template < typename CollectionType >
   LDLTHierachicalPolynomials( uint_t                         minLevel,
                               uint_t                         maxLevel,
                               const std::array< uint_t, 3 >& degrees,
                               const CollectionType&          directions )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , collections( maxLevel - minLevel + 1, LDLTPolynomials( degrees, directions ) )
   {}

   LDLTPolynomials& getLevel( uint_t level )
   {
      WALBERLA_CHECK_LESS_EQUAL( minLevel_, level );
      WALBERLA_CHECK_GREATER_EQUAL( maxLevel_, level );
      return collections[level - minLevel_];
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;

   std::vector< LDLTPolynomials > collections;
};

class LDLTHierachicalPolynomialsDataHandling : public hyteg::OnlyInitializeDataHandling< LDLTHierachicalPolynomials, Cell >
{
 public:
   template < uint_t numDirections >
   explicit LDLTHierachicalPolynomialsDataHandling( const uint_t&                          minLevel,
                                                    const uint_t&                          maxLevel,
                                                    const std::array< uint_t, 3 >&         degrees,
                                                    const std::array< SD, numDirections >& directions )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , degrees_( degrees )
   , directions_( directions.begin(), directions.end() )
   {}

   std::shared_ptr< LDLTHierachicalPolynomials > initialize( const Cell* const ) const override
   {
      auto collection = std::make_shared< LDLTHierachicalPolynomials >( minLevel_, maxLevel_, degrees_, directions_ );
      return collection;
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;

   std::array< uint_t, 3 > degrees_;

   std::vector< SD > directions_;
};

class Interpolators
{
 public:
   using Basis          = QuadrilateralBasis3D;
   using Polynomial     = QuadrilateralPolynomial< 3, Point3D, Basis >;
   using Interpolator3D = VariableQuadrilateralLSQPInterpolator< QuadrilateralBasis3D, Polynomial, Point3D >;

   template < uint_t num_directions >
   Interpolators( std::array< uint_t, 3 > degrees, const std::array< SD, num_directions >& dir )
   : directions( dir.begin(), dir.end() )
   {
      Basis basis( degrees[0], degrees[1], degrees[2] );
      for ( auto d : directions )
         interpolators.emplace( d, basis );
   }

   Interpolators( std::array< uint_t, 3 > degrees )
   : Interpolators( degrees, lowerDirectionsAndCenter )
   {}

   Interpolator3D& operator()( SD direction ) { return interpolators.at( direction ); }

   void addStencil( const Point3D& p, const std::map< SD, real_t >& stencil )
   {
      for ( auto d : directions )
         interpolators.at( d ).addInterpolationPoint( p, stencil.at( d ) );
   }

   void addValue( const Point3D& p, SD d, real_t v ) { interpolators.at( d ).addInterpolationPoint( p, v ); }

   void interpolate( LDLTPolynomials& poly )
   {
      for ( auto d : directions )
         interpolators.at( d ).interpolate( poly.getPolynomial( d ) );
   }

   [[nodiscard]] std::vector< SD > getAllDirections() const { return directions; }

 private:
   std::vector< SD > directions;

   std::map< SD, Interpolator3D > interpolators;
};

template < size_t num_directions >
class PolyStencilNew
{
 public:
   PolyStencilNew( const std::array< uint_t, 3 > degrees, const std::array< SD, num_directions >& directions )
       : directions_( directions )
   {
      for ( uint_t i = 0; i < num_directions; i += 1 )
      {
         polynomials_.emplace_back( degrees[0], degrees[1], degrees[2] );
         offsetsX_.push_back( 0 );
         offsetsY_.push_back( 0 );
         offsetsZ_.push_back( 0 );
      }
   }

   template < typename PolyListType >
   void setPolynomial( PolyListType& polylist )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         polynomials_[i].setPolynomial( polylist[directions_[i]] );
      }
   }

   template < typename PolyListType >
   void setPolynomialSymmetrical( PolyListType& polylist, real_t h )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         if ( polylist.contains( directions_[i] ) )
         {
            polynomials_[i].setPolynomial( polylist[directions_[i]] );
         }
         else
         {
            polynomials_[i].setPolynomial( polylist[opposite( directions_[i] )] );
            auto indexIncrement = toIndex( directions_[i] );
            offsetsX_[i]        = h * indexIncrement[0];
            offsetsY_[i]        = h * indexIncrement[1];
            offsetsZ_[i]        = h * indexIncrement[2];
         }
      }
   }

   void setY( real_t y )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         polynomials_[i].setY( y + offsetsY_[i] );
   }

   void setZ( real_t z )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         polynomials_[i].setZ( z + offsetsZ_[i] );
   }

   void setStartX( real_t x, real_t h, std::array< real_t, num_directions >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[i] = polynomials_[i].setStartX( x + offsetsX_[i], h );
   }

   void incrementEval( std::array< real_t, num_directions >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[i] = polynomials_[i].incrementEval();
   };

   void setOffset( SD d, real_t x, real_t y, real_t z )
   {
      WALBERLA_LOG_INFO_ON_ROOT("NOT_IMPLEMENTED");

      auto it = std::find( directions_.begin(), directions_.end(), d );
      if ( it == directions_.end() )
      WALBERLA_ABORT( "direction not defined" );
      auto idx       = it - directions_.begin();
      offsetsX_[idx] = x;
      offsetsY_[idx] = y;
      offsetsZ_[idx] = z;
   }

 private:
   std::array< SD, num_directions > directions_;

   std::vector< QuadrilateralPolynomial3DEvaluator > polynomials_;

   std::vector< real_t > offsetsX_;
   std::vector< real_t > offsetsY_;
   std::vector< real_t > offsetsZ_;
};

template < size_t num_directions >
class PolyStencil
{
 public:
   PolyStencil( const std::array< uint_t, 3 > degrees, const std::array< SD, num_directions >& directions )
   : directions_( directions )
   {
      for ( uint_t i = 0; i < num_directions; i += 1 )
      {
         polynomials_.emplace_back( degrees[0], degrees[1], degrees[2] );
         offsetsX_.push_back( 0 );
         offsetsY_.push_back( 0 );
         offsetsZ_.push_back( 0 );
      }
   }

   template < typename PolyListType >
   void setPolynomial( PolyListType& polylist )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         polynomials_[i].setPolynomial( polylist[directions_[i]] );
      }
   }

   template < typename PolyListType >
   void setPolynomialSymmetrical( PolyListType& polylist, real_t h )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         if ( polylist.contains( directions_[i] ) )
         {
            polynomials_[i].setPolynomial( polylist[directions_[i]] );
         }
         else
         {
            polynomials_[i].setPolynomial( polylist[opposite( directions_[i] )] );
            auto indexIncrement = toIndex( directions_[i] );
            offsetsX_[i]        = h * indexIncrement[0];
            offsetsY_[i]        = h * indexIncrement[1];
            offsetsZ_[i]        = h * indexIncrement[2];
         }
      }
   }

   void setY( real_t y )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         polynomials_[i].setY( y + offsetsY_[i] );
   }

   void setZ( real_t z )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         polynomials_[i].setZ( z + offsetsZ_[i] );
   }

   void setStartX( real_t x, real_t h, std::map< SD, real_t >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[directions_[i]] = polynomials_[i].setStartX( x + offsetsX_[i], h );
   }

   void incrementEval( std::map< SD, real_t >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[directions_[i]] = polynomials_[i].incrementEval();
   };

   void setOffset( SD d, real_t x, real_t y, real_t z )
   {
      auto it = std::find( directions_.begin(), directions_.end(), d );
      if ( it == directions_.end() )
         WALBERLA_ABORT( "direction not defined" );
      auto idx       = it - directions_.begin();
      offsetsX_[idx] = x;
      offsetsY_[idx] = y;
      offsetsZ_[idx] = z;
   }

 private:
   std::array< SD, num_directions > directions_;

   std::vector< QuadrilateralPolynomial3DEvaluator > polynomials_;

   std::vector< real_t > offsetsX_;
   std::vector< real_t > offsetsY_;
   std::vector< real_t > offsetsZ_;
};

template < typename FormType >
class AssembledStencil
{
 public:
   AssembledStencil( uint_t level, const Cell& cell, const FormType& form )
   : level_( level )
   , point_( 0, 0, 0 )
   , h_( 1. / real_c( levelinfo::num_microedges_per_edge( level ) ) )
   , cell_( cell )
   , form_( form )
   {
      form_.setGeometryMap( cell.getGeometryMap() );
   }

   void setY( real_t y ) { point_[1] = uint_c( std::round( y / h_ ) ); }

   void setZ( real_t z ) { point_[2] = uint_c( std::round( z / h_ ) ); }

   void setStartX( real_t x, real_t h, std::map< SD, real_t >& stencil )
   {
      h_        = h;
      point_[0] = uint_c( std::round( x / h_ ) );
      assemble( stencil );
   }

   void incrementEval( std::map< SD, real_t >& stencil )
   {
      point_[0] += 1;
      assemble( stencil );
   };

   void assemble( std::map< SD, real_t >& stencil )
   {
      stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( point_, cell_, level_, form_ );
   }

 private:
   uint_t level_;

   indexing::Index point_;

   real_t h_;

   const Cell cell_;

   FormType form_;
};

template < typename FormType >
class ConstantStencil
{
 public:
   ConstantStencil( uint_t level, const Cell& cell, const FormType& form )
   : level_( level )
   , cell_( cell )
   , form_( form )
   {
      form_.setGeometryMap( cell.getGeometryMap() );
      stencil_ = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 1, 1, 1 }, cell_, level_, form_ );
   }

   void setY( real_t ) {}

   void setZ( real_t ) {}

   void setStartX( real_t, real_t, std::map< SD, real_t >& stencil ) { assemble( stencil ); }

   void incrementEval( std::map< SD, real_t >& stencil ) { assemble( stencil ); };

   void assemble( std::map< SD, real_t >& stencil ) { stencil = stencil_; }

 private:
   uint_t level_;

   const Cell cell_;

   FormType form_;

   std::map< SD, real_t > stencil_;
};

template < typename FunctionType >
void apply_surrogate_operator_new( LDLTPolynomials&    polynomials,
                               uint_t              level,
                               bool                useSymmetry,
                               Cell&               cell,
                               const FunctionType& src_function,
                               const FunctionType& dst_function )
{
   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   const auto idx = [N_edge]( uint_t x, uint_t y, uint_t z ) { return indexing::macroCellIndex( N_edge, x, y, z ); };

   real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

   // unpack u and b
   auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
   auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

   PolyStencilNew< 15 > poly_stencil( polynomials.getDegrees(), allDirections );
   if ( useSymmetry )
      poly_stencil.setPolynomialSymmetrical( polynomials, h );
   else
      poly_stencil.setPolynomial( polynomials );

   std::array< real_t, 15 > a_stencil {};

   LIKWID_MARKER_START( "surrogate_new" );
   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      poly_stencil.setZ( h * real_c( z ) );
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         poly_stencil.setY( h * real_c( y ) );
         poly_stencil.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            poly_stencil.incrementEval( a_stencil );

            real_t sum = 0;
            sum += a_stencil[0] * src[idx( x, y, z )];
            sum += a_stencil[1] * src[idx( x - 1, y, z )];
            sum += a_stencil[2] * src[idx( x, y - 1, z )];
            sum += a_stencil[3] * src[idx( x + 1, y - 1, z )];
            sum += a_stencil[4] * src[idx( x - 1, y + 1, z + 1 )];
            sum += a_stencil[5] * src[idx( x, y + 1, z + 1 )];
            sum += a_stencil[6] * src[idx( x, y, z + 1 )];
            sum += a_stencil[7] * src[idx( x - 1, y, z + 1 )];
            sum += a_stencil[8] * src[idx( x + 1, y, z )];
            sum += a_stencil[9] * src[idx( x, y + 1, z )];
            sum += a_stencil[10] * src[idx( x - 1, y + 1, z )];
            sum += a_stencil[11] * src[idx( x + 1, y - 1, z + 1 )];
            sum += a_stencil[12] * src[idx( x, y - 1, z + 1 )];
            sum += a_stencil[13] * src[idx( x, y, z + 1 )];
            sum += a_stencil[14] * src[idx( x - 1, y, z + 1 )];
            dst[idx( x, y, z )] = sum;
         }
      }
   }
   LIKWID_MARKER_STOP( "surrogate_new" );
}

template < typename FunctionType >
void apply_surrogate_operator( LDLTPolynomials&    polynomials,
                               uint_t              level,
                               bool                useSymmetry,
                               Cell&               cell,
                               const FunctionType& src_function,
                               const FunctionType& dst_function )
{
   const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
      return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
   };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

   // unpack u and b
   auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
   auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

   PolyStencil< 15 > poly_stencil( polynomials.getDegrees(), allDirections );
   if ( useSymmetry )
      poly_stencil.setPolynomialSymmetrical( polynomials, h );
   else
      poly_stencil.setPolynomial( polynomials );

   std::map< SD, real_t > a_stencil;

   LIKWID_MARKER_START( "surrogate" );
   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      poly_stencil.setZ( h * real_c( z ) );
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         poly_stencil.setY( h * real_c( y ) );
         poly_stencil.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            poly_stencil.incrementEval( a_stencil );

            dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
            for ( auto d : allDirections )
               dst[cidx( x, y, z, SD::VERTEX_C )] += a_stencil[d] * src[cidx( x, y, z, d )];
         }
      }
   }
   LIKWID_MARKER_STOP( "surrogate" );
}

} // namespace dim3
} // namespace p1
} // namespace ldlt

using namespace hyteg;

using SD = stencilDirection;

static constexpr std::array< SD, 15 > allDirections = { SD::VERTEX_C,
                                                        SD::VERTEX_W,
                                                        SD::VERTEX_S,
                                                        SD::VERTEX_SE,
                                                        SD::VERTEX_BNW,
                                                        SD::VERTEX_BN,
                                                        SD::VERTEX_BC,
                                                        SD::VERTEX_BE,
                                                        SD::VERTEX_E,
                                                        SD::VERTEX_N,
                                                        SD::VERTEX_NW,
                                                        SD::VERTEX_TSE,
                                                        SD::VERTEX_TS,
                                                        SD::VERTEX_TC,
                                                        SD::VERTEX_TW };

constexpr inline uint_t index( uint_t N_edge, uint_t x, uint_t y, uint_t z, SD dir )
{
   switch ( dir )
   {
   case SD::VERTEX_C:
      return indexing::macroCellIndex( N_edge, x, y, z );
   case SD::VERTEX_W:
      return indexing::macroCellIndex( N_edge, x - 1, y, z );
   case SD::VERTEX_S:
      return indexing::macroCellIndex( N_edge, x, y - 1, z );
   case SD::VERTEX_SE:
      return indexing::macroCellIndex( N_edge, x + 1, y - 1, z );
   case SD::VERTEX_BNW:
      return indexing::macroCellIndex( N_edge, x - 1, y + 1, z - 1 );
   case SD::VERTEX_BN:
      return indexing::macroCellIndex( N_edge, x, y + 1, z - 1 );
   case SD::VERTEX_BC:
      return indexing::macroCellIndex( N_edge, x, y, z - 1 );
   case SD::VERTEX_BE:
      return indexing::macroCellIndex( N_edge, x + 1, y, z - 1 );
   case SD::VERTEX_E:
      return indexing::macroCellIndex( N_edge, x + 1, y, z );
   case SD::VERTEX_N:
      return indexing::macroCellIndex( N_edge, x, y + 1, z );
   case SD::VERTEX_NW:
      return indexing::macroCellIndex( N_edge, x - 1, y + 1, z );
   case SD::VERTEX_TSE:
      return indexing::macroCellIndex( N_edge, x + 1, y - 1, z + 1 );
   case SD::VERTEX_TS:
      return indexing::macroCellIndex( N_edge, x, y - 1, z + 1 );
   case SD::VERTEX_TC:
      return indexing::macroCellIndex( N_edge, x, y, z + 1 );
   case SD::VERTEX_TW:
      return indexing::macroCellIndex( N_edge, x - 1, y, z + 1 );
   default:
      return std::numeric_limits< uint_t >::max();
   }
}

constexpr inline uint_t index( uint_t N_edge, uint_t x, uint_t y, uint_t z, uint_t dir )
{
   switch ( dir )
   {
   case 0:
      return indexing::macroCellIndex( N_edge, x, y, z );
   case 1:
      return indexing::macroCellIndex( N_edge, x - 1, y, z );
   case 2:
      return indexing::macroCellIndex( N_edge, x, y - 1, z );
   case 3:
      return indexing::macroCellIndex( N_edge, x + 1, y - 1, z );
   case 4:
      return indexing::macroCellIndex( N_edge, x - 1, y + 1, z - 1 );
   case 5:
      return indexing::macroCellIndex( N_edge, x, y + 1, z - 1 );
   case 6:
      return indexing::macroCellIndex( N_edge, x, y, z - 1 );
   case 7:
      return indexing::macroCellIndex( N_edge, x + 1, y, z - 1 );
   case 8:
      return indexing::macroCellIndex( N_edge, x + 1, y, z );
   case 9:
      return indexing::macroCellIndex( N_edge, x, y + 1, z );
   case 10:
      return indexing::macroCellIndex( N_edge, x - 1, y + 1, z );
   case 11:
      return indexing::macroCellIndex( N_edge, x + 1, y - 1, z + 1 );
   case 12:
      return indexing::macroCellIndex( N_edge, x, y - 1, z + 1 );
   case 13:
      return indexing::macroCellIndex( N_edge, x, y, z + 1 );
   case 14:
      return indexing::macroCellIndex( N_edge, x - 1, y, z + 1 );
   default:
      return std::numeric_limits< uint_t >::max();
   }
}

constexpr inline uint_t stencilIndex( SD dir )
{
   if ( dir == SD::VERTEX_C )
      return 0;
   else if ( dir == SD::VERTEX_W )
      return 1;
   else if ( dir == SD::VERTEX_S )
      return 2;
   else if ( dir == SD::VERTEX_SE )
      return 3;
   else if ( dir == SD::VERTEX_BNW )
      return 4;
   else if ( dir == SD::VERTEX_BN )
      return 5;
   else if ( dir == SD::VERTEX_BC )
      return 6;
   else if ( dir == SD::VERTEX_BE )
      return 7;
   else if ( dir == SD::VERTEX_E )
      return 8;
   else if ( dir == SD::VERTEX_N )
      return 9;
   else if ( dir == SD::VERTEX_NW )
      return 10;
   else if ( dir == SD::VERTEX_TSE )
      return 11;
   else if ( dir == SD::VERTEX_TS )
      return 12;
   else if ( dir == SD::VERTEX_TC )
      return 13;
   else if ( dir == SD::VERTEX_TW )
      return 14;
   else
      return std::numeric_limits< uint_t >::max();
}

constexpr inline SD indexToDirection( uint_t idx )
{
   if ( idx == 0 )
      return SD::VERTEX_C;
   else if ( idx == 1 )
      return SD::VERTEX_W;
   else if ( idx == 2 )
      return SD::VERTEX_S;
   else if ( idx == 3 )
      return SD::VERTEX_SE;
   else if ( idx == 4 )
      return SD::VERTEX_BNW;
   else if ( idx == 5 )
      return SD::VERTEX_BN;
   else if ( idx == 6 )
      return SD::VERTEX_BC;
   else if ( idx == 7 )
      return SD::VERTEX_BE;
   else if ( idx == 8 )
      return SD::VERTEX_E;
   else if ( idx == 9 )
      return SD::VERTEX_N;
   else if ( idx == 10 )
      return SD::VERTEX_NW;
   else if ( idx == 11 )
      return SD::VERTEX_TSE;
   else if ( idx == 12 )
      return SD::VERTEX_TS;
   else if ( idx == 13 )
      return SD::VERTEX_TC;
   else if ( idx == 14 )
      return SD::VERTEX_TW;
   else
      return SD::VERTEX_C;
}

}

void runPerf(){
   using namespace hyteg;

   std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0.0, 0.0, 0.0 } ),
                                                hyteg::Point3D( { 1.0, 0.0, 0.0 } ),
                                                hyteg::Point3D( { 0.0, 1.0, 0.0 } ),
                                                hyteg::Point3D( { 0.0, 0.0, 1.0 } ) };
   // hyteg::MeshInfo                 meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

   hyteg::MeshInfo meshInfo = hyteg::MeshInfo::meshCuboid(hyteg::Point3D({0, 0, 0}), hyteg::Point3D({1, 1, 1}), 1, 1, 1);

   auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
       meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );
   const auto storage = std::make_shared< PrimitiveStorage >( *setupStorage );

   const uint_t level = 8;

   P1Function< real_t > src( "src", storage, level, level );
   src.interpolate( [](auto p){ return p[0] + p[1] + p[2]; }, level, All );
   P1Function< real_t > dst( "dst", storage, level, level );

   uint_t maxIter = 2;

   ldlt::p1::dim3::LDLTPolynomials polys({8, 8, 8}, allDirections);

   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      for ( const auto& cit : storage->getCells())
      {
         auto& cell = *cit.second;
         ldlt::p1::dim3::apply_surrogate_operator( polys, level, false, cell, src, dst );
         if ( cell.getData( dst.getCellDataID() )->getPointer( level )[0] > 100 )
            WALBERLA_LOG_INFO_ON_ROOT( "op1 " << dst.getMaxMagnitude( level, All, true ) );
      }
      for ( const auto& cit : storage->getCells())
      {
         auto& cell = *cit.second;
         ldlt::p1::dim3::apply_surrogate_operator_new( polys, level, false, cell, src, dst );
         // toy_matmul_3( level, src, dst );
         if ( cell.getData( dst.getCellDataID() )->getPointer( level )[0] > 100 )
            WALBERLA_LOG_INFO_ON_ROOT( "op3 " << dst.getMaxMagnitude( level, All, true ) );
      }
      for ( const auto& cit : storage->getCells())
      {
         auto& cell = *cit.second;
         // toy_matmul_6( level, src, dst );
         if ( cell.getData( dst.getCellDataID() )->getPointer( level )[levelinfo::num_microvertices_per_cell( level ) / 2] > 100 )
            WALBERLA_LOG_INFO_ON_ROOT( "op6 " << dst.getMaxMagnitude( level, All, true ) );
      }
   }
}

int main( int argc, char** argv )
{
   LIKWID_MARKER_INIT;

   walberla::Environment env( argc, argv );
   walberla::mpi::MPIManager::instance()->useWorldComm();

   runPerf();

   LIKWID_MARKER_CLOSE;
}
