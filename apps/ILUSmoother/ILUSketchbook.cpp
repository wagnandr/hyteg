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

#include "hyteg/LikwidWrapper.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;
using walberla::math::pi;

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

void toy_matmul_0( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      /*
      const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
         return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
      };
      */

      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      const auto cidx = [N_edge]( uint_t x, uint_t y, uint_t z, SD dir ) {
         switch ( dir )
         {
         case SD::VERTEX_C:
            return indexing::macroCellIndex( N_edge, x, y, z );
         case SD::VERTEX_W:
            return indexing::macroCellIndex( N_edge, x - 1, y, z );
         case SD::VERTEX_E:
            return indexing::macroCellIndex( N_edge, x + 1, y, z );
         case SD::VERTEX_N:
            return indexing::macroCellIndex( N_edge, x, y + 1, z );
         case SD::VERTEX_S:
            return indexing::macroCellIndex( N_edge, x, y - 1, z );
         case SD::VERTEX_NW:
            return indexing::macroCellIndex( N_edge, x - 1, y + 1, z );
         case SD::VERTEX_SE:
            return indexing::macroCellIndex( N_edge, x + 1, y - 1, z );
         case SD::VERTEX_TC:
            return indexing::macroCellIndex( N_edge, x, y, z + 1 );
         case SD::VERTEX_TW:
            return indexing::macroCellIndex( N_edge, x - 1, y, z + 1 );
         case SD::VERTEX_TS:
            return indexing::macroCellIndex( N_edge, x, y - 1, z + 1 );
         case SD::VERTEX_TSE:
            return indexing::macroCellIndex( N_edge, x + 1, y - 1, z + 1 );
         case SD::VERTEX_BC:
            return indexing::macroCellIndex( N_edge, x, y, z - 1 );
         case SD::VERTEX_BN:
            return indexing::macroCellIndex( N_edge, x, y + 1, z - 1 );
         case SD::VERTEX_BE:
            return indexing::macroCellIndex( N_edge, x + 1, y, z - 1 );
         case SD::VERTEX_BNW:
            return indexing::macroCellIndex( N_edge, x - 1, y + 1, z - 1 );
         default:
            return std::numeric_limits< uint_t >::max();
         }
      };

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil_map =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );
      std::array< real_t, 15 > stencil{ 0 };

      for ( uint_t d = 0; d < 15; d += 1 )
         stencil[d] = stencil_map[allDirections[d]];

      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // std::map< SD, real_t > stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );
               dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
               for ( uint_t d = 0; d < allDirections.size(); d += 1 )
                  dst[cidx( x, y, z, SD::VERTEX_C )] += stencil[d] * src[cidx( x, y, z, allDirections[d] )];
            }
         }
      }
   }
}

void toy_matmul_1( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      /*
      const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
         return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
      };
      */

      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      const auto cidx = [N_edge]( uint_t x, uint_t y, uint_t z, SD dir ) {
         switch ( dir )
         {
         case SD::VERTEX_C:
            return indexing::macroCellIndex( N_edge, x, y, z );
         case SD::VERTEX_W:
            return indexing::macroCellIndex( N_edge, x - 1, y, z );
         case SD::VERTEX_E:
            return indexing::macroCellIndex( N_edge, x + 1, y, z );
         case SD::VERTEX_N:
            return indexing::macroCellIndex( N_edge, x, y + 1, z );
         case SD::VERTEX_S:
            return indexing::macroCellIndex( N_edge, x, y - 1, z );
         case SD::VERTEX_NW:
            return indexing::macroCellIndex( N_edge, x - 1, y + 1, z );
         case SD::VERTEX_SE:
            return indexing::macroCellIndex( N_edge, x + 1, y - 1, z );
         case SD::VERTEX_TC:
            return indexing::macroCellIndex( N_edge, x, y, z + 1 );
         case SD::VERTEX_TW:
            return indexing::macroCellIndex( N_edge, x - 1, y, z + 1 );
         case SD::VERTEX_TS:
            return indexing::macroCellIndex( N_edge, x, y - 1, z + 1 );
         case SD::VERTEX_TSE:
            return indexing::macroCellIndex( N_edge, x + 1, y - 1, z + 1 );
         case SD::VERTEX_BC:
            return indexing::macroCellIndex( N_edge, x, y, z - 1 );
         case SD::VERTEX_BN:
            return indexing::macroCellIndex( N_edge, x, y + 1, z - 1 );
         case SD::VERTEX_BE:
            return indexing::macroCellIndex( N_edge, x + 1, y, z - 1 );
         case SD::VERTEX_BNW:
            return indexing::macroCellIndex( N_edge, x - 1, y + 1, z - 1 );
         default:
            return std::numeric_limits< uint_t >::max();
         }
      };

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );

      LIKWID_MARKER_START( "region1" );
      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // std::map< SD, real_t > stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );
               dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
               for ( auto d : allDirections )
                  dst[cidx( x, y, z, SD::VERTEX_C )] += stencil[d] * src[cidx( x, y, z, d )];
            }
         }
      }
      LIKWID_MARKER_STOP( "region1" );
   }
}

void toy_matmul_2( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   const auto idx = [level]( uint_t x, uint_t y, uint_t z ) { return vertexdof::macrocell::index( level, x, y, z ); };

   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil_map =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );
      std::array< real_t, 15 > stencil{ 0 };

      for ( uint_t d = 0; d < 15; d += 1 )
         stencil[d] = stencil_map[allDirections[d]];

      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // std::map< SD, real_t > stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );

               // dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
               // for ( uint_t d = 0; d < 15; d += 1 )
               //    dst[cidx( x, y, z, SD::VERTEX_C )] += stencil[d] * src[cidx( x, y, z, allDirections[d] )];

               real_t sum = 0;
               sum += stencil[0] * src[idx( x, y, z )];
               sum += stencil[1] * src[idx( x - 1, y, z )];
               sum += stencil[2] * src[idx( x, y - 1, z )];
               sum += stencil[3] * src[idx( x + 1, y - 1, z )];
               sum += stencil[4] * src[idx( x - 1, y + 1, z + 1 )];
               sum += stencil[5] * src[idx( x, y + 1, z + 1 )];
               sum += stencil[6] * src[idx( x, y, z + 1 )];
               sum += stencil[7] * src[idx( x - 1, y, z + 1 )];
               sum += stencil[8] * src[idx( x + 1, y, z )];
               sum += stencil[9] * src[idx( x, y + 1, z )];
               sum += stencil[10] * src[idx( x - 1, y + 1, z )];
               sum += stencil[11] * src[idx( x + 1, y - 1, z + 1 )];
               sum += stencil[12] * src[idx( x, y - 1, z + 1 )];
               sum += stencil[13] * src[idx( x, y, z + 1 )];
               sum += stencil[14] * src[idx( x - 1, y, z + 1 )];
               dst[idx( x, y, z )] = sum;
            }
         }
      }
   }
}

void toy_matmul_3( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      const auto idx = [N_edge]( uint_t x, uint_t y, uint_t z ) { return indexing::macroCellIndex( N_edge, x, y, z ); };

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil_map =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );
      std::array< real_t, 15 > stencil{ 0 };

      for ( uint_t d = 0; d < 15; d += 1 )
         stencil[d] = stencil_map[allDirections[d]];

      LIKWID_MARKER_START( "region3" );
      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // std::map< SD, real_t > stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );

               // dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
               // for ( uint_t d = 0; d < 15; d += 1 )
               //    dst[cidx( x, y, z, SD::VERTEX_C )] += stencil[d] * src[cidx( x, y, z, allDirections[d] )];

               real_t sum = 0;
               sum += stencil[0] * src[idx( x, y, z )];
               sum += stencil[1] * src[idx( x - 1, y, z )];
               sum += stencil[2] * src[idx( x, y - 1, z )];
               sum += stencil[3] * src[idx( x + 1, y - 1, z )];
               sum += stencil[4] * src[idx( x - 1, y + 1, z + 1 )];
               sum += stencil[5] * src[idx( x, y + 1, z + 1 )];
               sum += stencil[6] * src[idx( x, y, z + 1 )];
               sum += stencil[7] * src[idx( x - 1, y, z + 1 )];
               sum += stencil[8] * src[idx( x + 1, y, z )];
               sum += stencil[9] * src[idx( x, y + 1, z )];
               sum += stencil[10] * src[idx( x - 1, y + 1, z )];
               sum += stencil[11] * src[idx( x + 1, y - 1, z + 1 )];
               sum += stencil[12] * src[idx( x, y - 1, z + 1 )];
               sum += stencil[13] * src[idx( x, y, z + 1 )];
               sum += stencil[14] * src[idx( x - 1, y, z + 1 )];
               dst[idx( x, y, z )] = sum;
            }
         }
      }
      LIKWID_MARKER_STOP( "region3" );
   }
}

void toy_matmul_4( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      const auto idx = [N_edge]( uint_t x, uint_t y, uint_t z ) { return indexing::macroCellIndex( N_edge, x, y, z ); };

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil_map =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );
      std::array< real_t, 15 > stencil{ 0 };

      for ( uint_t d = 0; d < 15; d += 1 )
         stencil[d] = stencil_map[allDirections[d]];

      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               double sum = 0;
               sum += 0.0 * src[idx( x, y, z )];
               sum += 1.0 * src[idx( x - 1, y, z )];
               sum += 2.0 * src[idx( x, y - 1, z )];
               sum += 3.0 * src[idx( x + 1, y - 1, z )];
               sum += 4.0 * src[idx( x - 1, y + 1, z + 1 )];
               sum += 5.0 * src[idx( x, y + 1, z + 1 )];
               sum += 6.0 * src[idx( x, y, z + 1 )];
               sum += 7.0 * src[idx( x - 1, y, z + 1 )];
               sum += 8.0 * src[idx( x + 1, y, z )];
               sum += 9.0 * src[idx( x, y + 1, z )];
               sum += 10.0 * src[idx( x - 1, y + 1, z )];
               sum += 11.0 * src[idx( x + 1, y - 1, z + 1 )];
               sum += 12.0 * src[idx( x, y - 1, z + 1 )];
               sum += 13.0 * src[idx( x, y, z + 1 )];
               sum += 14.0 * src[idx( x - 1, y, z + 1 )];
               dst[idx( x, y, z )] = sum;
            }
         }
      }
   }
}

void toy_matmul_5( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      const auto idx = [N_edge]( uint_t x, uint_t y, uint_t z ) { return indexing::macroCellIndex( N_edge, x, y, z ); };

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil_map =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );
      std::array< real_t, 15 > stencil{ 0 };

      for ( uint_t d = 0; d < 15; d += 1 )
         stencil[d] = stencil_map[allDirections[d]];

      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // std::map< SD, real_t > stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );

               // dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
               // for ( uint_t d = 0; d < 15; d += 1 )
               //    dst[cidx( x, y, z, SD::VERTEX_C )] += stencil[d] * src[cidx( x, y, z, allDirections[d] )];

               std::array< real_t, 15 > src_array{ 0 };
               src_array[0]  = src[idx( x, y, z )];
               src_array[1]  = src[idx( x - 1, y, z )];
               src_array[2]  = src[idx( x, y - 1, z )];
               src_array[3]  = src[idx( x + 1, y - 1, z )];
               src_array[4]  = src[idx( x - 1, y + 1, z + 1 )];
               src_array[5]  = src[idx( x, y + 1, z + 1 )];
               src_array[6]  = src[idx( x, y, z + 1 )];
               src_array[7]  = src[idx( x - 1, y, z + 1 )];
               src_array[8]  = src[idx( x + 1, y, z )];
               src_array[9]  = src[idx( x, y + 1, z )];
               src_array[10] = src[idx( x - 1, y + 1, z )];
               src_array[11] = src[idx( x + 1, y - 1, z + 1 )];
               src_array[12] = src[idx( x, y - 1, z + 1 )];
               src_array[13] = src[idx( x, y, z + 1 )];
               src_array[14] = src[idx( x - 1, y, z + 1 )];

               real_t sum = 0;
               for ( uint_t d = 0; d < 15; d += 1 )
                  sum += stencil[d] * src_array[d];

               dst[idx( x, y, z )] = sum;
            }
         }
      }
   }
}

void toy_matmul_6( uint_t level, const P1Function< real_t >& src_function, const P1Function< real_t >& dst_function )
{
   for ( auto& cit : src_function.getStorage()->getCells() )
   {
      Cell& cell = *cit.second;

      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      //const auto cidx = [N_edge]( uint_t x, uint_t y, uint_t z, uint_t dir ) { return index( N_edge, x, y, z, dir ); };
      const auto cidx = [N_edge]( uint_t x, uint_t y, uint_t z, SD dir ) { return index( N_edge, x, y, z, dir ); };

      const auto sidx = []( SD dir ) { return stencilIndex( dir ); };

      // const auto idx = [N_edge]( uint_t x, uint_t y, uint_t z ) { return indexing::macroCellIndex( N_edge, x, y, z ); };

      // unpack u and b
      auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
      auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

      using FormType = forms::p1_div_k_grad_blending_q3;
      FormType form( []( auto ) { return 1.; }, []( auto ) { return 1.; } );
      form.setGeometryMap( cell.getGeometryMap() );

      std::map< SD, real_t > stencil_map =
          P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 0, 0, 0 }, cell, level, form );
      std::array< real_t, 15 > stencil{ 0 };

      for ( uint_t d = 0; d < 15; d += 1 )
         stencil[d] = stencil_map[allDirections[d]];

      // LIKWID_MARKER_START( "region6" );
      for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
      {
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // std::map< SD, real_t > stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );

               real_t sum = 0;

               /*
               dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
               for ( uint_t d = 0; d < 15; d += 1 )
                  dst[cidx( x, y, z, SD::VERTEX_C )] += stencil[d] * src[cidx( x, y, z, allDirections[d] )];
               */

               sum += stencil[sidx( SD::VERTEX_C )] * src[cidx( x, y, z, SD::VERTEX_C )];
               sum += stencil[sidx( SD::VERTEX_W )] * src[cidx( x, y, z, SD::VERTEX_W )];
               sum += stencil[sidx( SD::VERTEX_S )] * src[cidx( x, y, z, SD::VERTEX_S )];
               sum += stencil[sidx( SD::VERTEX_SE )] * src[cidx( x, y, z, SD::VERTEX_SE )];
               sum += stencil[sidx( SD::VERTEX_BNW )] * src[cidx( x, y, z, SD::VERTEX_BNW )];
               sum += stencil[sidx( SD::VERTEX_BN )] * src[cidx( x, y, z, SD::VERTEX_BN )];
               sum += stencil[sidx( SD::VERTEX_BC )] * src[cidx( x, y, z, SD::VERTEX_BC )];
               sum += stencil[sidx( SD::VERTEX_BE )] * src[cidx( x, y, z, SD::VERTEX_BE )];
               sum += stencil[sidx( SD::VERTEX_E )] * src[cidx( x, y, z, SD::VERTEX_E )];
               sum += stencil[sidx( SD::VERTEX_N )] * src[cidx( x, y, z, SD::VERTEX_N )];
               sum += stencil[sidx( SD::VERTEX_NW )] * src[cidx( x, y, z, SD::VERTEX_NW )];
               sum += stencil[sidx( SD::VERTEX_TSE )] * src[cidx( x, y, z, SD::VERTEX_TSE )];
               sum += stencil[sidx( SD::VERTEX_TS )] * src[cidx( x, y, z, SD::VERTEX_TS )];
               sum += stencil[sidx( SD::VERTEX_TC )] * src[cidx( x, y, z, SD::VERTEX_TC )];
               sum += stencil[sidx( SD::VERTEX_TW )] * src[cidx( x, y, z, SD::VERTEX_TW )];

               /*
               for (auto d: allDirections)
                  sum += stencil[sidx( d )] * src[cidx( x, y, z, d )];
               */

               /*
               for ( uint_t d = 0; d < 15; d += 1 )
                  //   sum += stencil[d] * src[cidx( x, y, z, allDirections[d] )];
                  sum += stencil[d] * src[cidx( x, y, z, d )];
                  */

               // real_t sum = 0;
               // for (uint_t d = 0; d < stencil.size(); d += 1)
               //     sum += stencil[d] * src[cidx( x, y, z, allDirections[d] )];

               dst[index( N_edge, x, y, z, SD::VERTEX_C )] = sum;
            }
         }
      }
      // LIKWID_MARKER_STOP( "region6" );
   }
}

void runPerf(){
   std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0.0, 0.0, 0.0 } ),
                                                hyteg::Point3D( { 1.0, 0.0, 0.0 } ),
                                                hyteg::Point3D( { 0.0, 1.0, 0.0 } ),
                                                hyteg::Point3D( { 0.0, 0.0, 1.0 } ) };
   hyteg::MeshInfo                 meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

   auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
       meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );
   const auto storage = std::make_shared< PrimitiveStorage >( *setupStorage );

   const uint_t level = 10;

   P1Function< real_t > src( "src", storage, level, level );
   src.interpolate( [](auto p){ return p[0] + p[1] + p[2]; }, level, All );
   P1Function< real_t > dst( "dst", storage, level, level );

   P1Function< real_t > src2( "src", storage, level, level );
   src.interpolate( [](auto p){ return p[0] + p[1] + p[2]; }, level, All );
   P1Function< real_t > dst2( "dst", storage, level, level );

   uint_t maxIter = 1;

   auto& cell = *storage->getCell( storage->getCellIDs()[0] );

   /*
   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      toy_matmul_1( level, src, dst );
      // WALBERLA_LOG_INFO_ON_ROOT( "op1 " << dst.getMaxMagnitude( level, All, true ) );
   }
    */

   /*
   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      toy_matmul_2( level, src, dst );
      if ( cell.getData(dst.getCellDataID())->getPointer(level)[0] > 100)
         WALBERLA_LOG_INFO_ON_ROOT( "op3 " << dst.getMaxMagnitude( level, All, true ) );
   }
    */

   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      toy_matmul_6( level, src, dst );
      if ( cell.getData( dst.getCellDataID() )->getPointer( level )[levelinfo::num_microvertices_per_cell( level ) / 2] > 100 )
         WALBERLA_LOG_INFO_ON_ROOT( "op6 " << dst.getMaxMagnitude( level, All, true ) );

      toy_matmul_6( level, src2, dst2 );
      if ( cell.getData( dst2.getCellDataID() )->getPointer( level )[levelinfo::num_microvertices_per_cell( level ) / 2] > 100 )
         WALBERLA_LOG_INFO_ON_ROOT( "op6 " << dst2.getMaxMagnitude( level, All, true ) );
   }

   likwid_pinProcess(0);
   LIKWID_MARKER_INIT;
   LIKWID_MARKER_REGISTER( "matmul6" );
   LIKWID_MARKER_START( "test" );
   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      /*
      toy_matmul_1( level, src, dst );
      if ( cell.getData( dst.getCellDataID() )->getPointer( level )[0] > 100 )
         WALBERLA_LOG_INFO_ON_ROOT( "op1 " << dst.getMaxMagnitude( level, All, true ) );

      toy_matmul_3( level, src, dst );
      if ( cell.getData( dst.getCellDataID() )->getPointer( level )[0] > 100 )
         WALBERLA_LOG_INFO_ON_ROOT( "op3 " << dst.getMaxMagnitude( level, All, true ) );
      */

      LIKWID_MARKER_START( "matmul6" );
      toy_matmul_6( level, src, dst );
      LIKWID_MARKER_STOP( "matmul6" );

      if ( cell.getData( dst.getCellDataID() )->getPointer( level )[levelinfo::num_microvertices_per_cell( level ) / 2] > 100 )
         WALBERLA_LOG_INFO_ON_ROOT( "op6 " << dst.getMaxMagnitude( level, All, true ) );

      toy_matmul_6( level, src2, dst2 );

      if ( cell.getData( dst2.getCellDataID() )->getPointer( level )[levelinfo::num_microvertices_per_cell( level ) / 2] > 100 )
         WALBERLA_LOG_INFO_ON_ROOT( "op6 " << dst2.getMaxMagnitude( level, All, true ) );
   }
   LIKWID_MARKER_STOP( "test" );
   LIKWID_MARKER_CLOSE;

   /*

   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      toy_matmul_4( level, src, dst );
      // WALBERLA_LOG_INFO_ON_ROOT( "op4 " << dst.getMaxMagnitude( level, All, true ) );
   }
    */
}

int main( int argc, char** argv )
{
   walberla::Environment env( argc, argv );
   walberla::mpi::MPIManager::instance()->useWorldComm();

}
