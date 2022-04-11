/*
* Copyright (c) 2021 Andreas Wagner.
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

#include "hyteg/memory/FunctionMemory.hpp"
#include "hyteg/p1functionspace/P1Elements.hpp"
#include "hyteg/p1functionspace/VertexDoFMacroFace.hpp"
#include "hyteg/primitives/Primitive.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"

#include "PrimitiveSmoothers.hpp"

namespace hyteg {

template < typename P1Form >
void assemble_variableStencil_edge( const std::shared_ptr< PrimitiveStorage > storage,
                                    uint_t                                    level,
                                    const Edge&                               edge,
                                    P1Form&                                   form,
                                    real_t*                                   edge_stencil,
                                    const uint_t                              i )
{
   using namespace vertexdof::macroedge;

   const auto stencilSize = hyteg::vertexDoFMacroEdgeStencilMemorySize( level, edge );

   // 3D version (old version)
   if ( storage->hasGlobalCells() )
   {
      // old linear stencil still used at certain points (e.g. in P2ConstantOperator)
      auto stencil = P1Elements::P1Elements3D::assembleP1LocalStencil_new< P1Form >(
          storage, edge, indexing::Index( i, 0, 0 ), level, form );

      WALBERLA_ASSERT_EQUAL( stencilSize, stencil.size() );

      for ( uint_t j = 0; j < stencilSize; j++ )
      {
         edge_stencil[j] = stencil[j];
      }
   }
   // 2D version
   else
   {
      WALBERLA_ABORT( "not implemented" );
   }
}

template < typename P1Form >
void smooth_sor_edge3D( const std::shared_ptr< PrimitiveStorage >                storage,
                        Edge&                                                    edge,
                        P1Form&                                                  form,
                        const PrimitiveDataID< FunctionMemory< real_t >, Edge >& dstId,
                        const PrimitiveDataID< FunctionMemory< real_t >, Edge >& rhsId,
                        const uint_t&                                            level,
                        real_t                                                   relax,
                        const bool&                                              backwards = false )
{
   using sD       = stencilDirection;
   size_t rowsize = levelinfo::num_microvertices_per_edge( level );

   std::vector< real_t > opr_data_vec( hyteg::vertexDoFMacroEdgeStencilMemorySize( level, edge ), 0. );
   auto                  opr_data = &opr_data_vec.at( 0 );
   auto                  rhs      = edge.getData( rhsId )->getPointer( level );
   auto                  dst      = edge.getData( dstId )->getPointer( level );

   const auto stencilIdxW = vertexdof::macroedge::stencilIndexOnEdge( sD::VERTEX_W );
   const auto stencilIdxC = vertexdof::macroedge::stencilIndexOnEdge( sD::VERTEX_C );
   const auto stencilIdxE = vertexdof::macroedge::stencilIndexOnEdge( sD::VERTEX_E );

   real_t tmp;

   const int start = backwards ? (int) rowsize - 2 : 1;
   const int stop  = backwards ? 0 : (int) rowsize - 1;
   const int incr  = backwards ? -1 : 1;

   for ( int ii = start; ii != stop; ii += incr )
   {
      const uint_t i = uint_c( ii );

      form.setGeometryMap( edge.getGeometryMap() );
      assemble_variableStencil_edge( storage, level, edge, form, opr_data, i );
      auto invCenterWeight = 1.0 / opr_data[stencilIdxC];

      const auto dofIdxW = vertexdof::macroedge::indexFromVertex( level, i, sD::VERTEX_W );
      const auto dofIdxC = vertexdof::macroedge::indexFromVertex( level, i, sD::VERTEX_C );
      const auto dofIdxE = vertexdof::macroedge::indexFromVertex( level, i, sD::VERTEX_E );

      tmp = rhs[dofIdxC];

      tmp -= opr_data[stencilIdxW] * dst[dofIdxW] + opr_data[stencilIdxE] * dst[dofIdxE];

      for ( uint_t neighborFace = 0; neighborFace < edge.getNumNeighborFaces(); neighborFace++ )
      {
         const auto stencilIdxWNeighborFace = vertexdof::macroedge::stencilIndexOnNeighborFace( sD::VERTEX_W, neighborFace );
         const auto stencilIdxENeighborFace = vertexdof::macroedge::stencilIndexOnNeighborFace( sD::VERTEX_E, neighborFace );
         const auto stencilWeightW          = opr_data[stencilIdxWNeighborFace];
         const auto stencilWeightE          = opr_data[stencilIdxENeighborFace];
         const auto dofIdxWNeighborFace =
             vertexdof::macroedge::indexFromVertexOnNeighborFace( level, i, neighborFace, sD::VERTEX_W );
         const auto dofIdxENeighborFace =
             vertexdof::macroedge::indexFromVertexOnNeighborFace( level, i, neighborFace, sD::VERTEX_E );
         tmp -= stencilWeightW * dst[dofIdxWNeighborFace] + stencilWeightE * dst[dofIdxENeighborFace];
      }

      for ( uint_t neighborCell = 0; neighborCell < edge.getNumNeighborCells(); neighborCell++ )
      {
         const auto stencilIdx = vertexdof::macroedge::stencilIndexOnNeighborCell( neighborCell, edge.getNumNeighborFaces() );
         const auto dofIdx =
             vertexdof::macroedge::indexFromVertexOnNeighborCell( level, i, neighborCell, edge.getNumNeighborFaces() );
         tmp -= opr_data[stencilIdx] * dst[dofIdx];
      }

      dst[dofIdxC] = ( 1.0 - relax ) * dst[dofIdxC] + relax * invCenterWeight * tmp;
   }
}

template < typename OperatorType, typename P1Form >
class GSEdgeSmoother : public EdgeSmoother< OperatorType >
{
 public:
   using FSFunctionType = typename OperatorType::srcType;

   explicit GSEdgeSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel, P1Form& form )
   : storage_( std::move( storage ) )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , form_( form )
   {}

   void smooth( const OperatorType&, uint_t level, Edge& edge, const FSFunctionType& x, const FSFunctionType& b ) override
   {
      smooth_sor_edge3D( storage_, edge, form_, x.getEdgeDataID(), b.getEdgeDataID(), level, 1., false );
   }

   void smooth_backwards( const OperatorType&,
                          uint_t                level,
                          Edge&                 edge,
                          const FSFunctionType& x,
                          const FSFunctionType& b ) override
   {
      smooth_sor_edge3D( storage_, edge, form_, x.getEdgeDataID(), b.getEdgeDataID(), level, 1., true );
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;
   DoFType                             flag_;
   size_t                              minLevel_;
   size_t                              maxLevel_;
   P1Form                              form_;
};

} // namespace hyteg