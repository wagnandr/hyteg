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
void assemble_variableStencil_face3D( const std::shared_ptr< PrimitiveStorage >& storage,
                                      const Face&                                face,
                                      P1Form&                                    form,
                                      vertexdof::macroface::StencilMap_T&        face_stencil,
                                      const uint_t                               level,
                                      const uint_t                               i,
                                      const uint_t                               j )
{
   WALBERLA_ASSERT( storage->hasGlobalCells() );

   for ( uint_t neighborCellID = 0; neighborCellID < face.getNumNeighborCells(); ++neighborCellID )
   {
      auto neighborCell = storage->getCell( face.neighborCells().at( neighborCellID ) );
      auto vertexAssemblyIndexInCell =
          vertexdof::macroface::getIndexInNeighboringMacroCell( { i, j, 0 }, face, neighborCellID, *storage, level );
      face_stencil[neighborCellID] = P1Elements::P1Elements3D::assembleP1LocalStencilNew_new< P1Form >(
          storage, *neighborCell, vertexAssemblyIndexInCell, level, form );
   }
}

// apply the sor-operator to all DoF on a given macro-face
template < typename P1Form >
void smooth_sor_face3D( const std::shared_ptr< PrimitiveStorage >&               storage,
                        const Face&                                              face,
                        P1Form&                                                  form,
                        const PrimitiveDataID< FunctionMemory< real_t >, Face >& dstId,
                        const PrimitiveDataID< FunctionMemory< real_t >, Face >& rhsId,
                        const uint_t&                                            level,
                        real_t                                                   relax,
                        const bool&                                              backwards = false )
{
   vertexdof::macroface::StencilMap_T opr_data;

   auto rhs = face.getData( rhsId )->getPointer( level );
   auto dst = face.getData( dstId )->getPointer( level );

   real_t centerWeight = real_c( 0 );

   for ( uint_t neighborCellIdx = 0; neighborCellIdx < face.getNumNeighborCells(); neighborCellIdx++ )
   {
      centerWeight += opr_data[neighborCellIdx][{ 0, 0, 0 }];
   }

   auto invCenterWeight = 1.0 / centerWeight;

   // todo loop ij
   // for ( const auto& idxIt : vertexdof::macroface::Iterator( level, 1 ) )
   const int ydir   = backwards ? -1 : 1;
   const int ystart = backwards ? static_cast< int >( levelinfo::num_microvertices_per_edge( level ) ) - 2 : 1;
   for ( int y = ystart; 1 <= y && y < static_cast< int >( levelinfo::num_microvertices_per_edge( level ) ) - 1; y += ydir )
   {
      const int xdir   = backwards ? -1 : 1;
      const int xstart = backwards ? static_cast< int >( levelinfo::num_microvertices_per_edge( level ) ) - 2 - y : 1;

      for ( int x = xstart; 1 <= x && x < static_cast< int >( levelinfo::num_microvertices_per_edge( level ) ) - y - 1; x += xdir )
      {
         real_t tmp = rhs[vertexdof::macroface::index( level, uint_c( x ), uint_c( y ) )];

         assemble_variableStencil_face3D( storage, face, form, opr_data, level, x, y );
         centerWeight = real_c( 0 );

         for ( uint_t neighborCellIdx = 0; neighborCellIdx < face.getNumNeighborCells(); neighborCellIdx++ )
         {
            centerWeight += opr_data[neighborCellIdx][{ 0, 0, 0 }];
         }

         invCenterWeight = 1.0 / centerWeight;

         for ( uint_t neighborCellIdx = 0; neighborCellIdx < face.getNumNeighborCells(); neighborCellIdx++ )
         {
            auto neighborCell      = storage->getCell( face.neighborCells().at( neighborCellIdx ) );
            auto centerIndexInCell = vertexdof::macroface::getIndexInNeighboringMacroCell(
                indexing::Index( uint_c( x ), uint_c( y ), 0 ), face, neighborCellIdx, *storage, level );

            for ( auto stencilIt : opr_data[neighborCellIdx] )
            {
               if ( stencilIt.first == indexing::IndexIncrement( { 0, 0, 0 } ) )
                  continue;

               auto weight               = stencilIt.second;
               auto leafIndexInMacroCell = centerIndexInCell + stencilIt.first;
               auto leafIndexInMacroFace = vertexdof::macrocell::getIndexInNeighboringMacroFace(
                   leafIndexInMacroCell, *neighborCell, neighborCell->getLocalFaceID( face.getID() ), *storage, level );

               uint_t leafArrayIndexInMacroFace;

               if ( leafIndexInMacroFace.z() == 0 )
               {
                  leafArrayIndexInMacroFace =
                      vertexdof::macroface::index( level, leafIndexInMacroFace.x(), leafIndexInMacroFace.y() );
               }
               else
               {
                  WALBERLA_ASSERT_EQUAL( leafIndexInMacroFace.z(), 1 );
                  leafArrayIndexInMacroFace =
                      vertexdof::macroface::index( level, leafIndexInMacroFace.x(), leafIndexInMacroFace.y(), neighborCellIdx );
               }

               tmp -= weight * dst[leafArrayIndexInMacroFace];
            }
         }

         dst[vertexdof::macroface::index( level, uint_c( x ), uint_c( y ) )] =
             ( 1.0 - relax ) * dst[vertexdof::macroface::index( level, uint_c( x ), uint_c( y ) )] +
             relax * tmp * invCenterWeight;
      }
   }
}

template < typename OperatorType, typename P1Form >
class GSFaceSmoother : public FaceSmoother< OperatorType >
{
 public:
   using FSFunctionType = typename OperatorType::srcType;

   explicit GSFaceSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel, P1Form& form )
   : storage_( std::move( storage ) )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , form_( form )
   {}

   virtual void smooth( const OperatorType&, uint_t level, Face& face, const FSFunctionType& x, const FSFunctionType& b )
   {
      smooth_sor_face3D( storage_, face, form_, x.getFaceDataID(), b.getFaceDataID(), level, 1., false );
   }

   virtual void
       smooth_backwards( const OperatorType&, uint_t level, Face& face, const FSFunctionType& x, const FSFunctionType& b )
   {
      smooth_sor_face3D( storage_, face, form_, x.getFaceDataID(), b.getFaceDataID(), level, 1., true );
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;
   DoFType                             flag_;
   size_t                              minLevel_;
   size_t                              maxLevel_;
   P1Form                              form_;
};

} // namespace hyteg