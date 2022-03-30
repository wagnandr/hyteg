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
void assemble_variableStencil_cell( const std::shared_ptr< PrimitiveStorage > storage,
                                    P1Form&                                   form,
                                    Cell&                                     cell,
                                    uint_t                                    level,
                                    vertexdof::macrocell::StencilMap_T&       cell_stencil,
                                    const uint_t                              i,
                                    const uint_t                              j,
                                    const uint_t                              k )
{
   cell_stencil = P1Elements::P1Elements3D::assembleP1LocalStencilNew_new< P1Form >(
       storage, cell, indexing::Index( i, j, k ), level, form );
}

template < typename P1Form >
void smooth_sor_cell( const std::shared_ptr< PrimitiveStorage >                storage,
                      Cell&                                                    cell,
                      P1Form&                                                  form,
                      const PrimitiveDataID< FunctionMemory< real_t >, Cell >& dstId,
                      const PrimitiveDataID< FunctionMemory< real_t >, Cell >& rhsId,
                      const uint_t&                                            level,
                      real_t                                                   relax,
                      const bool&                                              backwards = false )
{
   WALBERLA_UNUSED( backwards );

   typedef stencilDirection sd;

   vertexdof::macrocell::StencilMap_T operatorData;
   const real_t*                      rhs = cell.getData( rhsId )->getPointer( level );
   real_t*                            dst = cell.getData( dstId )->getPointer( level );

   real_t tmp;

   auto inverseCenterWeight = 1.0 / operatorData[{ 0, 0, 0 }];

   const uint_t rowsizeZ = levelinfo::num_microvertices_per_edge( level );
   uint_t       rowsizeY, rowsizeX;

   // skip level 0 (no interior points)
   if ( level == 0 )
      return;

   for ( uint_t k = 1; k < rowsizeZ - 3; ++k )
   {
      rowsizeY = rowsizeZ - k;

      for ( uint_t j = 1; j < rowsizeY - 2; ++j )
      {
         rowsizeX = rowsizeY - j;

         for ( uint_t i = 1; i < rowsizeX - 1; ++i )
         {
            assemble_variableStencil_cell( storage, form, cell, level, operatorData, i, j, k );
            inverseCenterWeight = 1.0 / operatorData[{ 0, 0, 0 }];

            const uint_t centerIdx = vertexdof::macrocell::indexFromVertex( level, i, j, k, sd::VERTEX_C );

            tmp = rhs[centerIdx];

            for ( const auto& neighbor : vertexdof::macrocell::neighborsWithoutCenter )
            {
               const uint_t idx = vertexdof::macrocell::indexFromVertex( level, i, j, k, neighbor );
               tmp -= operatorData[vertexdof::logicalIndexOffsetFromVertex( neighbor )] * dst[idx];
            }

            dst[centerIdx] = ( 1.0 - relax ) * dst[centerIdx] + tmp * relax * inverseCenterWeight;
         }
      }
   }
}

template < typename OperatorType, typename P1Form >
class GSCellSmoother : public CellSmoother< OperatorType >
{
 public:
   using FSFunctionType = typename OperatorType::srcType;

   explicit GSCellSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel, P1Form& form )
   : storage_( std::move( storage ) )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , form_( form )
   {}

   void smooth( const OperatorType&, uint_t level, Cell& cell, const FSFunctionType& x, const FSFunctionType& b ) override
   {
      smooth_sor_cell( storage_, cell, form_, x.getCellDataID(), b.getCellDataID(), level, 1., false );
   }

   void smooth_backwards( const OperatorType&, uint_t, Cell&, const FSFunctionType&, const FSFunctionType& ) override
   {
      WALBERLA_ABORT( "not implemented!" );
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;
   DoFType                             flag_;
   size_t                              minLevel_;
   size_t                              maxLevel_;
   P1Form                              form_;
};

} // namespace hyteg