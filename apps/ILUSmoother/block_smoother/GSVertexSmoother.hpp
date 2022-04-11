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

template < typename ValueType, typename P1Form >
inline void smooth_sor_vertex3D( const std::shared_ptr< PrimitiveStorage >&                    storage,
                                 Vertex&                                                       vertex,
                                 P1Form&                                                       form,
                                 const PrimitiveDataID< FunctionMemory< ValueType >, Vertex >& dstId,
                                 const PrimitiveDataID< FunctionMemory< ValueType >, Vertex >& rhsId,
                                 size_t                                                        level,
                                 ValueType                                                     relax )
{
   form.setGeometryMap( vertex.getGeometryMap() );
   auto stencil =
       P1Elements::P1Elements3D::assembleP1LocalStencil_new< P1Form >( storage, vertex, indexing::Index( 0, 0, 0 ), level, form );

   WALBERLA_ASSERT_EQUAL( stencil.size(), vertex.getNumNeighborEdges() + 1 );

   auto opr_data = &stencil.front();
   auto dst      = vertex.getData( dstId )->getPointer( level );
   auto rhs      = vertex.getData( rhsId )->getPointer( level );

   ValueType tmp;
   tmp = rhs[0];

   for ( size_t i = 0; i < vertex.getNumNeighborEdges(); ++i )
   {
      tmp -= opr_data[i + 1] * dst[i + 1];
   }

   dst[0] = ( 1.0 - relax ) * dst[0] + relax * tmp / opr_data[0];
}

template < typename OperatorType, typename P1Form >
class GSVertexSmoother : public VertexSmoother< OperatorType >
{
 public:
   using VSFunctionType = typename OperatorType::srcType;

   explicit GSVertexSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel, P1Form& form )
   : storage_( std::move( storage ) )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , form_( form )
   {}

   void smooth( const OperatorType&, uint_t level, Vertex& v, const VSFunctionType& x, const VSFunctionType& b ) override
   {
      smooth_sor_vertex3D( storage_, v, form_, x.getVertexDataID(), b.getVertexDataID(), level, 1. );
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;
   DoFType                             flag_;
   size_t                              minLevel_;
   size_t                              maxLevel_;
   P1Form                              form_;
};

} // namespace hyteg