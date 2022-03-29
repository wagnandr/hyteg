/*
 * Copyright (c) 2020 Andreas Wagner.
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

#include "hyteg/primitivestorage/StoragePermutator.hpp"

#include <core/all.h>
#include <random>

#include "hyteg/PrimitiveID.hpp"
#include "hyteg/primitives/Edge.hpp"
#include "hyteg/primitives/Face.hpp"
#include "hyteg/primitives/Primitive.hpp"
#include "hyteg/primitives/Vertex.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"

namespace hyteg {

void StoragePermutator::permutate_randomly( SetupPrimitiveStorage& storage, uint_t seed )
{
   if ( storage.getNumberOfCells() > 0 )
      WALBERLA_ABORT( "permutating primitives does not work in 3D yet" );

   // random engine
   std::default_random_engine generator( seed );

   // 2d permutation
   if (storage.getNumberOfCells() == 0)
   {
      WALBERLA_ABORT("not implemented for faces");
   } else {
      // test, we shift the coefficients
      std::array< uint_t, 4 > permutation_map{ 0, 1, 2, 3 };

      for ( const auto& cit : storage.getCells() )
      {
         Cell& cell = *cit.second;
         std::shuffle( permutation_map.begin(), permutation_map.end(), generator );
         permutate(storage, cell, permutation_map);
      };
   }
}

void StoragePermutator::permutate_ilu( SetupPrimitiveStorage& storage )
{

   for ( const auto& cit : storage.getCells() )
   {
      Cell& cell = *cit.second;

      // find largest area facet
      real_t maxArea = 0;
      uint_t maxAreaIndex = 0;
      std::vector< uint_t > maxLocalVertexIds;
      for (uint_t i=0; i<4; i+= 1)
      {
         std::vector< uint_t > localVertexIds;
         for (uint_t j=0; j<4; j+= 1)
            if (j != i)
               localVertexIds.push_back(j);

         auto direction1 = cell.coordinates_[localVertexIds[1]] - cell.coordinates_[localVertexIds[0]];
         auto direction2 = cell.coordinates_[localVertexIds[2]] - cell.coordinates_[localVertexIds[0]];

         const real_t area = 0.5 * crossProduct( direction1, direction2).norm();

         if (area >= maxArea)
         {
            maxArea = area;
            maxAreaIndex = i;
            maxLocalVertexIds = localVertexIds;
         }
      }

      // permutate facet such that largest to smallest angle
      auto d01 = cell.coordinates_[maxLocalVertexIds[1]] - cell.coordinates_[maxLocalVertexIds[0]];
      d01 /= d01.norm();
      auto d02 = cell.coordinates_[maxLocalVertexIds[2]] - cell.coordinates_[maxLocalVertexIds[0]];
      d02 /= d02.norm();
      auto d21 = cell.coordinates_[maxLocalVertexIds[1]] - cell.coordinates_[maxLocalVertexIds[2]];
      d21 /= d21.norm();

      const std::array< real_t, 3 > angles = {
          std::acos( d01.dot( d02 ) ), // vertex 0
          std::acos( d01.dot( d21 ) ), // vertex 1
          std::acos( -d21.dot( d02 ) ) // vertex 2
      };

      const auto maxElementIdx   = static_cast< uint_t > ( std::max_element( angles.begin(), angles.end() ) - angles.begin() );
      const auto minElementIdx   = static_cast< uint_t > ( std::min_element( angles.begin(), angles.end() ) - angles.begin() );
      const uint_t otherElementIdx = 3 - maxElementIdx - minElementIdx;

      std::array< uint_t, 4 > permutation = {
          maxLocalVertexIds[maxElementIdx],
          maxLocalVertexIds[minElementIdx],
          maxLocalVertexIds[otherElementIdx],
          maxAreaIndex
      };

      permutate(storage, cell, permutation);
   }
}

void StoragePermutator::permutate( SetupPrimitiveStorage& setup, Cell& cell, std::array< uint_t, 4 > permutation )
{
   PrimitiveID vertexID0 = cell.neighborVertices_[permutation[0]];
   PrimitiveID vertexID1 = cell.neighborVertices_[permutation[1]];
   PrimitiveID vertexID2 = cell.neighborVertices_[permutation[2]];
   PrimitiveID vertexID3 = cell.neighborVertices_[permutation[3]];

   // TODO: move into storage
   auto findEdgePrimitiveID = [&](auto vID0, auto vID1){
      Vertex& v = *setup.getVertex(vID0);
      const auto& edgeIds = v.neighborEdges();
      for (auto edgeId: edgeIds)
      {
         Edge& edge = *setup.getEdge( edgeId );
         const bool edgeAligned = edge.getVertexID0() == vID0 && edge.getVertexID1() == vID1;
         const bool edgeUnaligned = edge.getVertexID1() == vID0 && edge.getVertexID0() == vID1;
         if (edgeAligned || edgeUnaligned)
            return edgeId;
      }
      WALBERLA_ABORT("edge primitive could not be found");
   };

   auto findFacePrimitiveID = [&](auto vID0, auto vID1, auto vID2)
   {
      Edge& e = *setup.getEdge(findEdgePrimitiveID(vID0, vID1));
      const auto& faceIds = e.neighborFaces();
     for (auto faceId: faceIds)
     {
        Face& face = *setup.getFace( faceId );
        if(face.getVertexID0() == vID0 && face.getVertexID1() == vID1 && face.getVertexID2() == vID2)
           return faceId;
        if(face.getVertexID0() == vID0 && face.getVertexID1() == vID2 && face.getVertexID2() == vID1)
           return faceId;
        if(face.getVertexID0() == vID1 && face.getVertexID1() == vID0 && face.getVertexID2() == vID2)
           return faceId;
        if(face.getVertexID0() == vID1 && face.getVertexID1() == vID2 && face.getVertexID2() == vID0)
           return faceId;
        if(face.getVertexID0() == vID2 && face.getVertexID1() == vID0 && face.getVertexID2() == vID1)
           return faceId;
        if(face.getVertexID0() == vID2 && face.getVertexID1() == vID1 && face.getVertexID2() == vID0)
           return faceId;
     }
     WALBERLA_ABORT("edge primitive could not be found");
   };

   PrimitiveID edgeID0 = findEdgePrimitiveID( vertexID0, vertexID1 );
   PrimitiveID edgeID1 = findEdgePrimitiveID( vertexID0, vertexID2 );
   PrimitiveID edgeID2 = findEdgePrimitiveID( vertexID1, vertexID2 );
   PrimitiveID edgeID3 = findEdgePrimitiveID( vertexID0, vertexID3 );
   PrimitiveID edgeID4 = findEdgePrimitiveID( vertexID1, vertexID3 );
   PrimitiveID edgeID5 = findEdgePrimitiveID( vertexID2, vertexID3 );

   PrimitiveID faceID0 = findFacePrimitiveID( vertexID0, vertexID1, vertexID2 );
   PrimitiveID faceID1 = findFacePrimitiveID( vertexID0, vertexID1, vertexID3 );
   PrimitiveID faceID2 = findFacePrimitiveID( vertexID0, vertexID2, vertexID3 );
   PrimitiveID faceID3 = findFacePrimitiveID( vertexID1, vertexID2, vertexID3 );

   std::vector< PrimitiveID > cellVertices = {{vertexID0, vertexID1, vertexID2, vertexID3}};
   std::vector< PrimitiveID > cellEdges    = {{edgeID0, edgeID1, edgeID2, edgeID3, edgeID4, edgeID5}};
   std::vector< PrimitiveID > cellFaces    = {{faceID0, faceID1, faceID2, faceID3}};

   std::array< Point3D, 4 > cellCoordinates = {{setup.getVertex(vertexID0)->getCoordinates(),
                                                setup.getVertex(vertexID1)->getCoordinates(),
                                                setup.getVertex(vertexID2)->getCoordinates(),
                                                setup.getVertex(vertexID3)->getCoordinates()}};

   std::array< std::map< uint_t, uint_t >, 6 > edgeLocalVertexToCellLocalVertexMaps;

   // edgeLocalVertexToCellLocalVertexMaps[ cellLocalEdgeID ][ edgeLocalVertexID ] = cellLocalVertexID;

   edgeLocalVertexToCellLocalVertexMaps[0][setup.getEdge( edgeID0 )->vertex_index( vertexID0 )] = 0;
   edgeLocalVertexToCellLocalVertexMaps[0][setup.getEdge( edgeID0 )->vertex_index( vertexID1 )] = 1;

   edgeLocalVertexToCellLocalVertexMaps[1][setup.getEdge( edgeID1 )->vertex_index( vertexID0 )] = 0;
   edgeLocalVertexToCellLocalVertexMaps[1][setup.getEdge( edgeID1 )->vertex_index( vertexID2 )] = 2;

   edgeLocalVertexToCellLocalVertexMaps[2][setup.getEdge( edgeID2 )->vertex_index( vertexID1 )] = 1;
   edgeLocalVertexToCellLocalVertexMaps[2][setup.getEdge( edgeID2 )->vertex_index( vertexID2 )] = 2;

   edgeLocalVertexToCellLocalVertexMaps[3][setup.getEdge( edgeID3 )->vertex_index( vertexID0 )] = 0;
   edgeLocalVertexToCellLocalVertexMaps[3][setup.getEdge( edgeID3 )->vertex_index( vertexID3 )] = 3;

   edgeLocalVertexToCellLocalVertexMaps[4][setup.getEdge( edgeID4 )->vertex_index( vertexID1 )] = 1;
   edgeLocalVertexToCellLocalVertexMaps[4][setup.getEdge( edgeID4 )->vertex_index( vertexID3 )] = 3;

   edgeLocalVertexToCellLocalVertexMaps[5][setup.getEdge( edgeID5 )->vertex_index( vertexID2 )] = 2;
   edgeLocalVertexToCellLocalVertexMaps[5][setup.getEdge( edgeID5 )->vertex_index( vertexID3 )] = 3;

   std::array< std::map< uint_t, uint_t >, 4 > faceLocalVertexToCellLocalVertexMaps;

   // faceLocalVertexToCellLocalVertexMaps[ cellLocalFaceID ][ faceLocalVertexID ] = cellLocalVertexID;

   faceLocalVertexToCellLocalVertexMaps[0][setup.getFace( faceID0 )->vertex_index( vertexID0 )] = 0;
   faceLocalVertexToCellLocalVertexMaps[0][setup.getFace( faceID0 )->vertex_index( vertexID1 )] = 1;
   faceLocalVertexToCellLocalVertexMaps[0][setup.getFace( faceID0 )->vertex_index( vertexID2 )] = 2;

   faceLocalVertexToCellLocalVertexMaps[1][setup.getFace( faceID1 )->vertex_index( vertexID0 )] = 0;
   faceLocalVertexToCellLocalVertexMaps[1][setup.getFace( faceID1 )->vertex_index( vertexID1 )] = 1;
   faceLocalVertexToCellLocalVertexMaps[1][setup.getFace( faceID1 )->vertex_index( vertexID3 )] = 3;

   faceLocalVertexToCellLocalVertexMaps[2][setup.getFace( faceID2 )->vertex_index( vertexID0 )] = 0;
   faceLocalVertexToCellLocalVertexMaps[2][setup.getFace( faceID2 )->vertex_index( vertexID2 )] = 2;
   faceLocalVertexToCellLocalVertexMaps[2][setup.getFace( faceID2 )->vertex_index( vertexID3 )] = 3;

   faceLocalVertexToCellLocalVertexMaps[3][setup.getFace( faceID3 )->vertex_index( vertexID1 )] = 1;
   faceLocalVertexToCellLocalVertexMaps[3][setup.getFace( faceID3 )->vertex_index( vertexID2 )] = 2;
   faceLocalVertexToCellLocalVertexMaps[3][setup.getFace( faceID3 )->vertex_index( vertexID3 )] = 3;

   cell.coordinates_ = cellCoordinates;
   cell.neighborVertices_ = cellVertices;
   cell.neighborEdges_ = cellEdges;
   cell.neighborFaces_ = cellFaces;
   cell.edgeLocalVertexToCellLocalVertexMaps_ = edgeLocalVertexToCellLocalVertexMaps;
   cell.faceLocalVertexToCellLocalVertexMaps_ = faceLocalVertexToCellLocalVertexMaps;
   cell.calculateInwardNormals();
}

} // namespace hyteg