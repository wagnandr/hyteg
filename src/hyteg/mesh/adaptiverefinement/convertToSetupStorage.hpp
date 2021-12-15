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
#include "mesh.hpp"

namespace hyteg {
namespace adaptiveRefinement {

// create setupstorage and add primitiveIDs to all elements
template < class K_Simplex >
inline SetupPrimitiveStorage
    K_Mesh< K_Simplex >::CreateSetupStorage( const std::vector< Point3D >&                       vertices,
                                             std::map< uint_t, std::shared_ptr< GeometryMap > >& vertexGeometryMap,
                                             std::set< std::shared_ptr< Simplex1 > >&            edges,
                                             std::set< std::shared_ptr< Simplex2 > >&            faces,
                                             std::set< std::shared_ptr< Simplex3 > >&            cells,
                                             const uint_t&                                       n_processes )
{
   SetupPrimitiveStorage::VertexMap vertices_sps;
   SetupPrimitiveStorage::EdgeMap   edges_sps;
   SetupPrimitiveStorage::FaceMap   faces_sps;
   SetupPrimitiveStorage::CellMap   cells_sps;

   // We cache the inserted primitives (edges, faces and cells) by filling
   // these maps with the surrounding vertexIDs as keys and the inserted
   // PrimitiveIDs as values.
   // This way we do not need to search for the neighboring lower level
   // primitives when building inner primitives.
   // std::map< std::vector< PrimitiveID >, PrimitiveID > vertexIDsToEdgeIDs;
   // std::map< std::vector< PrimitiveID >, PrimitiveID > vertexIDsToFaceIDs;
   // auto findCachedPrimitiveID = []( const std::vector< PrimitiveID >&                          unsortedPrimitiveIDs,
   //                                  const std::map< std::vector< PrimitiveID >, PrimitiveID >& cache ) -> PrimitiveID {
   //    std::vector< PrimitiveID > sortedKey( unsortedPrimitiveIDs );
   //    std::sort( sortedKey.begin(), sortedKey.end() );
   //    WALBERLA_ASSERT_GREATER(
   //        cache.count( sortedKey ), 0, "Could not find primitive in cache during SetupStorage construction." );
   //    return cache.at( sortedKey );
   // };
   std::map< Idx< 2 >, PrimitiveID > vertexIDsToEdgeID;
   std::map< Idx< 3 >, PrimitiveID > vertexIDsToFaceID;

   // give each primitive a running id
   uint_t id = 0;

   // Adding vertices to storage
   for ( const auto& vtx : vertices )
   {
      PrimitiveID vtxID( id );

      // add new vertex
      auto primitive   = std::make_shared< Vertex >( vtxID, vtx );
      vertices_sps[id] = primitive;

      // add properties
      //   primitive->meshBoundaryFlag_ = vtx.getBoundaryFlag() // todo
      primitive->geometryMap_ = vertexGeometryMap[id];

      ++id;
   }

   // Adding edges to storage
   for ( const auto& edge : edges )
   {
      constexpr uint_t K = 1;

      PrimitiveID edgeID( id );
      edge->setPrimitiveID( edgeID );

      const auto& v = edge->get_vertices();
      // vertex IDs
      std::array< PrimitiveID, K + 1 > vertexIDs;
      for ( uint_t i = 0; i < K + 1; ++i )
      {
         vertexIDs[i] = PrimitiveID( v[i] );
      }
      // vertex coordinates
      auto coords = edge->coordinates( vertices );

      // add new edge
      auto primitive = std::make_shared< Edge >( edgeID, vertexIDs[0], vertexIDs[1], coords );
      edges_sps[id]  = primitive;

      // add properties
      //   primitive->meshBoundaryFlag_ = edge->getBoundaryFlag() // todo
      primitive->geometryMap_ = edge->getGeometryMap();

      // Adding edge ID as neighbor to SetupVertices
      for ( const auto& vertexID : v )
      {
         vertices_sps[vertexID]->addEdge( edgeID );
      }

      // Caching neighboring vertices
      vertexIDsToEdgeID[v] = edgeID;

      ++id;
   }

   // Adding faces to storage
   for ( const auto& face : faces )
   {
      constexpr uint_t K = 2;

      PrimitiveID faceID( id );
      face->setPrimitiveID( faceID );

      const auto& v = face->get_vertices();
      // vertex ids
      std::array< PrimitiveID, K + 1 > vertexIDs;
      for ( uint_t i = 0; i < K + 1; ++i )
      {
         vertexIDs[i] = PrimitiveID( v[i] );
      }
      // vertex coordinates
      auto coords = face->coordinates( vertices );

      // edge ids
      std::array< PrimitiveID, K + 1 > edgeIDs;
      // edge orientation
      std::array< int, K + 1 > edgeOrientation;
      auto                     faceEdges = face->get_edges_sorted( v );
      for ( uint_t i = 0; i < K + 1; ++i )
      {
         edgeIDs[i] = faceEdges[i]->getPrimitiveID();
         if ( faceEdges[i]->get_vertices()[0] == v[i] )
            edgeOrientation[i] = 1;
         else
            edgeOrientation[i] = -1;
      }

      // add new face
      auto primitive = std::make_shared< Face >( faceID, vertexIDs, edgeIDs, edgeOrientation, coords );
      faces_sps[id]  = primitive;

      // add properties
      //   primitive->meshBoundaryFlag_ = face->getBoundaryFlag(); // todo
      primitive->geometryMap_ = face->getGeometryMap();

      // Adding face ID to vertices as neighbors
      for ( const auto& vertexID : v )
      {
         vertices_sps[vertexID]->addFace( faceID );
      }
      // Adding face ID to edges as neighbors
      for ( const auto& edgeID : edgeIDs )
      {
         edges_sps[edgeID.getID()]->addFace( faceID );
      }

      // Caching neighboring vertices
      vertexIDsToFaceID[v] = faceID;

      ++id;
   }

   // Adding cells to storage
   for ( const auto& cell : cells )
   {
      constexpr uint_t K = 3;

      PrimitiveID cellID( id );
      cell->setPrimitiveID( cellID );

      const auto& v = cell->get_vertices();
      // vertex ids
      std::vector< PrimitiveID > vertexIDs( K + 1 );
      for ( uint_t i = 0; i < K + 1; ++i )
      {
         vertexIDs[i] = PrimitiveID( v[i] );
      }
      // vertex coordinates
      auto coords = cell->coordinates( vertices );

      // edge ids
      std::vector< PrimitiveID > edgeIDs( 6 );
      edgeIDs[0] = vertexIDsToEdgeID[{ v[0], v[1] }];
      edgeIDs[1] = vertexIDsToEdgeID[{ v[0], v[2] }];
      edgeIDs[2] = vertexIDsToEdgeID[{ v[1], v[2] }];
      edgeIDs[3] = vertexIDsToEdgeID[{ v[0], v[3] }];
      edgeIDs[4] = vertexIDsToEdgeID[{ v[1], v[3] }];
      edgeIDs[5] = vertexIDsToEdgeID[{ v[2], v[3] }];

      // face IDs
      std::vector< PrimitiveID > faceIDs( K + 1 );
      faceIDs[0] = vertexIDsToFaceID[{ v[0], v[1], v[2] }];
      faceIDs[1] = vertexIDsToFaceID[{ v[0], v[1], v[3] }];
      faceIDs[2] = vertexIDsToFaceID[{ v[0], v[2], v[3] }];
      faceIDs[3] = vertexIDsToFaceID[{ v[1], v[1], v[3] }];

      std::array< std::map< uint_t, uint_t >, 6 > edgeLocalVertexToCellLocalVertexMaps;

      // edgeLocalVertexToCellLocalVertexMaps[ cellLocalEdgeID ][ edgeLocalVertexID ] = cellLocalVertexID;

      edgeLocalVertexToCellLocalVertexMaps[0][edges_sps.at( edgeIDs[0].getID() )->vertex_index( vertexIDs[0] )] = 0;
      edgeLocalVertexToCellLocalVertexMaps[0][edges_sps.at( edgeIDs[0].getID() )->vertex_index( vertexIDs[1] )] = 1;

      edgeLocalVertexToCellLocalVertexMaps[1][edges_sps.at( edgeIDs[1].getID() )->vertex_index( vertexIDs[0] )] = 0;
      edgeLocalVertexToCellLocalVertexMaps[1][edges_sps.at( edgeIDs[1].getID() )->vertex_index( vertexIDs[2] )] = 2;

      edgeLocalVertexToCellLocalVertexMaps[2][edges_sps.at( edgeIDs[2].getID() )->vertex_index( vertexIDs[1] )] = 1;
      edgeLocalVertexToCellLocalVertexMaps[2][edges_sps.at( edgeIDs[2].getID() )->vertex_index( vertexIDs[2] )] = 2;

      edgeLocalVertexToCellLocalVertexMaps[3][edges_sps.at( edgeIDs[3].getID() )->vertex_index( vertexIDs[0] )] = 0;
      edgeLocalVertexToCellLocalVertexMaps[3][edges_sps.at( edgeIDs[3].getID() )->vertex_index( vertexIDs[3] )] = 3;

      edgeLocalVertexToCellLocalVertexMaps[4][edges_sps.at( edgeIDs[4].getID() )->vertex_index( vertexIDs[1] )] = 1;
      edgeLocalVertexToCellLocalVertexMaps[4][edges_sps.at( edgeIDs[4].getID() )->vertex_index( vertexIDs[3] )] = 3;

      edgeLocalVertexToCellLocalVertexMaps[5][edges_sps.at( edgeIDs[5].getID() )->vertex_index( vertexIDs[2] )] = 2;
      edgeLocalVertexToCellLocalVertexMaps[5][edges_sps.at( edgeIDs[5].getID() )->vertex_index( vertexIDs[3] )] = 3;

      std::array< std::map< uint_t, uint_t >, 4 > faceLocalVertexToCellLocalVertexMaps;

      // faceLocalVertexToCellLocalVertexMaps[ cellLocalFaceID ][ faceLocalVertexID ] = cellLocalVertexID;

      faceLocalVertexToCellLocalVertexMaps[0][faces_sps.at( faceIDs[0].getID() )->vertex_index( vertexIDs[0] )] = 0;
      faceLocalVertexToCellLocalVertexMaps[0][faces_sps.at( faceIDs[0].getID() )->vertex_index( vertexIDs[1] )] = 1;
      faceLocalVertexToCellLocalVertexMaps[0][faces_sps.at( faceIDs[0].getID() )->vertex_index( vertexIDs[2] )] = 2;

      faceLocalVertexToCellLocalVertexMaps[1][faces_sps.at( faceIDs[1].getID() )->vertex_index( vertexIDs[0] )] = 0;
      faceLocalVertexToCellLocalVertexMaps[1][faces_sps.at( faceIDs[1].getID() )->vertex_index( vertexIDs[1] )] = 1;
      faceLocalVertexToCellLocalVertexMaps[1][faces_sps.at( faceIDs[1].getID() )->vertex_index( vertexIDs[3] )] = 3;

      faceLocalVertexToCellLocalVertexMaps[2][faces_sps.at( faceIDs[2].getID() )->vertex_index( vertexIDs[0] )] = 0;
      faceLocalVertexToCellLocalVertexMaps[2][faces_sps.at( faceIDs[2].getID() )->vertex_index( vertexIDs[2] )] = 2;
      faceLocalVertexToCellLocalVertexMaps[2][faces_sps.at( faceIDs[2].getID() )->vertex_index( vertexIDs[3] )] = 3;

      faceLocalVertexToCellLocalVertexMaps[3][faces_sps.at( faceIDs[3].getID() )->vertex_index( vertexIDs[1] )] = 1;
      faceLocalVertexToCellLocalVertexMaps[3][faces_sps.at( faceIDs[3].getID() )->vertex_index( vertexIDs[2] )] = 2;
      faceLocalVertexToCellLocalVertexMaps[3][faces_sps.at( faceIDs[3].getID() )->vertex_index( vertexIDs[3] )] = 3;

      // add new cell
      auto primitive = std::make_shared< Cell >( cellID,
                                                 vertexIDs,
                                                 edgeIDs,
                                                 faceIDs,
                                                 coords,
                                                 edgeLocalVertexToCellLocalVertexMaps,
                                                 faceLocalVertexToCellLocalVertexMaps );

      cells_sps[id] = primitive;

      // add properties
      //   primitive->meshBoundaryFlag_ = cell->getBoundaryFlag(); // todo
      primitive->geometryMap_ = cell->getGeometryMap();

      // Adding cell ID to vertices as neighbors
      for ( const auto& vertexID : v )
      {
         vertices_sps[vertexID]->addCell( cellID );
      }
      // Adding cell ID to edges as neighbors
      for ( const auto& edgeID : edgeIDs )
      {
         edges_sps[edgeID.getID()]->addCell( cellID );
      }
      // Adding cell ID to faces as neighbors
      for ( const auto& faceID : edgeIDs )
      {
         faces_sps[faceID.getID()]->addCell( cellID );
      }

      ++id;
   }

   // add indirect neighbor faces
   for ( const auto& [faceID, face] : faces_sps )
   {
      std::set< PrimitiveID > indirectNeighborsSet;

      for ( const auto& vertexID : face->neighborVertices() )
      {
         auto vertex = vertices_sps[vertexID.getID()];
         for ( const auto& neighborFaceID : vertex->neighborFaces() )
         {
            if ( neighborFaceID != faceID )
            {
               indirectNeighborsSet.insert( neighborFaceID );
            }
         }
      }

      face->indirectNeighborFaceIDs_.clear();
      face->indirectNeighborFaceIDs_.insert(
          face->indirectNeighborFaceIDs_.begin(), indirectNeighborsSet.begin(), indirectNeighborsSet.end() );
   }

   // add indirect neighbor cells
   for ( const auto& [cellID, cell] : cells_sps )
   {
      std::set< PrimitiveID > indirectNeighborsSet;

      for ( const auto& vertexID : cell->neighborVertices() )
      {
         auto vertex = vertices_sps[vertexID.getID()];
         for ( const auto& neighborCellID : vertex->neighborCells() )
         {
            if ( neighborCellID != cellID )
            {
               indirectNeighborsSet.insert( neighborCellID );
            }
         }
      }

      cell->indirectNeighborCellIDs_.clear();
      cell->indirectNeighborCellIDs_.insert(
          cell->indirectNeighborCellIDs_.begin(), indirectNeighborsSet.begin(), indirectNeighborsSet.end() );
   }

   return SetupPrimitiveStorage( vertices_sps, edges_sps, faces_sps, cells_sps, n_processes );
}

} // namespace adaptiveRefinement
} // namespace hyteg