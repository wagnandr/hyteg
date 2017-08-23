
#include "core/debug/CheckFunctions.h"
#include "core/debug/Debug.h"
#include "core/logging/Logging.h"

#include "tinyhhg_core/primitivestorage/SetupPrimitiveStorage.hpp"

#include <algorithm>
#include <iomanip>
#include <set>

namespace hhg {

SetupPrimitiveStorage::SetupPrimitiveStorage( const MeshInfo & meshInfo, const uint_t & numberOfProcesses ) :
    numberOfProcesses_( numberOfProcesses )
{
  WALBERLA_ASSERT_GREATER( numberOfProcesses_, 0, "Number of processes must be positive" );

  // Adding vertices to storage
  MeshInfo::VertexContainer vertices = meshInfo.getVertices();
  for ( auto it = vertices.begin(); it != vertices.end(); it++ )
  {
    PrimitiveID vertexID( it->first );
    Point3D coordinates( it->second );
    vertices_[ vertexID.getID() ] = std::make_shared< Vertex >( vertexID, coordinates );

    // All to root by default
    primitiveIDToTargetRankMap_[ vertexID.getID() ] = 0;
  }

  // Adding edges to storage
  MeshInfo::EdgeContainer edges = meshInfo.getEdges();
  for ( auto it = edges.begin(); it != edges.end(); it++ )
  {
    PrimitiveID edgeID = generatePrimitiveID();
    PrimitiveID vertexID0 = PrimitiveID( it->first.first  );
    PrimitiveID vertexID1 = PrimitiveID( it->first.second );
    DoFType dofType = it->second;

    std::array<Point3D, 2> coords;

    coords[0] = vertices_[vertexID0.getID()]->getCoordinates();
    coords[1] = vertices_[vertexID1.getID()]->getCoordinates();

    WALBERLA_ASSERT_EQUAL( edges_.count( edgeID.getID() ), 0 );
    WALBERLA_ASSERT_EQUAL( vertices_.count( vertexID0.getID() ), 1 );
    WALBERLA_ASSERT_EQUAL( vertices_.count( vertexID1.getID() ), 1 );
    edges_[ edgeID.getID() ] = std::make_shared< Edge >( edgeID, vertexID0, vertexID1, dofType, coords);

    // All to root by default
    primitiveIDToTargetRankMap_[ edgeID.getID() ] = 0;

    // Adding edge ID as neighbor to SetupVertices
    vertices_[ vertexID0.getID() ]->addEdge( edgeID );
    vertices_[ vertexID1.getID() ]->addEdge( edgeID );
  }

  // Adding faces to storage
  MeshInfo::FaceContainer faces = meshInfo.getFaces();
  for ( auto it = faces.begin(); it != faces.end(); it++ )
  {
    PrimitiveID faceID = generatePrimitiveID();
    PrimitiveID vertexID0 = PrimitiveID( (*it)[0] );
    PrimitiveID vertexID1 = PrimitiveID( (*it)[1] );
    PrimitiveID vertexID2 = PrimitiveID( (*it)[2] );

    WALBERLA_ASSERT_EQUAL( faces_.count( faceID.getID() ), 0 );
    WALBERLA_ASSERT_EQUAL( vertices_.count( vertexID0.getID() ), 1 );
    WALBERLA_ASSERT_EQUAL( vertices_.count( vertexID1.getID() ), 1 );
    WALBERLA_ASSERT_EQUAL( vertices_.count( vertexID2.getID() ), 1 );

    PrimitiveID edgeID0;
    PrimitiveID edgeID1;
    PrimitiveID edgeID2;

    bool foundEdge0 = findEdgeByVertexIDs( vertexID0, vertexID1, edgeID0 );
    bool foundEdge1 = findEdgeByVertexIDs( vertexID1, vertexID2, edgeID1 );
    bool foundEdge2 = findEdgeByVertexIDs( vertexID2, vertexID0, edgeID2 );

    WALBERLA_CHECK( foundEdge0 && foundEdge1 && foundEdge2, "Could not successfully construct faces from MeshInfo" );

    WALBERLA_ASSERT_EQUAL( edges_.count( edgeID0.getID() ), 1 );
    WALBERLA_ASSERT_EQUAL( edges_.count( edgeID1.getID() ), 1 );
    WALBERLA_ASSERT_EQUAL( edges_.count( edgeID2.getID() ), 1 );

    // Edge Orientation
    std::array< int, 3 > edgeOrientation;

    PrimitiveID edge0Vertex0 = edges_[ edgeID0.getID() ]->getVertexID0();
    PrimitiveID edge0Vertex1 = edges_[ edgeID0.getID() ]->getVertexID1();
    PrimitiveID edge1Vertex0 = edges_[ edgeID1.getID() ]->getVertexID0();
    PrimitiveID edge1Vertex1 = edges_[ edgeID1.getID() ]->getVertexID1();
    PrimitiveID edge2Vertex0 = edges_[ edgeID2.getID() ]->getVertexID0();
    PrimitiveID edge2Vertex1 = edges_[ edgeID2.getID() ]->getVertexID1();

    if (edge0Vertex1 == edge1Vertex0 && edge1Vertex1 == edge2Vertex0 && edge2Vertex1 == edge0Vertex0)
    {
      edgeOrientation = {{1, 1, 1}};
    }
    else if (edge0Vertex1 == edge1Vertex0 && edge1Vertex1 == edge2Vertex1 && edge2Vertex0 == edge0Vertex0)
    {
      edgeOrientation = {{1, 1, -1}};
    }
    else if (edge0Vertex1 == edge1Vertex1 && edge1Vertex0 == edge2Vertex0 && edge2Vertex1 == edge0Vertex0)
    {
      edgeOrientation = {{1, -1, 1}};
    }
    else if (edge0Vertex1 == edge1Vertex1 && edge1Vertex0 == edge2Vertex1 && edge2Vertex0 == edge0Vertex0)
    {
      edgeOrientation = {{1, -1, -1}};
    }
    else if (edge0Vertex0 == edge1Vertex0 && edge1Vertex1 == edge2Vertex0 && edge2Vertex1 == edge0Vertex1)
    {
      edgeOrientation = {{-1, 1, 1}};
    }
    else if (edge0Vertex0 == edge1Vertex0 && edge1Vertex1 == edge2Vertex1 && edge2Vertex0 == edge0Vertex1)
    {
      edgeOrientation = {{-1, 1, -1}};
    }
    else if (edge0Vertex0 == edge1Vertex1 && edge1Vertex0 == edge2Vertex0 && edge2Vertex1 == edge0Vertex1)
    {
      edgeOrientation = {{-1, -1, 1}};
    }
    else if (edge0Vertex0 == edge1Vertex1 && edge1Vertex0 == edge2Vertex1 && edge2Vertex0 == edge0Vertex1)
    {
      edgeOrientation = {{-1, -1, -1}};
    }

    // Corner coordinates
    std::array< Point3D, 3 > coordinates;
    std::array< PrimitiveID, 3 > vertexIDs;

    if (edgeOrientation[0] == 1)
    {
      coordinates[0] = vertices_[ edge0Vertex0.getID() ]->getCoordinates();
      coordinates[1] = vertices_[ edge0Vertex1.getID() ]->getCoordinates();

      vertexIDs[0] = edge0Vertex0.getID();
      vertexIDs[1] = edge0Vertex1.getID();
    }
    else
    {
      coordinates[0] = vertices_[ edge0Vertex1.getID() ]->getCoordinates();
      coordinates[1] = vertices_[ edge0Vertex0.getID() ]->getCoordinates();

      vertexIDs[0] = edge0Vertex1.getID();
      vertexIDs[1] = edge0Vertex0.getID();
    }

    if (edgeOrientation[1] == 1)
    {
      coordinates[2] = vertices_[ edge1Vertex1.getID() ]->getCoordinates();

      vertexIDs[2] = edge1Vertex1.getID();
    }
    else
    {
      coordinates[2] = vertices_[ edge1Vertex0.getID() ]->getCoordinates();

      vertexIDs[2] = edge1Vertex0.getID();
    }

    faces_[ faceID.getID() ] = std::shared_ptr< Face >( new Face( faceID, vertexIDs, {{edgeID0, edgeID1, edgeID2}}, edgeOrientation, coordinates ) );

    // All to root by default
    primitiveIDToTargetRankMap_[ faceID.getID() ] = 0;

    // Adding face ID to vertices as neighbors
    vertices_[vertexIDs[0].getID()]->addFace(faceID);
    vertices_[vertexIDs[1].getID()]->addFace(faceID);
    vertices_[vertexIDs[2].getID()]->addFace(faceID);

    // Adding face ID to edges as neighbors
    edges_[ edgeID0.getID() ]->addFace( faceID );
    edges_[ edgeID1.getID() ]->addFace( faceID );
    edges_[ edgeID2.getID() ]->addFace( faceID );
  }

  for (auto& it : edges_) {
    Edge& edge = *it.second;

    if (testFlag(edge.type, hhg::NeumannBoundary)) {

      for (auto& itv : edge.neighborVertices()) {
        vertices_[itv.getID()]->dofType_ = hhg::NeumannBoundary;
      }

    }
  }

  for (auto& it : edges_) {
    Edge& edge = *it.second;

    if (testFlag(edge.type, hhg::DirichletBoundary)) {

      for (auto& itv : edge.neighborVertices()) {
        vertices_[itv.getID()]->dofType_ = hhg::DirichletBoundary;
      }

    }
  }
}

const Primitive * SetupPrimitiveStorage::getPrimitive( const PrimitiveID & id ) const
{
  if ( vertexExists( id ) ) { return getVertex( id ); }
  if ( edgeExists( id ) )   { return getEdge( id ); }
  if ( faceExists( id ) )   { return getFace( id ); }
  return nullptr;
}

bool SetupPrimitiveStorage::findEdgeByVertexIDs( const PrimitiveID & vertexID0, const PrimitiveID & vertexID1, PrimitiveID & edge ) const
{
  if ( vertices_.count( vertexID0.getID() ) == 0 || vertices_.count( vertexID1.getID() ) == 0 )
  {
    return false;
  }

  for ( auto it = edges_.begin(); it != edges_.end(); it++ )
  {
    if (   ( it->second->getVertexID0() == vertexID0 && it->second->getVertexID1() == vertexID1 )
        || ( it->second->getVertexID0() == vertexID1 && it->second->getVertexID1() == vertexID0 ) )
    {
      edge = PrimitiveID( it->first );
      return true;
    }
  }

  return false;
}


void SetupPrimitiveStorage::assembleRankToSetupPrimitivesMap( RankToSetupPrimitivesMap & rankToSetupPrimitivesMap ) const
{
  rankToSetupPrimitivesMap.clear();

  PrimitiveMap setupPrimitives;
  getSetupPrimitives( setupPrimitives );
  for ( uint_t rank = 0; rank < numberOfProcesses_; rank++ )
  {
    rankToSetupPrimitivesMap[ rank ] = std::vector< PrimitiveID::IDType >();
    for ( auto setupPrimitive : setupPrimitives )
    {
      if ( rank == getTargetRank( setupPrimitive.first ) )
      {
	      rankToSetupPrimitivesMap[ rank ].push_back( setupPrimitive.first );
      }
    }
  }

  WALBERLA_ASSERT_LESS_EQUAL( rankToSetupPrimitivesMap.size(), numberOfProcesses_ );
}

uint_t SetupPrimitiveStorage::getNumberOfEmptyProcesses() const
{
  uint_t numberOfEmptyProcesses = 0;
  RankToSetupPrimitivesMap rankToSetupPrimitivesMap;
  assembleRankToSetupPrimitivesMap( rankToSetupPrimitivesMap );
  for ( auto const & rankToSetupPrimitives : rankToSetupPrimitivesMap )
  {
    if ( rankToSetupPrimitives.second.size() == 0 )
    {
      numberOfEmptyProcesses++;
    }
  }
  return numberOfEmptyProcesses;
}

uint_t SetupPrimitiveStorage::getNumberOfPrimitives() const
{
  return vertices_.size() + edges_.size() + faces_.size();
}

uint_t SetupPrimitiveStorage::getMinPrimitivesPerRank() const
{
  uint_t minNumberOfPrimitives = std::numeric_limits< uint_t >::max();
  RankToSetupPrimitivesMap rankToSetupPrimitivesMap;
  assembleRankToSetupPrimitivesMap( rankToSetupPrimitivesMap );
  for ( auto const & rankToSetupPrimitives : rankToSetupPrimitivesMap )
  {
    minNumberOfPrimitives = std::min( rankToSetupPrimitives.second.size(), minNumberOfPrimitives );
  }
  return minNumberOfPrimitives;
}

uint_t SetupPrimitiveStorage::getMaxPrimitivesPerRank() const
{
  uint_t maxNumberOfPrimitives = 0;
  RankToSetupPrimitivesMap rankToSetupPrimitivesMap;
  assembleRankToSetupPrimitivesMap( rankToSetupPrimitivesMap );
  for ( auto const & rankToSetupPrimitives : rankToSetupPrimitivesMap )
  {
    maxNumberOfPrimitives = std::max( rankToSetupPrimitives.second.size(), maxNumberOfPrimitives );
  }
  return maxNumberOfPrimitives;
}

real_t SetupPrimitiveStorage::getAvgPrimitivesPerRank() const
{
  return real_t( getNumberOfPrimitives() ) / real_t( numberOfProcesses_ );
}


void SetupPrimitiveStorage::toStream( std::ostream & os ) const
{
  os << "SetupPrimitiveStorage:\n";

  os << " - Processes (overall): " << std::setw(10) << numberOfProcesses_ << "\n";
  os << " - Processes (empty)  : " << std::setw(10) << getNumberOfEmptyProcesses() << "\n";

  os << " - Number of...\n"
     << "   +  Vertices: " << std::setw(10) << vertices_.size() << "\n"
     << "   +     Edges: " << std::setw(10) << edges_.size() << "\n"
     << "   +     Faces: " << std::setw(10) << faces_.size() << "\n";

  os << " - Primitives per process...\n"
     << "   +      min: " << std::setw(10) << getMinPrimitivesPerRank() << "\n"
     << "   +      max: " << std::setw(10) << getMaxPrimitivesPerRank() << "\n"
     << "   +      avg: " << std::setw(10) << getAvgPrimitivesPerRank() << "\n";

#ifndef NDEBUG
  os << "\n";
  os << "Vertices:   ID | Target Rank | Position  | Neighbor Edges \n"
     << "---------------------------------------------------------\n";
  for ( auto it = vertices_.begin(); it != vertices_.end(); it++ )
  {
    Point3D coordinates = it->second->getCoordinates();
    os << "          " << std::setw(4) << it->first << " | "
       << std::setw(11) << getTargetRank( it->first ) << " | "
       << coordinates << " | ";
    for ( const auto & neighborEdgeID : it->second->getHigherDimNeighbors() )
    {
      os << neighborEdgeID.getID() << " ";
    }
    os << "\n";

  }
  os << "\n";

  os << "Edges:      ID | Target Rank | VertexID_0 | VertexID_1 | DoF Type             | Neighbor Faces \n"
     << "----------------------------------------------------------------------------------------------\n";
  for ( auto it = edges_.begin(); it != edges_.end(); it++ )
  {
    os << "          " << std::setw(4) << it->first << " | "
       << std::setw(11) << getTargetRank( it->first ) << " | "
       << std::setw(10) << it->second->getVertexID0().getID() << " | "
       << std::setw(10) << it->second->getVertexID1().getID() << " | "
       << std::setw(20) << it->second->getDoFType() << " | ";
    for ( const auto & neighborFaceID : it->second->getHigherDimNeighbors() )
    {
      os << neighborFaceID.getID() << " ";
    }
        os << "\n";
  }
  os << "\n";

  os << "Faces:      ID | Target Rank | EdgeID_0 | EdgeID_1 | EdgeID_2\n"
     << "-------------------------------------------------------------\n";
  for ( auto it = faces_.begin(); it != faces_.end(); it++ )
  {
    os << "          " << std::setw(4) << it->first << " | "
       << std::setw(11) << getTargetRank( it->first ) << " | "
       << std::setw(8) << it->second->getEdgeID0().getID() << " | "
       << std::setw(8) << it->second->getEdgeID1().getID() << " | "
       << std::setw(8) << it->second->getEdgeID2().getID() << "\n";
  }
#endif
}


void SetupPrimitiveStorage::getSetupPrimitives( PrimitiveMap & setupPrimitiveMap ) const
{
  setupPrimitiveMap.clear();

  setupPrimitiveMap.insert( beginVertices(), endVertices() );
  setupPrimitiveMap.insert( beginEdges(), endEdges() );
  setupPrimitiveMap.insert( beginFaces(), endFaces() );

  WALBERLA_ASSERT_EQUAL( setupPrimitiveMap.size(), vertices_.size() + edges_.size() + faces_.size() );
}


PrimitiveID SetupPrimitiveStorage::generatePrimitiveID() const
{
  uint_t maxIDVertices = vertices_.size() == 0 ? 0 : vertices_.rbegin()->first;
  uint_t maxIDEdges    = edges_.size() == 0 ? 0 : edges_.rbegin()->first;
  uint_t maxIDFaces    = faces_.size() == 0 ? 0 : faces_.rbegin()->first;

  return PrimitiveID( std::max( std::max( maxIDVertices, maxIDFaces ), maxIDEdges ) + 1 );
}

}
