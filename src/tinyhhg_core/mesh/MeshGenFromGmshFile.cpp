
#include "tinyhhg_core/mesh/MeshInfo.hpp"

#include "core/logging/Logging.h"
#include "core/debug/CheckFunctions.h"
#include "core/debug/Debug.h"

#include <array>
#include <vector>

namespace hhg {

MeshInfo MeshInfo::fromGmshFile( const std::string & meshFileName )
{
  MeshInfo meshInfo;
  const std::array< DoFType, 3 > BOUNDARY_TYPE_TO_FLAG = {{ Inner, DirichletBoundary, NeumannBoundary }}; // double-braces to silence warning

  WALBERLA_LOG_INFO_ON_ROOT( "[Mesh] Opening mesh file: " << meshFileName );

  std::ifstream meshFile;
  meshFile.open( meshFileName.c_str() );

  WALBERLA_CHECK( !!meshFile, "[Mesh] Error opening file: " << meshFileName );

  std::string token;
  meshFile >> token; // $MeshFormat

  WALBERLA_CHECK_EQUAL( token, "$MeshFormat", "[Mesh] Missing: $MeshFormat" );

  meshFile >> token; // version

  WALBERLA_CHECK_EQUAL( token, "2.2", "[Mesh] Meshfile version should be 2.2");

  meshFile >> token; // 0
  meshFile >> token; // 8
  meshFile >> token; // $EndMeshFormat
  meshFile >> token; // $Nodes

  WALBERLA_CHECK_EQUAL( token, "$Nodes", "[Mesh] Missing: $Nodes");

  uint_t numVertices;
  meshFile >> numVertices;

  for ( uint_t vertex = 0; vertex < numVertices; ++vertex )
  {
    IDType id;
    real_t x[3];
    meshFile >> id;
    meshFile >> x[0];
    meshFile >> x[1];
    meshFile >> x[2];

    meshInfo.vertices_[id] = MeshInfo::Vertex( id, Point3D( x ), Inner, 0 );
  }

  WALBERLA_ASSERT_EQUAL( meshInfo.vertices_.size(), numVertices );

  meshFile >> token; // $EndNodes
  meshFile >> token; // $Elements

  WALBERLA_CHECK_EQUAL( token, "$Elements", "[Mesh] Missing: $Elements" )

  uint_t numPrimitives;
  meshFile >> numPrimitives;

  // Two pass approach:
  // First we parse the file and store the information plainly into vector.
  // This way, we can then first process the edges and then all faces.

  EdgeContainer parsedEdges;
  FaceContainer parsedFaces;
  CellContainer parsedCells;

  for ( uint_t primitive = 0; primitive < numPrimitives; ++primitive )
  {
    uint_t ig;
    meshFile >> ig; // ignore

    uint_t primitiveType;
    meshFile >> primitiveType;

    if (primitiveType == 1) // edge
    {
      uint_t boundaryFlag;
      std::array< IDType, 2 > edgeVertices;
      meshFile >> ig; // ignore
      meshFile >> boundaryFlag;
      meshFile >> ig; // ignore
      meshFile >> edgeVertices[0];
      meshFile >> edgeVertices[1];

      WALBERLA_ASSERT_LESS( boundaryFlag, BOUNDARY_TYPE_TO_FLAG.size() );
      DoFType dofType = BOUNDARY_TYPE_TO_FLAG[boundaryFlag];

      parsedEdges[ edgeVertices ] = Edge( edgeVertices, dofType, boundaryFlag );
    }

    else if (primitiveType == 2) // triangle
    {
      uint_t boundaryFlag;
      std::vector< IDType > triangleVertices( 3 );
      meshFile >> ig; // ignore
      meshFile >> boundaryFlag;
      meshFile >> ig; // ignore
      meshFile >> triangleVertices[0];
      meshFile >> triangleVertices[1];
      meshFile >> triangleVertices[2];

      WALBERLA_ASSERT_LESS( boundaryFlag, BOUNDARY_TYPE_TO_FLAG.size() );
      DoFType dofType = BOUNDARY_TYPE_TO_FLAG[boundaryFlag];

      parsedFaces[ triangleVertices ] = MeshInfo::Face( triangleVertices, dofType, boundaryFlag );
    }

    else if (primitiveType == 4) // tetrahedron
    {
      std::vector< IDType > tetrahedronVertices( 4 );
      meshFile >> ig; // ignore
      meshFile >> ig; // ignore
      meshFile >> ig; // ignore
      meshFile >> tetrahedronVertices[0];
      meshFile >> tetrahedronVertices[1];
      meshFile >> tetrahedronVertices[2];
      meshFile >> tetrahedronVertices[3];

      parsedCells[ tetrahedronVertices ] = Cell( tetrahedronVertices, Inner, 0 );
    }

    else if (primitiveType == 15) // vertex
    {
      // do nothing
      meshFile >> ig;
      meshFile >> ig;
      meshFile >> ig;
      meshFile >> ig;
    }

    else
    {
      WALBERLA_ABORT( "[Mesh] Unsupported element type: " << primitiveType );
    }
  }

  meshFile >> token; // $EndElements

  WALBERLA_ASSERT_EQUAL( token, "$EndElements", "[Mesh] Misparsed: cannot find $EndElements tag." );

  for ( const auto & it : parsedEdges )
  {
    const MeshInfo::Edge meshInfoEdge = it.second;
    meshInfo.addEdge( meshInfoEdge );
  }

  for ( const auto & it : parsedFaces )
  {
    const std::vector< IDType > faceCoordinates = it.first;
    const MeshInfo::Face        meshInfoFace    = it.second;

    // If the corresponding edge was not already added, add an edge of type Inner
    WALBERLA_ASSERT_EQUAL( faceCoordinates.size(), 3, "[Mesh] Only triangle faces supported." );

    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ faceCoordinates[0], faceCoordinates[1] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ faceCoordinates[1], faceCoordinates[2] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ faceCoordinates[2], faceCoordinates[0] }} ), Inner, 0 ) );

    meshInfo.addFace( meshInfoFace );
  }

  for ( const auto & it : parsedCells )
  {
    const std::vector< IDType > cellCoordinates = it.first;
    const MeshInfo::Cell        meshInfoCell    = it.second;

    // If the corresponding edge was not already added, add an edge of type Inner
    WALBERLA_ASSERT_EQUAL( cellCoordinates.size(), 4, "[Mesh] Only tetrahedron cells supported." );

    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ cellCoordinates[0], cellCoordinates[1] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ cellCoordinates[0], cellCoordinates[2] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ cellCoordinates[0], cellCoordinates[3] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ cellCoordinates[1], cellCoordinates[2] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ cellCoordinates[1], cellCoordinates[3] }} ), Inner, 0 ) );
    meshInfo.addEdge( Edge( std::array< IDType, 2 >( {{ cellCoordinates[2], cellCoordinates[3] }} ), Inner, 0 ) );

    meshInfo.addFace( Face( std::vector< IDType >( {{ cellCoordinates[0], cellCoordinates[1], cellCoordinates[2] }} ), Inner, 0 ) );
    meshInfo.addFace( Face( std::vector< IDType >( {{ cellCoordinates[0], cellCoordinates[1], cellCoordinates[3] }} ), Inner, 0 ) );
    meshInfo.addFace( Face( std::vector< IDType >( {{ cellCoordinates[0], cellCoordinates[2], cellCoordinates[3] }} ), Inner, 0 ) );
    meshInfo.addFace( Face( std::vector< IDType >( {{ cellCoordinates[1], cellCoordinates[2], cellCoordinates[3] }} ), Inner, 0 ) );

    meshInfo.cells_[ cellCoordinates ] = meshInfoCell;

  }

  meshFile.close();

  return meshInfo;
}

} // namespace hhg
