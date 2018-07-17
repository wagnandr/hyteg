#pragma once

#include "tinyhhg_core/communication/DoFSpacePackInfo.hpp"
#include "tinyhhg_core/primitives/all.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroVertex.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroEdge.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroFace.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFIndexing.hpp"

namespace hhg {

template< typename ValueType >
class VertexDoFAdditivePackInfo : public communication::DoFSpacePackInfo<ValueType>
{
public:

  VertexDoFAdditivePackInfo( uint_t level,
                             PrimitiveDataID< FunctionMemory< ValueType >, Vertex > dataIDVertex,
                             PrimitiveDataID< FunctionMemory< ValueType >, Edge >   dataIDEdge,
                             PrimitiveDataID< FunctionMemory< ValueType >, Face >   dataIDFace,
                             PrimitiveDataID< FunctionMemory< ValueType >, Cell >   dataIDCell,
                             std::weak_ptr< PrimitiveStorage >                      storage,
                             BoundaryCondition                                      boundaryCondition,
                             DoFType                                                boundaryTypeToSkip ) :
      communication::DoFSpacePackInfo< ValueType >( level, dataIDVertex, dataIDEdge, dataIDFace, dataIDCell, storage ),
      boundaryCondition_( boundaryCondition ), boundaryTypeToSkip_( boundaryTypeToSkip )
  {}

  void packVertexForEdge(const Vertex *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackEdgeFromVertex(Edge *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalVertexToEdge(const Vertex *sender, Edge *receiver) const override;

  void packEdgeForVertex(const Edge *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackVertexFromEdge(Vertex *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalEdgeToVertex(const Edge *sender, Vertex *receiver) const override;

  void packEdgeForFace(const Edge *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackFaceFromEdge(Face *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalEdgeToFace(const Edge *sender, Face *receiver) const override;

  void packFaceForEdge(const Face *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackEdgeFromFace(Edge *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalFaceToEdge(const Face *sender, Edge *receiver) const override;

  void packFaceForCell(const Face *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackCellFromFace(Cell *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalFaceToCell(const Face *sender, Cell *receiver) const override;

  void packCellForFace(const Cell *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackFaceFromCell(Face *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalCellToFace(const Cell *sender, Face *receiver) const override;

  void packCellForEdge(const Cell *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackEdgeFromCell(Edge *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalCellToEdge(const Cell *sender, Edge *receiver) const override;

  void packCellForVertex(const Cell *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const override;

  void unpackVertexFromCell(Vertex *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const override;

  void communicateLocalCellToVertex(const Cell *sender, Vertex *receiver) const override;

  
private:

  using communication::DoFSpacePackInfo< ValueType >::level_;
  using communication::DoFSpacePackInfo< ValueType >::dataIDVertex_;
  using communication::DoFSpacePackInfo< ValueType >::dataIDEdge_;
  using communication::DoFSpacePackInfo< ValueType >::dataIDFace_;
  using communication::DoFSpacePackInfo< ValueType >::dataIDCell_;
  using communication::DoFSpacePackInfo< ValueType >::storage_;

  BoundaryCondition boundaryCondition_;
  DoFType boundaryTypeToSkip_;

};

/// @name Vertex to Edge
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packVertexForEdge(const Vertex *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const 
{
  WALBERLA_ABORT( "Additive communication Vertex -> Edge not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackEdgeFromVertex(Edge *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Vertex -> Edge not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalVertexToEdge(const Vertex *sender, Edge *receiver) const
{
  WALBERLA_ABORT( "Additive communication Vertex -> Edge not supported." );
}

///@}
/// @name Edge to Vertex
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packEdgeForVertex(const Edge *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Edge -> Vertex not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackVertexFromEdge(Vertex *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Edge -> Vertex not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalEdgeToVertex(const Edge *sender, Vertex *receiver) const
{
  WALBERLA_ABORT( "Additive communication Edge -> Vertex not supported." );
}

///@}
/// @name Edge to Face
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packEdgeForFace(const Edge *sender, const PrimitiveID &/*receiver*/, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Edge -> Face not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackFaceFromEdge(Face *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Edge -> Face not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalEdgeToFace(const Edge *sender, Face *receiver) const
{
  WALBERLA_ABORT( "Additive communication Edge -> Face not supported." );
}

///@}
/// @name Face to Edge
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packFaceForEdge(const Face *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_CHECK( !this->storage_.lock()->hasGlobalCells(), "Additive communication Face -> Edge only meaningful in 2D." );
  WALBERLA_ABORT( "Additive communication Face -> Edge not implemented." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackEdgeFromFace(Edge *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_CHECK( !this->storage_.lock()->hasGlobalCells(), "Additive communication Face -> Edge only meaningful in 2D." );
  WALBERLA_ABORT( "Additive communication Face -> Edge not implemented." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalFaceToEdge(const Face *sender, Edge *receiver) const
{
  WALBERLA_CHECK( !this->storage_.lock()->hasGlobalCells(), "Additive communication Face -> Edge only meaningful in 2D." );
  WALBERLA_ABORT( "Additive communication Face -> Edge not implemented." );
}

///@}
/// @name Face to Cell
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packFaceForCell(const Face *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Face -> Cell not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackCellFromFace(Cell *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_ABORT( "Additive communication Face -> Cell not supported." );
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalFaceToCell(const Face *sender, Cell *receiver) const
{
  WALBERLA_ABORT( "Additive communication Face -> Cell not supported." );
}

///@}
/// @name Cell to Face
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packCellForFace(const Cell *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Face only meaningful in 3D." );

  const ValueType * cellData = sender->getData( dataIDCell_ )->getPointer( level_ );
  const uint_t localFaceID = sender->getLocalFaceID( receiver );
  const uint_t iterationVertex0 = sender->getFaceLocalVertexToCellLocalVertexMaps().at( localFaceID ).at( 0 );
  const uint_t iterationVertex1 = sender->getFaceLocalVertexToCellLocalVertexMaps().at( localFaceID ).at( 1 );
  const uint_t iterationVertex2 = sender->getFaceLocalVertexToCellLocalVertexMaps().at( localFaceID ).at( 2 );

  for ( const auto & it : vertexdof::macrocell::BorderIterator( level_, iterationVertex0, iterationVertex1, iterationVertex2, 0 ) )
  {
    buffer << cellData[ vertexdof::macrocell::indexFromVertex( level_, it.x(), it.y(), it.z(), stencilDirection::VERTEX_C ) ];
  }
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackFaceFromCell(Face *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Face only meaningful in 3D." );

  ValueType * faceData = receiver->getData( dataIDFace_ )->getPointer( level_ );

  WALBERLA_ASSERT_GREATER( receiver->getNumNeighborCells(), 0 );
  WALBERLA_ASSERT( receiver->neighborPrimitiveExists( sender ) );

  if ( boundaryCondition_.getBoundaryType( receiver->getMeshBoundaryFlag() ) == boundaryTypeToSkip_ )
  {
    for ( const auto & it : vertexdof::macroface::Iterator( level_ ))
    {
      real_t tmp;
      buffer >> tmp;
    }
  }
  else
  {
    for ( const auto & it : vertexdof::macroface::Iterator( level_ ))
    {
      real_t tmp;
      buffer >> tmp;
      faceData[vertexdof::macroface::indexFromVertex( level_, it.x(), it.y(), stencilDirection::VERTEX_C )] += tmp;
    }
  }
}

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalCellToFace(const Cell *sender, Face *receiver) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Face -> Edge only meaningful in 3D." );

  WALBERLA_ASSERT_GREATER( receiver->getNumNeighborCells(), 0 );
  WALBERLA_ASSERT( receiver->neighborPrimitiveExists( sender->getID() ) );

  if ( boundaryCondition_.getBoundaryType( receiver->getMeshBoundaryFlag() ) == boundaryTypeToSkip_ )
    return;

  const ValueType *cellData = sender->getData( dataIDCell_ )->getPointer( level_ );
  const uint_t localFaceID = sender->getLocalFaceID( receiver->getID());
  const uint_t iterationVertex0 = sender->getFaceLocalVertexToCellLocalVertexMaps().at( localFaceID ).at( 0 );
  const uint_t iterationVertex1 = sender->getFaceLocalVertexToCellLocalVertexMaps().at( localFaceID ).at( 1 );
  const uint_t iterationVertex2 = sender->getFaceLocalVertexToCellLocalVertexMaps().at( localFaceID ).at( 2 );

  ValueType *faceData = receiver->getData( dataIDFace_ )->getPointer( level_ );

  auto cellIterator = vertexdof::macrocell::BorderIterator( level_, iterationVertex0, iterationVertex1, iterationVertex2, 0 );

  for ( const auto & it : vertexdof::macroface::Iterator( level_ ))
  {
    auto cellIdx = *cellIterator;
    faceData[vertexdof::macroface::indexFromVertex( level_, it.x(), it.y(), stencilDirection::VERTEX_C )] +=
    cellData[vertexdof::macrocell::indexFromVertex( level_, cellIdx.x(), cellIdx.y(), cellIdx.z(), stencilDirection::VERTEX_C )];
    cellIterator++;
  }

  WALBERLA_ASSERT( cellIterator == cellIterator.end() );
}

///@}
/// @name Cell to Edge
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packCellForEdge(const Cell *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Edge only meaningful in 3D." );

  const ValueType * cellData = sender->getData( dataIDCell_ )->getPointer( level_ );
  const uint_t localEdgeID = sender->getLocalEdgeID( receiver );
  const uint_t iterationVertex0 = sender->getEdgeLocalVertexToCellLocalVertexMaps().at( localEdgeID ).at( 0 );
  const uint_t iterationVertex1 = sender->getEdgeLocalVertexToCellLocalVertexMaps().at( localEdgeID ).at( 1 );
  std::set< uint_t > possibleIterationVertices = {0, 1, 2, 3};
  possibleIterationVertices.erase( iterationVertex0 );
  possibleIterationVertices.erase( iterationVertex1 );
  const uint_t iterationVertex2 = *possibleIterationVertices.begin();

  const uint_t edgeSize = levelinfo::num_microvertices_per_edge( level_ );
  auto it  = vertexdof::macrocell::BorderIterator( level_, iterationVertex0, iterationVertex1, iterationVertex2, 0 );
  for ( uint_t i = 0; i < edgeSize; i++ )
  {
    buffer << cellData[ vertexdof::macrocell::indexFromVertex( level_, it->x(), it->y(), it->z(), stencilDirection::VERTEX_C ) ];
    it++;
  }
}


template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackEdgeFromCell(Edge *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Edge only meaningful in 3D." );

  ValueType * edgeData = receiver->getData( dataIDEdge_ )->getPointer( level_ );

  WALBERLA_ASSERT_GREATER( receiver->getNumNeighborCells(), 0 );
  WALBERLA_ASSERT( receiver->neighborPrimitiveExists( sender ) );

  if ( boundaryCondition_.getBoundaryType( receiver->getMeshBoundaryFlag() ) == boundaryTypeToSkip_ )
  {
    for ( const auto & it : vertexdof::macroedge::Iterator( level_ ))
    {
      real_t tmp;
      buffer >> tmp;
    }
  }
  else
  {
    for ( const auto & it : vertexdof::macroedge::Iterator( level_ ))
    {
      real_t tmp;
      buffer >> tmp;
      edgeData[vertexdof::macroedge::indexFromVertex( level_, it.x(), stencilDirection::VERTEX_C )] += tmp;
    }
  }
}


template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalCellToEdge(const Cell *sender, Edge *receiver) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Edge only meaningful in 3D." );

  WALBERLA_ASSERT_GREATER( receiver->getNumNeighborCells(), 0 );
  WALBERLA_ASSERT( receiver->neighborPrimitiveExists( sender->getID() ) );

  if ( boundaryCondition_.getBoundaryType( receiver->getMeshBoundaryFlag() ) == boundaryTypeToSkip_ )
    return;

        ValueType * edgeData = receiver->getData( dataIDEdge_ )->getPointer( level_ );
  const ValueType * cellData = sender->getData( dataIDCell_ )->getPointer( level_ );
  const uint_t localEdgeID = sender->getLocalEdgeID( receiver->getID() );
  const uint_t iterationVertex0 = sender->getEdgeLocalVertexToCellLocalVertexMaps().at( localEdgeID ).at( 0 );
  const uint_t iterationVertex1 = sender->getEdgeLocalVertexToCellLocalVertexMaps().at( localEdgeID ).at( 1 );
  std::set< uint_t > possibleIterationVertices = {0, 1, 2, 3};
  possibleIterationVertices.erase( iterationVertex0 );
  possibleIterationVertices.erase( iterationVertex1 );
  const uint_t iterationVertex2 = *possibleIterationVertices.begin();

  const uint_t edgeSize = levelinfo::num_microvertices_per_edge( level_ );
  auto it  = vertexdof::macrocell::BorderIterator( level_, iterationVertex0, iterationVertex1, iterationVertex2, 0 );
  for ( uint_t i = 0; i < edgeSize; i++ )
  {
    edgeData[i] += cellData[ vertexdof::macrocell::indexFromVertex( level_, it->x(), it->y(), it->z(), stencilDirection::VERTEX_C ) ];
    it++;
  }
}

///@}
/// @name Cell to Vertex
///@{

template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::packCellForVertex(const Cell *sender, const PrimitiveID &receiver, walberla::mpi::SendBuffer &buffer) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Vertex only meaningful in 3D." );

  const ValueType * cellData = sender->getData( dataIDCell_ )->getPointer( level_ );
  const uint_t localVertexID = sender->getLocalVertexID( receiver );
  indexing::Index microVertexIndexInMacroCell( 0, 0, 0 );
  switch ( localVertexID )
  {
    case 1:
      microVertexIndexInMacroCell.x() = levelinfo::num_microvertices_per_edge( level_ ) - 1;
    case 2:
      microVertexIndexInMacroCell.y() = levelinfo::num_microvertices_per_edge( level_ ) - 1;
    case 3:
      microVertexIndexInMacroCell.z() = levelinfo::num_microvertices_per_edge( level_ ) - 1;
    default:
      break;
  }
  buffer << cellData[ vertexdof::macrocell::indexFromVertex( level_,
                                                             microVertexIndexInMacroCell.x(),
                                                             microVertexIndexInMacroCell.y(),
                                                             microVertexIndexInMacroCell.z(),
                                                             stencilDirection::VERTEX_C ) ];
}


template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::unpackVertexFromCell(Vertex *receiver, const PrimitiveID &sender, walberla::mpi::RecvBuffer &buffer) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Vertex only meaningful in 3D." );

  ValueType * vertexData = receiver->getData( dataIDVertex_ )->getPointer( level_ );

  WALBERLA_ASSERT_GREATER( receiver->getNumNeighborCells(), 0 );
  WALBERLA_ASSERT( receiver->neighborPrimitiveExists( sender ) );

  if ( boundaryCondition_.getBoundaryType( receiver->getMeshBoundaryFlag() ) == boundaryTypeToSkip_ )
  {
    real_t tmp;
    buffer >> tmp;
  }
  else
  {
    real_t tmp;
    buffer >> tmp;
    vertexData[ 0 ] += tmp;
  }
}


template< typename ValueType >
void VertexDoFAdditivePackInfo< ValueType >::communicateLocalCellToVertex(const Cell *sender, Vertex *receiver) const
{
  WALBERLA_CHECK( this->storage_.lock()->hasGlobalCells(), "Additive communication Cell -> Vertex only meaningful in 3D." );

  WALBERLA_ASSERT_GREATER( receiver->getNumNeighborCells(), 0 );
  WALBERLA_ASSERT( receiver->neighborPrimitiveExists( sender->getID() ) );

  if ( boundaryCondition_.getBoundaryType( receiver->getMeshBoundaryFlag() ) == boundaryTypeToSkip_ )
    return;

        ValueType * vertexData = receiver->getData( dataIDVertex_ )->getPointer( level_ );
  const ValueType * cellData = sender->getData( dataIDCell_ )->getPointer( level_ );
  const uint_t localVertexID = sender->getLocalVertexID( receiver->getID() );
  indexing::Index microVertexIndexInMacroCell( 0, 0, 0 );
  switch ( localVertexID )
  {
    case 1:
      microVertexIndexInMacroCell.x() = levelinfo::num_microvertices_per_edge( level_ ) - 1;
    case 2:
      microVertexIndexInMacroCell.y() = levelinfo::num_microvertices_per_edge( level_ ) - 1;
    case 3:
      microVertexIndexInMacroCell.z() = levelinfo::num_microvertices_per_edge( level_ ) - 1;
    default:
      break;
  }
  vertexData[0] += cellData[ vertexdof::macrocell::indexFromVertex( level_,
                                                                    microVertexIndexInMacroCell.x(),
                                                                    microVertexIndexInMacroCell.y(),
                                                                    microVertexIndexInMacroCell.z(),
                                                                    stencilDirection::VERTEX_C ) ];
}

///@}

} //namespace hhg