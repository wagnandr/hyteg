#pragma once

#include "tinyhhg_core/Function.hpp"
#include "tinyhhg_core/FunctionMemory.hpp"
#include "tinyhhg_core/PrimitiveID.hpp"
#include "tinyhhg_core/boundary/BoundaryConditions.hpp"
#include "tinyhhg_core/communication/BufferedCommunication.hpp"
#include "tinyhhg_core/p1functionspace/P1DataHandling.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroCell.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroEdge.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroFace.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMacroVertex.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFMemory.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFPackInfo.hpp"
#include "tinyhhg_core/p1functionspace/VertexDoFAdditivePackInfo.hpp"
#include "tinyhhg_core/primitivedata/PrimitiveDataID.hpp"
#include "tinyhhg_core/primitives/Edge.hpp"
#include "tinyhhg_core/primitives/Face.hpp"
#include "tinyhhg_core/primitives/Vertex.hpp"
#include "tinyhhg_core/types/pointnd.hpp"

namespace hhg {
namespace vertexdof {

template < typename ValueType >
class VertexDoFFunction : public Function< VertexDoFFunction< ValueType > >
{
 public:

   VertexDoFFunction( const std::string & name,
                      const std::shared_ptr< PrimitiveStorage >& storage ) :
     Function< VertexDoFFunction< ValueType > >( name ),
     vertexDataID_( storage->generateInvalidPrimitiveDataID< MemoryDataHandling< FunctionMemory< ValueType >, Vertex >, Vertex >() ),
     edgeDataID_(   storage->generateInvalidPrimitiveDataID< MemoryDataHandling< FunctionMemory< ValueType >, Edge >,   Edge >() ),
     faceDataID_(   storage->generateInvalidPrimitiveDataID< MemoryDataHandling< FunctionMemory< ValueType >, Face >,   Face >() ),
     cellDataID_(   storage->generateInvalidPrimitiveDataID< MemoryDataHandling< FunctionMemory< ValueType >, Cell >,   Cell >() )
     {}

   VertexDoFFunction( const std::string&                         name,
                      const std::shared_ptr< PrimitiveStorage >& storage,
                      uint_t                                     minLevel,
                      uint_t                                     maxLevel )
   : VertexDoFFunction( name, storage, minLevel, maxLevel, BoundaryCondition::create012BC() )
   {}

   VertexDoFFunction( const std::string&                         name,
                      const std::shared_ptr< PrimitiveStorage >& storage,
                      uint_t                                     minLevel,
                      uint_t                                     maxLevel,
                      BoundaryCondition                          boundaryCondition )
   : Function< VertexDoFFunction< ValueType > >( name, storage, minLevel, maxLevel )
   , boundaryCondition_( boundaryCondition )
   , boundaryTypeToSkipDuringAdditiveCommunication_( DoFType::DirichletBoundary )
   {
      auto cellVertexDoFFunctionMemoryDataHandling = std::make_shared< MemoryDataHandling< FunctionMemory< ValueType >, Cell > >(
          minLevel, maxLevel, vertexDoFMacroCellFunctionMemorySize );
      auto faceVertexDoFFunctionMemoryDataHandling = std::make_shared< MemoryDataHandling< FunctionMemory< ValueType >, Face > >(
          minLevel, maxLevel, vertexDoFMacroFaceFunctionMemorySize );
      auto edgeVertexDoFFunctionMemoryDataHandling = std::make_shared< MemoryDataHandling< FunctionMemory< ValueType >, Edge > >(
          minLevel, maxLevel, vertexDoFMacroEdgeFunctionMemorySize );
      auto vertexVertexDoFFunctionMemoryDataHandling =
          std::make_shared< MemoryDataHandling< FunctionMemory< ValueType >, Vertex > >(
              minLevel, maxLevel, vertexDoFMacroVertexFunctionMemorySize );

      storage->addCellData( cellDataID_, cellVertexDoFFunctionMemoryDataHandling, name );
      storage->addFaceData( faceDataID_, faceVertexDoFFunctionMemoryDataHandling, name );
      storage->addEdgeData( edgeDataID_, edgeVertexDoFFunctionMemoryDataHandling, name );
      storage->addVertexData( vertexDataID_, vertexVertexDoFFunctionMemoryDataHandling, name );

      for( uint_t level = minLevel; level <= maxLevel; ++level )
      {
         communicators_[level]->addPackInfo( std::make_shared< VertexDoFPackInfo< ValueType > >(
             level, vertexDataID_, edgeDataID_, faceDataID_, cellDataID_, this->getStorage() ) );
         additiveCommunicators_[level]->addPackInfo( std::make_shared< VertexDoFAdditivePackInfo< ValueType > >(
             level, vertexDataID_, edgeDataID_, faceDataID_, cellDataID_, this->getStorage(),
             boundaryCondition_, boundaryTypeToSkipDuringAdditiveCommunication_ ) );

      }
   }

   const PrimitiveDataID< FunctionMemory< ValueType >, Vertex >& getVertexDataID() const { return vertexDataID_; }
   const PrimitiveDataID< FunctionMemory< ValueType >, Edge >&   getEdgeDataID() const { return edgeDataID_; }
   const PrimitiveDataID< FunctionMemory< ValueType >, Face >&   getFaceDataID() const { return faceDataID_; }
   const PrimitiveDataID< FunctionMemory< ValueType >, Cell >&   getCellDataID() const { return cellDataID_; }

   inline void assign( const std::vector< ValueType >                       scalars,
                       const std::vector< VertexDoFFunction< ValueType >* > functions,
                       uint_t                                               level,
                       DoFType                                              flag = All );

   inline void add( const ValueType & scalar,
                    const uint_t &    level,
                    DoFType           flag = All );

   inline void add( const std::vector< ValueType >                       scalars,
                    const std::vector< VertexDoFFunction< ValueType >* > functions,
                    uint_t                                               level,
                    DoFType                                              flag = All );

   inline real_t dot( VertexDoFFunction< ValueType >& rhs, uint_t level, DoFType flag = All );

   inline void integrateDG( DGFunction< ValueType >& rhs, VertexDoFFunction< ValueType >& rhsP1, uint_t level, DoFType flag );

   /// Interpolates a given expression to a VertexDoFFunction
   inline void interpolate( const ValueType & constant, uint_t level, DoFType flag = All ) const;

   inline void interpolate( const std::function< ValueType( const Point3D& ) >& expr, uint_t level, DoFType flag = All );

   inline void interpolateExtended( const std::function< ValueType( const Point3D&, const std::vector< ValueType >& ) >& expr,
                                    const std::vector< VertexDoFFunction* >                                              srcFunctions,
                                    uint_t                                                                               level,
                                    DoFType                                                                              flag = All );

   // TODO: write more general version(s)
   inline real_t getMaxValue( uint_t level, DoFType flag = All );
   inline real_t getMinValue( uint_t level, DoFType flag = All );
   inline real_t getMaxMagnitude( uint_t level, DoFType flag = All, bool mpiReduce = true );

   inline BoundaryCondition getBoundaryCondition() const { return boundaryCondition_; }

   template < typename SenderType, typename ReceiverType >
   inline void startCommunication( const uint_t& level ) const
   {
      if ( isDummy() ) { return; }
      communicators_.at( level )->template startCommunication< SenderType, ReceiverType >();
   }

   template < typename SenderType, typename ReceiverType >
   inline void endCommunication( const uint_t& level ) const
   {
      if ( isDummy() ) { return; }
      communicators_.at( level )->template endCommunication< SenderType, ReceiverType >();
   }

   template < typename SenderType, typename ReceiverType >
   inline void communicate( const uint_t& level ) const
   {
      if ( isDummy() ) { return; }
      communicators_.at( level )->template communicate< SenderType, ReceiverType >();
   }

   template < typename SenderType, typename ReceiverType >
   inline void startAdditiveCommunication( const uint_t& level ) const
   {
      if ( isDummy() ) { return; }
      interpolateByPrimitiveType< ReceiverType >( real_c( 0 ), level, DoFType::All ^ boundaryTypeToSkipDuringAdditiveCommunication_ );
      additiveCommunicators_.at( level )->template startCommunication< SenderType, ReceiverType >();
   }

   template < typename SenderType, typename ReceiverType >
   inline void endAdditiveCommunication( const uint_t& level ) const
   {
     if ( isDummy() ) { return; }
      additiveCommunicators_.at( level )->template endCommunication< SenderType, ReceiverType >();
   }

   template < typename SenderType, typename ReceiverType >
   inline void communicateAdditively( const uint_t& level ) const
   {
     if ( isDummy() ) { return; }
      interpolateByPrimitiveType< ReceiverType >( real_c( 0 ), level, DoFType::All ^ boundaryTypeToSkipDuringAdditiveCommunication_ );
      additiveCommunicators_.at( level )->template communicate< SenderType, ReceiverType >();
   }

   inline void setLocalCommunicationMode( const communication::BufferedCommunicator::LocalCommunicationMode& localCommunicationMode )
   {
     if ( isDummy() ) { return; }
      for( auto& communicator : communicators_ )
      {
         communicator.second->setLocalCommunicationMode( localCommunicationMode );
      }
      for( auto& communicator : additiveCommunicators_ )
      {
        communicator.second->setLocalCommunicationMode( localCommunicationMode );
      }
   }

   using Function< VertexDoFFunction< ValueType > >::isDummy;

 private:

   template< typename PrimitiveType >
   void interpolateByPrimitiveType( const ValueType & constant, uint_t level, DoFType flag = All ) const
   {
     if ( isDummy() ) { return; }
     this->startTiming( "Interpolate" );

     if ( std::is_same< PrimitiveType, Vertex >::value )
     {
       for( const auto& it : this->getStorage()->getVertices() )
       {
         Vertex& vertex = *it.second;

         if( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
         {
           vertexdof::macrovertex::interpolate( level, vertex, vertexDataID_, constant );
         }
       }
     }
     else if ( std::is_same< PrimitiveType, Edge >::value  )
     {
       for( const auto& it : this->getStorage()->getEdges() )
       {
         Edge& edge = *it.second;

         if( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
         {
           vertexdof::macroedge::interpolate( level, edge, edgeDataID_, constant );
         }
       }
     }
     else if ( std::is_same< PrimitiveType, Face >::value )
     {
       for( const auto& it : this->getStorage()->getFaces() )
       {
         Face& face = *it.second;

         if( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
         {
           vertexdof::macroface::interpolate( level, face, faceDataID_, constant );
         }
       }
     }
     else if ( std::is_same< PrimitiveType, Cell >::value )
     {
       for( const auto& it : this->getStorage()->getCells() )
       {
         Cell& cell = *it.second;

         if( testFlag( boundaryCondition_.getBoundaryType( cell.getMeshBoundaryFlag() ), flag ) )
         {
           vertexdof::macrocell::interpolate( level, cell, cellDataID_, constant );
         }
       }
     }

     this->stopTiming( "Interpolate" );
   }

   using Function< VertexDoFFunction< ValueType > >::communicators_;
   using Function< VertexDoFFunction< ValueType > >::additiveCommunicators_;

   inline void enumerate_impl( uint_t level, uint_t& num );

   PrimitiveDataID< FunctionMemory< ValueType >, Vertex > vertexDataID_;
   PrimitiveDataID< FunctionMemory< ValueType >, Edge >   edgeDataID_;
   PrimitiveDataID< FunctionMemory< ValueType >, Face >   faceDataID_;
   PrimitiveDataID< FunctionMemory< ValueType >, Cell >   cellDataID_;

   BoundaryCondition boundaryCondition_;

   DoFType boundaryTypeToSkipDuringAdditiveCommunication_;
};

template < typename ValueType >
inline void VertexDoFFunction< ValueType >::interpolate( const ValueType & constant, uint_t level, DoFType flag ) const
{
  if ( isDummy() ) { return; }
   this->startTiming( "Interpolate" );

   interpolateByPrimitiveType< Vertex >( constant, level, flag );
   interpolateByPrimitiveType< Edge   >( constant, level, flag );
   interpolateByPrimitiveType< Face   >( constant, level, flag );
   interpolateByPrimitiveType< Cell   >( constant, level, flag );

   this->stopTiming( "Interpolate" );
}

template < typename ValueType >
inline void
    VertexDoFFunction< ValueType >::interpolate( const std::function< ValueType( const Point3D& ) >& expr, uint_t level, DoFType flag )
{
  if ( isDummy() ) { return; }
   std::function< ValueType( const Point3D&, const std::vector< ValueType >& ) > exprExtended =
       [&expr]( const hhg::Point3D& x, const std::vector< ValueType >& ) { return expr( x ); };
   interpolateExtended( exprExtended, {}, level, flag );
}

template < typename ValueType >
inline void VertexDoFFunction< ValueType >::interpolateExtended(
    const std::function< ValueType( const Point3D&, const std::vector< ValueType >& ) >& expr,
    const std::vector< VertexDoFFunction* >                                        srcFunctions,
    uint_t                                                                         level,
    DoFType                                                                        flag )
{
  if ( isDummy() ) { return; }
   this->startTiming( "Interpolate" );
   // Collect all source IDs in a vector
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Vertex > > srcVertexIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Edge > >   srcEdgeIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Face > >   srcFaceIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Cell > >   srcCellIDs;

   for( const auto& function : srcFunctions )
   {
      srcVertexIDs.push_back( function->vertexDataID_ );
      srcEdgeIDs.push_back( function->edgeDataID_ );
      srcFaceIDs.push_back( function->faceDataID_ );
      srcCellIDs.push_back( function->cellDataID_ );
   }

   for( const auto& it : this->getStorage()->getVertices() )
   {
      Vertex& vertex = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrovertex::interpolate( vertex, vertexDataID_, srcVertexIDs, expr, level );
      }
   }

   for( const auto& it : this->getStorage()->getEdges() )
   {
      Edge& edge = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroedge::interpolate< ValueType >( level, edge, edgeDataID_, srcEdgeIDs, expr );
      }
   }

   for( auto& it : this->getStorage()->getFaces() )
   {
      Face& face = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroface::interpolate< ValueType >( level, face, faceDataID_, srcFaceIDs, expr );
      }
   }

   for( const auto& it : this->getStorage()->getCells() )
   {
      Cell& cell = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( cell.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrocell::interpolate< ValueType >( level, cell, cellDataID_, srcCellIDs, expr );
      }
   }
   this->stopTiming( "Interpolate" );
}

template < typename ValueType >
inline void VertexDoFFunction< ValueType >::assign( const std::vector< ValueType >                       scalars,
                                                    const std::vector< VertexDoFFunction< ValueType >* > functions,
                                                    size_t                                               level,
                                                    DoFType                                              flag )
{
  if ( isDummy() ) { return; }
   this->startTiming( "Assign" );
   // Collect all source IDs in a vector
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Vertex > > srcVertexIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Edge > >   srcEdgeIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Face > >   srcFaceIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Cell > >   srcCellIDs;

   for( const auto& function : functions )
   {
      srcVertexIDs.push_back( function->vertexDataID_ );
      srcEdgeIDs.push_back( function->edgeDataID_ );
      srcFaceIDs.push_back( function->faceDataID_ );
      srcCellIDs.push_back( function->cellDataID_ );
   }

   for( const auto& it : this->getStorage()->getVertices() )
   {
      Vertex& vertex = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrovertex::assign< ValueType >( vertex, scalars, srcVertexIDs, vertexDataID_, level );
      }
   }

   for( const auto& it : this->getStorage()->getEdges() )
   {
      Edge& edge = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroedge::assign< ValueType >( level, edge, scalars, srcEdgeIDs, edgeDataID_ );
      }
   }

   for( const auto& it : this->getStorage()->getFaces() )
   {
      Face& face = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroface::assign< ValueType >( level, face, scalars, srcFaceIDs, faceDataID_ );
      }
   }

   for( const auto& it : this->getStorage()->getCells() )
   {
      Cell& cell = *it.second;
      if( testFlag( boundaryCondition_.getBoundaryType( cell.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrocell::assign< ValueType >( level, cell, scalars, srcCellIDs, cellDataID_ );
      }
   }
   this->stopTiming( "Assign" );
}

template< typename ValueType >
inline void VertexDoFFunction< ValueType >::add(const ValueType & scalar, const uint_t & level, DoFType flag)
{
  if ( isDummy() ) { return; }
  this->startTiming( "Add" );

  for ( const auto & it : this->getStorage()->getVertices() )
  {
    Vertex & vertex = *it.second;

    if ( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
    {
      vertexdof::macrovertex::add< ValueType >( vertex, scalar, vertexDataID_, level );
    }
  }

  for ( const auto & it : this->getStorage()->getEdges() )
  {
    Edge & edge = *it.second;

    if ( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
    {
      vertexdof::macroedge::add< ValueType >( level, edge, scalar, edgeDataID_ );
    }
  }

  for ( const auto & it : this->getStorage()->getFaces() )
  {
    Face & face = *it.second;

    if ( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
    {
      vertexdof::macroface::add< ValueType >( level, face, scalar, faceDataID_ );
    }
  }

  for ( const auto & it : this->getStorage()->getCells() )
  {
    Cell & cell = *it.second;
    if ( testFlag(  boundaryCondition_.getBoundaryType( cell.getMeshBoundaryFlag() ), flag  ) )
    {
      vertexdof::macrocell::add< ValueType >( level, cell, scalar, cellDataID_ );
    }
  }

  this->stopTiming( "Add" );
}

template < typename ValueType >
inline void VertexDoFFunction< ValueType >::add( const std::vector< ValueType >                       scalars,
                                                 const std::vector< VertexDoFFunction< ValueType >* > functions,
                                                 size_t                                               level,
                                                 DoFType                                              flag )
{
  if ( isDummy() ) { return; }
   this->startTiming( "Add" );
   // Collect all source IDs in a vector
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Vertex > > srcVertexIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Edge > >   srcEdgeIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Face > >   srcFaceIDs;
   std::vector< PrimitiveDataID< FunctionMemory< ValueType >, Cell > >   srcCellIDs;

   for( auto& function : functions )
   {
      srcVertexIDs.push_back( function->vertexDataID_ );
      srcEdgeIDs.push_back( function->edgeDataID_ );
      srcFaceIDs.push_back( function->faceDataID_ );
      srcCellIDs.push_back( function->cellDataID_ );
   }

   for( const auto& it : this->getStorage()->getVertices() )
   {
      Vertex& vertex = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrovertex::add( vertex, scalars, srcVertexIDs, vertexDataID_, level );
      }
   }

   for( const auto& it : this->getStorage()->getEdges() )
   {
      Edge& edge = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroedge::add< ValueType >( level, edge, scalars, srcEdgeIDs, edgeDataID_ );
      }
   }

   for( const auto& it : this->getStorage()->getFaces() )
   {
      Face& face = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroface::add< ValueType >( level, face, scalars, srcFaceIDs, faceDataID_ );
      }
   }

   for( const auto& it : this->getStorage()->getCells() )
   {
      Cell& cell = *it.second;
      if( testFlag( boundaryCondition_.getBoundaryType( cell.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrocell::add< ValueType >( level, cell, scalars, srcCellIDs, cellDataID_ );
      }
   }
   this->stopTiming( "Add" );
}

template < typename ValueType >
inline real_t VertexDoFFunction< ValueType >::dot( VertexDoFFunction< ValueType >& rhs, size_t level, DoFType flag )
{
  if ( isDummy() ) { return real_c(0); }
   this->startTiming( "Dot" );
   real_t scalarProduct = 0.0;

   for( const auto& it : this->getStorage()->getVertices() )
   {
      Vertex& vertex = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
      {
         scalarProduct += vertexdof::macrovertex::dot( vertex, vertexDataID_, rhs.vertexDataID_, level );
      }
   }

   for( const auto& it : this->getStorage()->getEdges() )
   {
      Edge& edge = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
      {
         scalarProduct += vertexdof::macroedge::dot< ValueType >( level, edge, edgeDataID_, rhs.edgeDataID_ );
      }
   }

   for( const auto& it : this->getStorage()->getFaces() )
   {
      Face& face = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
      {
         scalarProduct += vertexdof::macroface::dot< ValueType >( level, face, faceDataID_, rhs.faceDataID_ );
      }
   }

   for( const auto& it : this->getStorage()->getCells() )
   {
      Cell& cell = *it.second;
      if( testFlag( boundaryCondition_.getBoundaryType( cell.getMeshBoundaryFlag() ), flag ) )
      {
         scalarProduct += vertexdof::macrocell::dot< ValueType >( level, cell, cellDataID_, rhs.cellDataID_ );
      }
   }

   walberla::mpi::allReduceInplace( scalarProduct, walberla::mpi::SUM, walberla::mpi::MPIManager::instance()->comm() );

   this->stopTiming( "Dot" );
   return scalarProduct;
}

template < typename ValueType >
inline void VertexDoFFunction< ValueType >::enumerate_impl( uint_t level, uint_t& num )
{
  if ( isDummy() ) { return; }
   /// in contrast to other methods in the function class enumerate needs to communicate due to its usage in the PETSc solvers
   this->startTiming( "Enumerate" );
   for( auto& it : this->getStorage()->getVertices() )
   {
      Vertex& vertex = *it.second;
      vertexdof::macrovertex::enumerate( level, vertex, vertexDataID_, num );
   }

   communicators_[level]->template startCommunication< Vertex, Edge >();

   for( auto& it : this->getStorage()->getEdges() )
   {
      Edge& edge = *it.second;
      vertexdof::macroedge::enumerate< ValueType >( level, edge, edgeDataID_, num );
   }

   communicators_[level]->template startCommunication< Edge, Vertex >();
   communicators_[level]->template startCommunication< Edge, Face >();

   for( auto& it : this->getStorage()->getFaces() )
   {
      Face& face = *it.second;
      vertexdof::macroface::enumerate< ValueType >( level, face, faceDataID_, num );
   }
   communicators_[level]->template startCommunication< Face, Edge >();

   communicators_[level]->template endCommunication< Vertex, Edge >();
   communicators_[level]->template endCommunication< Edge, Vertex >();
   communicators_[level]->template endCommunication< Edge, Face >();
   communicators_[level]->template endCommunication< Face, Edge >();

   this->stopTiming( "Enumerate" );
}

template < typename ValueType >
inline void VertexDoFFunction< ValueType >::integrateDG( DGFunction< ValueType >&        rhs,
                                                         VertexDoFFunction< ValueType >& rhsP1,
                                                         uint_t                          level,
                                                         DoFType                         flag )
{
  if ( isDummy() ) { return; }
   this->startTiming( "integrateDG" );

   rhsP1.startCommunication< Edge, Vertex >( level );
   rhsP1.startCommunication< Face, Edge >( level );

   rhs.template startCommunication< Face, Edge >( level );
   rhs.template endCommunication< Face, Edge >( level );

   rhs.template startCommunication< Edge, Vertex >( level );
   rhs.template endCommunication< Edge, Vertex >( level );

   rhsP1.endCommunication< Edge, Vertex >( level );

   for( auto& it : this->getStorage()->getVertices() )
   {
      Vertex& vertex = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( vertex.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macrovertex::integrateDG< ValueType >(
             vertex, this->getStorage(), rhs.getVertexDataID(), rhsP1.getVertexDataID(), vertexDataID_, level );
      }
   }

   communicators_[level]->template startCommunication< Vertex, Edge >();
   rhsP1.endCommunication< Face, Edge >( level );

   for( auto& it : this->getStorage()->getEdges() )
   {
      Edge& edge = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( edge.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroedge::integrateDG< ValueType >(
             level, edge, this->getStorage(), rhs.getEdgeDataID(), rhsP1.getEdgeDataID(), edgeDataID_ );
      }
   }

   communicators_[level]->template endCommunication< Vertex, Edge >();
   communicators_[level]->template startCommunication< Edge, Face >();

   for( auto& it : this->getStorage()->getFaces() )
   {
      Face& face = *it.second;

      if( testFlag( boundaryCondition_.getBoundaryType( face.getMeshBoundaryFlag() ), flag ) )
      {
         vertexdof::macroface::integrateDG< ValueType >( level, face, rhs.getFaceDataID(), rhsP1.getFaceDataID(), faceDataID_ );
      }
   }

   communicators_[level]->template endCommunication< Edge, Face >();

   this->stopTiming( "integrateDG" );
}

inline void projectMean( VertexDoFFunction< real_t >& pressure, VertexDoFFunction< real_t >& tmp, uint_t level )
{
  if ( pressure.isDummy() ) { return; }
   std::function< real_t( const hhg::Point3D& ) > ones = []( const hhg::Point3D& ) { return 1.0; };

   tmp.interpolate( ones, level );

   real_t numGlobalVertices = tmp.dot( tmp, level, hhg::All );
   real_t mean              = pressure.dot( tmp, level, hhg::All );

   pressure.assign( {1.0, -mean / numGlobalVertices}, {&pressure, &tmp}, level, hhg::All );
}

template < typename ValueType >
inline real_t VertexDoFFunction< ValueType >::getMaxValue( uint_t level, DoFType flag )
{
  if ( isDummy() ) { return real_c(0); }
   real_t localMax = -std::numeric_limits< real_t >::max();

   for( auto& it : this->getStorage()->getFaces() )
   {
      Face&         face   = *it.second;
      const DoFType faceBC = this->getBoundaryCondition().getBoundaryType( face.getMeshBoundaryFlag() );
      if( testFlag( faceBC, flag ) )
      {
         localMax = std::max( localMax, vertexdof::macroface::getMaxValue< ValueType >( level, face, faceDataID_ ) );
      }
   }

   for( auto& it : this->getStorage()->getEdges() )
   {
      Edge&         edge   = *it.second;
      const DoFType edgeBC = this->getBoundaryCondition().getBoundaryType( edge.getMeshBoundaryFlag() );
      if( testFlag( edgeBC, flag ) )
      {
         localMax = std::max( localMax, vertexdof::macroedge::getMaxValue< ValueType >( level, edge, edgeDataID_ ) );
      }
   }

   for( auto& it : this->getStorage()->getVertices() )
   {
      Vertex&       vertex   = *it.second;
      const DoFType vertexBC = this->getBoundaryCondition().getBoundaryType( vertex.getMeshBoundaryFlag() );
      if( testFlag( vertexBC, flag ) )
      {
         localMax = std::max( localMax, vertexdof::macrovertex::getMaxValue< ValueType >( level, vertex, vertexDataID_ ) );
      }
   }

   real_t globalMax = walberla::mpi::allReduce( localMax, walberla::mpi::MAX );

   return globalMax;
}

template < typename ValueType >
inline real_t VertexDoFFunction< ValueType >::getMaxMagnitude( uint_t level, DoFType flag, bool mpiReduce )
{
  if ( isDummy() ) { return real_c(0); }
   real_t localMax = real_t( 0.0 );

   for( auto& it : this->getStorage()->getFaces() )
   {
      Face&         face   = *it.second;
      const DoFType faceBC = this->getBoundaryCondition().getBoundaryType( face.getMeshBoundaryFlag() );
      if( testFlag( faceBC, flag ) )
      {
         localMax = std::max( localMax, vertexdof::macroface::getMaxMagnitude< ValueType >( level, face, faceDataID_ ) );
      }
   }

   for( auto& it : this->getStorage()->getEdges() )
   {
      Edge&         edge   = *it.second;
      const DoFType edgeBC = this->getBoundaryCondition().getBoundaryType( edge.getMeshBoundaryFlag() );
      if( testFlag( edgeBC, flag ) )
      {
         localMax = std::max( localMax, vertexdof::macroedge::getMaxMagnitude< ValueType >( level, edge, edgeDataID_ ) );
      }
   }

   for( auto& it : this->getStorage()->getVertices() )
   {
      Vertex&       vertex   = *it.second;
      const DoFType vertexBC = this->getBoundaryCondition().getBoundaryType( vertex.getMeshBoundaryFlag() );
      if( testFlag( vertexBC, flag ) )
      {
         localMax = std::max( localMax, vertexdof::macrovertex::getMaxMagnitude< ValueType >( level, vertex, vertexDataID_ ) );
      }
   }

   real_t globalMax = localMax;
   if( mpiReduce )
   {
      globalMax = walberla::mpi::allReduce( localMax, walberla::mpi::MAX );
   }

   return globalMax;
}

template < typename ValueType >
inline real_t VertexDoFFunction< ValueType >::getMinValue( uint_t level, DoFType flag )
{
  if ( isDummy() ) { return real_c(0); }
   real_t localMin = std::numeric_limits< real_t >::max();

   for( auto& it : this->getStorage()->getFaces() )
   {
      Face&         face   = *it.second;
      const DoFType faceBC = this->getBoundaryCondition().getBoundaryType( face.getMeshBoundaryFlag() );
      if( testFlag( faceBC, flag ) )
      {
         localMin = std::min( localMin, vertexdof::macroface::getMinValue< ValueType >( level, face, faceDataID_ ) );
      }
   }

   for( auto& it : this->getStorage()->getEdges() )
   {
      Edge&         edge   = *it.second;
      const DoFType edgeBC = this->getBoundaryCondition().getBoundaryType( edge.getMeshBoundaryFlag() );
      if( testFlag( edgeBC, flag ) )
      {
         localMin = std::min( localMin, vertexdof::macroedge::getMinValue< ValueType >( level, edge, edgeDataID_ ) );
      }
   }

   for( auto& it : this->getStorage()->getVertices() )
   {
      Vertex&       vertex   = *it.second;
      const DoFType vertexBC = this->getBoundaryCondition().getBoundaryType( vertex.getMeshBoundaryFlag() );
      if( testFlag( vertexBC, flag ) )
      {
         localMin = std::min( localMin, vertexdof::macrovertex::getMinValue< ValueType >( level, vertex, vertexDataID_ ) );
      }
   }

   real_t globalMin = -walberla::mpi::allReduce( -localMin, walberla::mpi::MAX );

   return globalMin;
}


} // namespace vertexdof
} // namespace hhg
