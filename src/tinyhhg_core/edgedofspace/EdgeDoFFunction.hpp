#pragma once

#include "core/mpi/all.h"

#include "tinyhhg_core/Function.hpp"
#include "tinyhhg_core/boundary/BoundaryConditions.hpp"

namespace hhg {

using walberla::real_c;

template< typename ValueType >
class FunctionMemory;
template< typename DataType, typename PrimitiveType >
class PrimitiveDataID;

class PrimitiveStorage;
class Vertex;
class Edge;
class Face;
class Cell;

namespace edgedof {
///@name Size Functions
///@{

inline uint_t edgeDoFMacroVertexFunctionMemorySize( const uint_t &level, const Primitive & primitive );

inline uint_t edgeDoFMacroEdgeFunctionMemorySize( const uint_t &level, const Primitive & primitive );

inline uint_t edgeDoFMacroFaceFunctionMemorySize( const uint_t &level, const Primitive & primitive );

inline uint_t edgeDoFMacroCellFunctionMemorySize( const uint_t & level, const Primitive & primitive );

///@}

}// namespace edgedof

template< typename ValueType >
class EdgeDoFFunction : public Function< EdgeDoFFunction< ValueType > >
{
public:

  EdgeDoFFunction( const std::string & name, const std::shared_ptr< PrimitiveStorage > & storage );

  EdgeDoFFunction( const std::string & name, const std::shared_ptr< PrimitiveStorage > & storage, const uint_t & minLevel, const uint_t & maxLevel );

  EdgeDoFFunction( const std::string & name, const std::shared_ptr< PrimitiveStorage > & storage, const uint_t & minLevel, const uint_t & maxLevel, const BoundaryCondition & boundaryCondition );

  inline void
  assign( const std::vector< ValueType > scalars, const std::vector< EdgeDoFFunction< ValueType >* > functions,
          uint_t level, DoFType flag = All );

  inline void
  add( const std::vector< ValueType > scalars, const std::vector< EdgeDoFFunction< ValueType >* > functions,
       uint_t level, DoFType flag = All );

  /// Interpolates a given expression to a EdgeDoFFunction

  inline void
  interpolate( const ValueType& constant, uint_t level, DoFType flag = All );

  inline void
  interpolate( const std::function< ValueType( const Point3D & ) >& expr,
                          uint_t level, DoFType flag = All);

  inline void
  interpolateExtended( const std::function<ValueType(const Point3D &, const std::vector<ValueType>&)> &expr,
                       const std::vector<EdgeDoFFunction<ValueType>*> srcFunctions,
                      uint_t level,
                      DoFType flag = All);

  inline real_t
  dotLocal( EdgeDoFFunction< ValueType >& rhs, uint_t level, DoFType flag = All );

  inline void enumerate( uint_t level );

  const PrimitiveDataID< FunctionMemory< ValueType >, Vertex>   & getVertexDataID() const { return vertexDataID_; }
  const PrimitiveDataID< FunctionMemory< ValueType >,   Edge>   & getEdgeDataID()   const { return edgeDataID_; }
  const PrimitiveDataID< FunctionMemory< ValueType >,   Face>   & getFaceDataID()   const { return faceDataID_; }
  const PrimitiveDataID< FunctionMemory< ValueType >,   Cell>   & getCellDataID()   const { return cellDataID_; }


  inline ValueType getMaxMagnitude( uint_t level, DoFType flag = All, bool mpiReduce = true );

  inline BoundaryCondition getBoundaryCondition() const { return boundaryCondition_; }

  template< typename SenderType, typename ReceiverType >
  inline void startCommunication( const uint_t & level ) const
  {
    if ( isDummy() ) { return; }
    communicators_.at( level )->template startCommunication< SenderType, ReceiverType >();
  }

  template< typename SenderType, typename ReceiverType >
  inline void endCommunication( const uint_t & level ) const
  {
    if ( isDummy() ) { return; }
    communicators_.at( level )->template endCommunication< SenderType, ReceiverType >();
  }

  template< typename SenderType, typename ReceiverType >
  inline void communicate( const uint_t & level ) const
  {
    if ( isDummy() ) { return; }
    communicators_.at( level )->template communicate< SenderType, ReceiverType >();
  }

  inline void setLocalCommunicationMode( const communication::BufferedCommunicator::LocalCommunicationMode & localCommunicationMode )
  {
    if ( isDummy() ) { return; }
    for ( auto & communicator : communicators_ )
    {
      communicator.second->setLocalCommunicationMode( localCommunicationMode );
    }
  }

   using Function< EdgeDoFFunction< ValueType > >::isDummy;

private:

   inline void enumerate( uint_t level, ValueType& offset );

   using Function< EdgeDoFFunction< ValueType > >::communicators_;

   PrimitiveDataID< FunctionMemory< ValueType >, Vertex > vertexDataID_;
   PrimitiveDataID< FunctionMemory< ValueType >, Edge > edgeDataID_;
   PrimitiveDataID< FunctionMemory< ValueType >, Face > faceDataID_;
   PrimitiveDataID< FunctionMemory< ValueType >, Cell > cellDataID_;

   BoundaryCondition boundaryCondition_;

   /// friend P2Function for usage of enumerate
   friend class P2Function< ValueType >;
};


}// namespace hhg
