/*
 * Copyright (c) 2017-2019 Dominik Thoennes.
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
// test that the product one^T*M*one with mass matrix M and vector of ones gives area of domain
#include "core/Environment.h"
#include <core/math/Constants.h>

#include "hyteg/dataexport/VTKOutput.hpp"
#include "hyteg/elementwiseoperators/P1ElementwiseOperator.hpp"
#include "hyteg/elementwiseoperators/P2ElementwiseOperator.hpp"
#include "hyteg/p1functionspace/P1VariableOperator.hpp"
#include "hyteg/p2functionspace/P2ConstantOperator.hpp"
#include "hyteg/p2functionspace/P2Function.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/geometry/CircularMap.hpp"
#include "hyteg/geometry/PolarCoordsMap.hpp"

using walberla::math::pi;
using walberla::real_t;
using walberla::uint_t;

using namespace hyteg;

template < typename OperatorType >
void checkArea( std::shared_ptr< PrimitiveStorage > storage, real_t area, std::string tag, const uint_t minLevel = 2, bool outputVTK = false )
{
   // const uint_t minLevel = 2;
   const uint_t maxLevel = storage->hasGlobalCells() ? 3 : 4;

   OperatorType massOp( storage, minLevel, maxLevel );

   typename OperatorType::srcType aux( "aux", storage, minLevel, maxLevel );
   typename OperatorType::srcType vecOfOnes( "vecOfOnes", storage, minLevel, maxLevel );

   for ( uint_t lvl = minLevel; lvl <= maxLevel; ++lvl )
   {
      vecOfOnes.interpolate( real_c( 1.0 ), lvl, All );
      massOp.apply( vecOfOnes, aux, lvl, All );
      real_t measure = vecOfOnes.dotGlobal( aux, lvl );

      if( outputVTK ) {
        VTKOutput vtkOutput( "../../output", tag, storage );
        vtkOutput.add( vecOfOnes );
        vtkOutput.add( aux );
        vtkOutput.write( lvl );
      }

      WALBERLA_LOG_INFO_ON_ROOT( "measure = " << std::scientific << measure << " (" << tag << ")" );
      WALBERLA_CHECK_FLOAT_EQUAL( measure, area );
   }
}

void setMap( SetupPrimitiveStorage& setupStorage, std::shared_ptr<GeometryMap> map ) {
  for( auto it : setupStorage.getFaces() ) {
    setupStorage.setGeometryMap( it.second->getID(), map );
  }
  for( auto it : setupStorage.getEdges() ) {
    setupStorage.setGeometryMap( it.second->getID(), map );
  }
  for( auto it : setupStorage.getVertices() ) {
    setupStorage.setGeometryMap( it.second->getID(), map );
  }
}

void logSectionHeader( const char* header ) {
  std::string hdr( header );
  size_t len = hdr.length();
  std::string separator( len + 2, '-' );
  WALBERLA_LOG_INFO_ON_ROOT( separator << "\n " << hdr << "\n" << separator );
}


int main( int argc, char** argv )
{
   walberla::debug::enterTestMode();

   walberla::mpi::Environment MPIenv( argc, argv );
   walberla::logging::Logging::instance()->setLogLevel( walberla::logging::Logging::PROGRESS );
   walberla::MPIManager::instance()->useWorldComm();

   // ----------
   //  2D Tests
   // ----------

   // Test with rectangle
   logSectionHeader( "Testing with RECTANGLE" );
   MeshInfo meshInfo = MeshInfo::meshRectangle( Point2D( {0.0, -1.0} ), Point2D( {2.0, 3.0} ), MeshInfo::CRISSCROSS, 1, 2 );
   SetupPrimitiveStorage               setupStorage( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   std::shared_ptr< PrimitiveStorage > storage = std::make_shared< PrimitiveStorage >( setupStorage );

   checkArea< P1ConstantMassOperator >( storage, 8.0, "P1ConstantMassOperator" );
   checkArea< P2ConstantMassOperator >( storage, 8.0, "P2ConstantMassOperator" );
   checkArea< P1ElementwiseMassOperator >( storage, 8.0, "P1ElementwiseMassOperator" );
   checkArea< P2ElementwiseMassOperator >( storage, 8.0, "P2ElementwiseMassOperator" );

   // Test with backward facing step
   logSectionHeader( "Testing with BFS" );
   meshInfo = MeshInfo::fromGmshFile( "../../data/meshes/bfs_12el.msh" );
   SetupPrimitiveStorage setupStorageBFS( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   std::shared_ptr< PrimitiveStorage > storageBFS = std::make_shared< PrimitiveStorage >( setupStorageBFS );

   checkArea< P1ConstantMassOperator >( storageBFS, 1.75, "P1ConstantMassOperator" );
   checkArea< P2ConstantMassOperator >( storageBFS, 1.75, "P2ConstantMassOperator" );
   checkArea< P1ElementwiseMassOperator >( storageBFS, 1.75, "P1ElementwiseMassOperator" );
   checkArea< P2ElementwiseMassOperator >( storageBFS, 1.75, "P2ElementwiseMassOperator" );

   // ----------
   //  3D Tests
   // ----------

   // test with cuboid
   logSectionHeader( "Testing with Cuboid" );
   meshInfo = MeshInfo::meshCuboid( Point3D( {-1.0, -1.0, 0.0} ), Point3D( {2.0, 0.0, 2.0} ), 1, 2, 1 );
   SetupPrimitiveStorage setupStorageCuboid( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   std::shared_ptr< PrimitiveStorage > storageCuboid = std::make_shared< PrimitiveStorage >( setupStorageCuboid );

   checkArea< P1ConstantMassOperator >( storageCuboid, 6.0, "P1ConstantMassOperator" );
   checkArea< P2ConstantMassOperator >( storageCuboid, 6.0, "P2ConstantMassOperator" );
   checkArea< P1ElementwiseMassOperator >( storageCuboid, 6.0, "P1ElementwiseMassOperator" );
   checkArea< P2ElementwiseMassOperator >( storageCuboid, 6.0, "P2ElementwiseMassOperator" );

   // Test with coarse representation of thick spherical shell
   logSectionHeader( "Testing with Icosahedral Shell" );
   meshInfo = MeshInfo::meshSphericalShell( 2, {1.0, 2.0} );
   SetupPrimitiveStorage setupStorageTSS( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   std::shared_ptr< PrimitiveStorage > storageTSS = std::make_shared< PrimitiveStorage >( setupStorageTSS );
   real_t                              edgeLength = 8.0 / ( std::sqrt( 10.0 + 2.0 * std::sqrt( 5.0 ) ) );
   real_t                              volume = 5.0 / 12.0 * ( 3.0 + std::sqrt( 5.0 ) ) * edgeLength * edgeLength * edgeLength;
   edgeLength /= 2.0;
   volume -= 5.0 / 12.0 * ( 3.0 + std::sqrt( 5.0 ) ) * edgeLength * edgeLength * edgeLength;

   checkArea< P1ConstantMassOperator >( storageTSS, volume, "P1ConstantMassOperator" );
   checkArea< P2ConstantMassOperator >( storageTSS, volume, "P2ConstantMassOperator" );
   checkArea< P1ElementwiseMassOperator >( storageTSS, volume, "P1ElementwiseMassOperator" );
   checkArea< P2ElementwiseMassOperator >( storageTSS, volume, "P2ElementwiseMassOperator" );

   // -------------------
   //  2D Blending Tests
   // -------------------

   // Test with annulus
   logSectionHeader( "Testing with BLENDING( ANNULUS )" );
   MeshInfo meshInfoPolar = MeshInfo::meshRectangle( Point2D( { 1.0, 0.0 } ), Point2D( { 2.0, 2.0*pi } ), MeshInfo::CROSS, 1, 6 );
   SetupPrimitiveStorage               setupStoragePolar( meshInfoPolar, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   setMap( setupStoragePolar, std::make_shared< PolarCoordsMap >() );
   std::shared_ptr< PrimitiveStorage > storagePolar = std::make_shared< PrimitiveStorage >( setupStoragePolar );

   checkArea< P1BlendingMassOperator >( storagePolar, 3.0 * pi, "P1BlendingMassOperator" );
   checkArea< P1ElementwiseBlendingMassOperator >( storagePolar, 3.0 * pi, "P1ElementwiseBlendingMassOperator" );

   // Test with unit square containing circular hole
   logSectionHeader( "Testing with BLENDING( SQUARE with CIRCULAR HOLE )" );
   MeshInfo              meshInfoHole = MeshInfo::fromGmshFile( "../../data/meshes/unitsquare_with_circular_hole.msh" );
   SetupPrimitiveStorage setupStorageHole( meshInfoHole, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );

   Point3D circleCenter{ { 0.5, 0.5, 0 } };
   real_t  circleRadius = 0.25;

   for ( const auto& it : setupStorageHole.getFaces() )
   {
      Face& face = *( it.second );

      std::vector< PrimitiveID > neighborEdgesOnBoundary = face.neighborEdges();
      neighborEdgesOnBoundary.erase(
          std::remove_if( neighborEdgesOnBoundary.begin(),
                          neighborEdgesOnBoundary.end(),
                          [&setupStorageHole]( const PrimitiveID& id ) { return !setupStorageHole.onBoundary( id ); } ),
          neighborEdgesOnBoundary.end() );

      if ( neighborEdgesOnBoundary.size() > 0 )
      {
         Edge& edge = *setupStorageHole.getEdge( neighborEdgesOnBoundary[0] );

         if ( ( edge.getCoordinates()[0] - circleCenter ).norm() < 0.4 )
         {
            setupStorageHole.setGeometryMap( edge.getID(),
                                         std::make_shared< CircularMap >( face, setupStorageHole, circleCenter, circleRadius ) );
            setupStorageHole.setGeometryMap( face.getID(),
                                         std::make_shared< CircularMap >( face, setupStorageHole, circleCenter, circleRadius ) );
         }
      }
   }
   std::shared_ptr< PrimitiveStorage > storageHole = std::make_shared< PrimitiveStorage >( setupStorageHole );

   checkArea< P1BlendingMassOperator >( storageHole, 1.0 - pi / 16.0, "P1BlendingMassOperator", 3 );
   checkArea< P1ElementwiseBlendingMassOperator >( storageHole, 1.0 - pi / 16.0, "P1ElementwiseBlendingMassOperator", 3 );

   return EXIT_SUCCESS;
}