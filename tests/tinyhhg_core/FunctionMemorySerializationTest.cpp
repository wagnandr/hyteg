#include "core/Environment.h"
#include "core/debug/CheckFunctions.h"
#include "core/debug/TestSubsystem.h"
#include "core/mpi/SendBuffer.h"
#include "core/mpi/RecvBuffer.h"
#include "tinyhhg_core/tinyhhg.hpp"

namespace hhg {


static void testFunctionMemorySerialization()
{
  const uint_t rank         = uint_c( walberla::mpi::MPIManager::instance()->rank() );
  const uint_t numProcesses = uint_c( walberla::mpi::MPIManager::instance()->numProcesses() );

  const std::string meshFileName = "../../data/meshes/bfs_126el.msh";

  const uint_t minLevel = 2;
  const uint_t maxLevel = 3;

  MeshInfo meshInfo = MeshInfo::fromGmshFile( meshFileName );
  SetupPrimitiveStorage setupStorage( meshInfo, uint_c ( walberla::mpi::MPIManager::instance()->numProcesses() ) );

  loadbalancing::roundRobin( setupStorage );

  std::shared_ptr< PrimitiveStorage > storage( new PrimitiveStorage( setupStorage ) );

  P1Function< real_t > x("x", storage, minLevel, maxLevel);
  P1LaplaceOperator A(storage, minLevel, maxLevel);

  std::function<real_t(const hhg::Point3D&)> gradient = [](const hhg::Point3D& xx) { return xx[0]; };

  for ( uint_t level = minLevel; level <= maxLevel; level++ )
  {
    x.interpolate( gradient, level );
  }

  writeDomainPartitioningVTK( storage, "../../output/", "function_memory_serialization_test_domain_before_migration" );
  VTKWriter< P1Function< real_t > >( {&x}, maxLevel, "../../output/", "function_memory_serialization_test_data_before_migration" );

  WALBERLA_LOG_INFO( "Number of local primitives (before migration): " << storage->getNumberOfLocalPrimitives() );

  std::map< hhg::PrimitiveID::IDType, uint_t > primitivesToMigrate;
  std::vector< PrimitiveID > localPrimitiveIDs;
  storage->getPrimitiveIDs( localPrimitiveIDs );
  for ( const auto & id : localPrimitiveIDs )
  {
    primitivesToMigrate[ id.getID() ] = (rank + numProcesses / 2) % numProcesses;
  }
  storage->migratePrimitives( primitivesToMigrate );

  WALBERLA_LOG_INFO( "Number of local primitives (after migration): " << storage->getNumberOfLocalPrimitives() );

  writeDomainPartitioningVTK( storage, "../../output/", "function_memory_serialization_test_domain_after_migration" );
  VTKWriter< P1Function< real_t > >( {&x}, maxLevel, "../../output/", "function_memory_serialization_test_data_after_migration" );

}

} // namespace hhg


int main( int argc, char* argv[] )
{
  walberla::debug::enterTestMode();

  walberla::Environment walberlaEnv(argc, argv);
  walberla::logging::Logging::instance()->setLogLevel( walberla::logging::Logging::PROGRESS );
  walberla::MPIManager::instance()->useWorldComm();
  walberla::debug::enterTestMode();
  hhg::testFunctionMemorySerialization();

  return EXIT_SUCCESS;
}
