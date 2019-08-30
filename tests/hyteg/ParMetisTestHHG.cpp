#include "core/Environment.h"
#include "core/debug/CheckFunctions.h"
#include "core/debug/TestSubsystem.h"
#include "core/mpi/SendBuffer.h"
#include "core/mpi/RecvBuffer.h"

#include "hyteg/mesh/MeshInfo.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/loadbalancing/SimpleBalancer.hpp"
#include "hyteg/primitivestorage/loadbalancing/DistributedBalancer.hpp"
#include "hyteg/primitivestorage/Visualization.hpp"

namespace hyteg {


static void testParMetis()
{
  const std::string meshFileName = "../../data/meshes/3D/cube_24el.msh";

  MeshInfo meshInfo = MeshInfo::fromGmshFile( meshFileName );
  SetupPrimitiveStorage setupStorage( meshInfo, uint_c ( walberla::mpi::MPIManager::instance()->numProcesses() ) );

  loadbalancing::roundRobin( setupStorage );

  WALBERLA_LOG_INFO_ON_ROOT( setupStorage );

  std::shared_ptr< PrimitiveStorage > storage( new PrimitiveStorage( setupStorage ) );

  writeDomainPartitioningVTK( storage, "../../output/", "domain_partitioning_after_setup_load_balancing" );

  auto globalInfo = storage->getGlobalInfo();
  WALBERLA_LOG_INFO_ON_ROOT( globalInfo );

  loadbalancing::distributed::parmetis( *storage );

  globalInfo = storage->getGlobalInfo();
  WALBERLA_LOG_INFO_ON_ROOT( globalInfo );

  writeDomainPartitioningVTK( storage, "../../output/", "domain_partitioning_after_distributed_load_balancing" );

}

} // namespace hyteg


int main( int argc, char* argv[] )
{
  walberla::debug::enterTestMode();

  walberla::Environment walberlaEnv(argc, argv);
  walberla::logging::Logging::instance()->setLogLevel( walberla::logging::Logging::PROGRESS );
  walberla::MPIManager::instance()->useWorldComm();
  walberla::debug::enterTestMode();
  hyteg::testParMetis();

  return EXIT_SUCCESS;
}
