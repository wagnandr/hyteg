#include "core/mpi/MPIManager.h"
#include "core/timing/Timer.h"

#include "hyteg/composites/P1StokesBlockLaplaceOperator.hpp"
#include "hyteg/composites/P1StokesFunction.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/loadbalancing/SimpleBalancer.hpp"
#include "hyteg/solvers/CGSolver.hpp"

using walberla::real_t;

int main( int argc, char* argv[] )
{
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   walberla::MPIManager::instance()->useWorldComm();
   WALBERLA_LOG_INFO_ON_ROOT( "HyTeG CG Test\n" );

   std::string meshFileName = "../../data/meshes/quad_4el.msh";

   hyteg::MeshInfo              meshInfo = hyteg::MeshInfo::fromGmshFile( meshFileName );
   hyteg::SetupPrimitiveStorage setupStorage( meshInfo, walberla::uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );

   hyteg::loadbalancing::roundRobin( setupStorage );

   size_t minLevel = 2;
   size_t maxLevel = 2;
   //size_t maxiter  = 10000;

   std::shared_ptr< hyteg::PrimitiveStorage > storage = std::make_shared< hyteg::PrimitiveStorage >( setupStorage );

   hyteg::P1StokesFunction< real_t > r( "r", storage, minLevel, maxLevel );
   hyteg::P1StokesFunction< real_t > f( "f", storage, minLevel, maxLevel );
   hyteg::P1StokesFunction< real_t > u( "u", storage, minLevel, maxLevel );
   hyteg::P1StokesFunction< real_t > u_exact( "u_exact", storage, minLevel, maxLevel );
   hyteg::P1StokesFunction< real_t > err( "err", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t >       npoints_helper( "npoints_helper", storage, minLevel, maxLevel );

   hyteg::P1StokesBlockLaplaceOperator L( storage, minLevel, maxLevel );

   std::function< real_t( const hyteg::Point3D& ) > exact = []( const hyteg::Point3D& xx ) { return xx[0] * xx[0] - xx[1] * xx[1]; };
   std::function< real_t( const hyteg::Point3D& ) > rhs   = []( const hyteg::Point3D& ) { return 0.0; };
   std::function< real_t( const hyteg::Point3D& ) > ones  = []( const hyteg::Point3D& ) { return 1.0; };

   u.u.interpolate( exact, maxLevel, hyteg::DirichletBoundary );
   u.v.interpolate( exact, maxLevel, hyteg::DirichletBoundary );

   u_exact.u.interpolate( exact, maxLevel );
   u_exact.v.interpolate( exact, maxLevel );

   auto solver = hyteg::CGSolver< hyteg::P1StokesBlockLaplaceOperator >( storage, minLevel, maxLevel );
   walberla::WcTimer timer;
   solver.solve( L, u, f, maxLevel );
   timer.end();
   WALBERLA_LOG_INFO_ON_ROOT( "time was: " << timer.last() );
   err.assign( {1.0, -1.0}, {u, u_exact}, maxLevel );

   npoints_helper.interpolate( ones, maxLevel );

   real_t npoints = npoints_helper.dotGlobal( npoints_helper, maxLevel );

   real_t discr_l2_err = std::sqrt( err.dotGlobal( err, maxLevel ) / npoints );

   WALBERLA_LOG_INFO_ON_ROOT( "discrete L2 error = " << discr_l2_err );

   WALBERLA_CHECK_LESS( discr_l2_err, 5e-17)

   //  hyteg::VTKWriter({ u, u_exact, &f, &r, &err }, maxLevel, "../output", "test");
   return 0;
}