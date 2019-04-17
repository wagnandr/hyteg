#include "core/Environment.h"
#include "core/logging/Logging.h"
#include "core/timing/Timer.h"
#include "core/math/Random.h"

#include "tinyhhg_core/p2functionspace/P2Function.hpp"
#include "tinyhhg_core/p2functionspace/P2ConstantOperator.hpp"
#include "tinyhhg_core/gridtransferoperators/P2toP2QuadraticProlongation.hpp"
#include "tinyhhg_core/gridtransferoperators/P2toP2QuadraticRestriction.hpp"
#include "tinyhhg_core/solvers/CGSolver.hpp"
#include "tinyhhg_core/solvers/EmptySolver.hpp"
#include "tinyhhg_core/solvers/GeometricMultigridSolver.hpp"
#include <tinyhhg_core/solvers/GaussSeidelSmoother.hpp>
#include "tinyhhg_core/primitivestorage/SetupPrimitiveStorage.hpp"
#include "tinyhhg_core/primitivestorage/PrimitiveStorage.hpp"
#include "tinyhhg_core/primitivestorage/Visualization.hpp"
#include "tinyhhg_core/VTKWriter.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;

using namespace hhg;

int main( int argc, char* argv[] )
{
  walberla::Environment walberlaEnv( argc, argv );
  walberla::MPIManager::instance()->useWorldComm();

  const uint_t      minLevel        = 2;
  const uint_t      maxLevel        = 3;
  const std::string meshFile        = "../../data/meshes/3D/regular_octahedron_8el.msh";

  const uint_t      numVCycles      = 10;

  const bool        writeVTK        = false;

  const auto meshInfo = MeshInfo::fromGmshFile( meshFile );
  SetupPrimitiveStorage setupStorage( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
  setupStorage.setMeshBoundaryFlagsOnBoundary( 1, 0, true );
  const auto storage = std::make_shared< PrimitiveStorage >( setupStorage );

  WALBERLA_CHECK( storage->hasGlobalCells() );

  writeDomainPartitioningVTK( storage, "../../output", "P2_GMG_3D_convergence_partitioning" );

  P2ConstantLaplaceOperator laplaceOperator3D( storage, minLevel, maxLevel );

  std::function< real_t( const hhg::Point3D& ) > exact = []( const hhg::Point3D & p ) -> real_t
  {
      return sin(p[0]) * sinh(p[1]) * p[2];
  };

  std::function< real_t( const hhg::Point3D& ) > zero = []( const hhg::Point3D & ) -> real_t
  {
      return 0.0;
  };

  std::function< real_t( const hhg::Point3D& ) > one = []( const hhg::Point3D & ) -> real_t
  {
      return 1.0;
  };

  std::function< real_t( const hhg::Point3D& ) > rand = []( const hhg::Point3D & ) -> real_t
  {
      return walberla::math::realRandom( 0.0, 1.0 );
  };

  hhg::P2Function< real_t > res( "r", storage, minLevel, maxLevel );
  hhg::P2Function< real_t > f( "f", storage, minLevel, maxLevel );
  hhg::P2Function< real_t > u( "u", storage, minLevel, maxLevel );
  hhg::P2Function< real_t > uExact( "u_exact", storage, minLevel, maxLevel );
  hhg::P2Function< real_t > err( "err", storage, minLevel, maxLevel );
  hhg::P2Function< real_t > oneFunction( "oneFunction", storage, minLevel, maxLevel );

  u.interpolate( rand, maxLevel, DoFType::Inner );
  u.interpolate( exact, maxLevel, DoFType::DirichletBoundary );
  f.interpolate( zero, maxLevel, DoFType::All );
  res.interpolate( zero, maxLevel, DoFType::All );
  uExact.interpolate( exact, maxLevel, DoFType::All );
  oneFunction.interpolate( one, maxLevel, DoFType::All );

  auto smoother = std::make_shared< hhg::GaussSeidelSmoother<hhg::P2ConstantLaplaceOperator>  >();
  auto coarseGridSolver = std::make_shared< hhg::CGSolver< hhg::P2ConstantLaplaceOperator > >( storage, minLevel, maxLevel );
  auto restrictionOperator = std::make_shared< hhg::P2toP2QuadraticRestriction>();
  auto prolongationOperator = std::make_shared< hhg::P2toP2QuadraticProlongation >();

  auto gmgSolver = hhg::GeometricMultigridSolver< hhg::P2ConstantLaplaceOperator >(
     storage, smoother, coarseGridSolver, restrictionOperator, prolongationOperator, minLevel, maxLevel, 2, 2 );

  const real_t numPointsHigherLevel = oneFunction.dotGlobal( oneFunction, maxLevel, DoFType::Inner );

  VTKOutput vtkOutput("../../output", "P2GMG3DConvergenceTest", storage);
  vtkOutput.add( u );
  vtkOutput.add( err );

  if ( writeVTK )
  {
    vtkOutput.write( maxLevel, 0 );
  }

  err.assign( {1.0, -1.0}, {u, uExact}, maxLevel );
  laplaceOperator3D.apply( u, res, maxLevel, DoFType::Inner );

  real_t discrL2ResHigherLevel = res.dotGlobal( res, maxLevel, DoFType::Inner ) / numPointsHigherLevel;

  real_t lastResidual = discrL2ResHigherLevel;

  for ( uint_t i = 1; i <= numVCycles; i++ )
  {
    gmgSolver.solve( laplaceOperator3D, u, f, maxLevel);

    err.assign( {1.0, -1.0}, {u, uExact}, maxLevel );
    laplaceOperator3D.apply( u, res, maxLevel, DoFType::Inner );

    if ( writeVTK )
    {
      vtkOutput.write( maxLevel, i );
    }

    const real_t discrL2ErrHigherLevel = err.dotGlobal( err, maxLevel, DoFType::Inner ) / numPointsHigherLevel;
    discrL2ResHigherLevel = res.dotGlobal( res, maxLevel, DoFType::Inner ) / numPointsHigherLevel;

    const real_t discrResConvRate = discrL2ResHigherLevel / lastResidual;
    WALBERLA_CHECK_LESS( discrResConvRate, 4.2e-02 )

    lastResidual = discrL2ResHigherLevel;

    WALBERLA_LOG_INFO_ON_ROOT( "After " << std::setw(3) << i << " VCycles: Residual: " << std::scientific << discrL2ResHigherLevel << " | convRate: " << discrResConvRate << " | Error L2: " << discrL2ErrHigherLevel );

  }

  return 0;
}
