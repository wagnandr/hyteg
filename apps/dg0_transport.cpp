#include <core/timing/Timer.h>
#include <tinyhhg_core/tinyhhg.hpp>
#include <fmt/format.h>
#include <core/Environment.h>

using walberla::real_t;
using walberla::uint_t;
using walberla::uint_c;

using namespace hhg;

int main(int argc, char* argv[])
{
  walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
  walberla::MPIManager::instance()->useWorldComm();

  std::string meshFileName = "../data/meshes/quad_4el.msh";

  hhg::MeshInfo meshInfo = hhg::MeshInfo::fromGmshFile( meshFileName );
  hhg::SetupPrimitiveStorage setupStorage( meshInfo, walberla::uint_c ( walberla::mpi::MPIManager::instance()->numProcesses() ) );

  hhg::loadbalancing::roundRobin( setupStorage );

  const uint_t minLevel = 2;
  const uint_t maxLevel = 7;
  const uint_t timesteps = 10;
  real_t dt = 0.25 * std::pow(2.0, -walberla::real_c(maxLevel+1));
  WALBERLA_LOG_DEVEL("dt = " << dt)

  std::function<real_t(const hhg::Point3D&, const std::vector<real_t>&)> initialConcentration = [](const hhg::Point3D& x,  const std::vector<real_t>&) {
    if ((x - Point3D{{{0.5, 0.5, 0.0}}}).norm() < 0.1) {
      return 1.0;
    } else {
      return 0.0;
    }
//    return 1.0;
  };

  std::function<real_t(const hhg::Point3D&)> vel_x = [](const hhg::Point3D&) {
//    return std::pow(x[1], 4.0) * (1.0 - x[0]) - x[0] * std::pow(1.0-x[1], 4.0);
    return 1.0;
  };

  std::function<real_t(const hhg::Point3D&)> vel_y = [](const hhg::Point3D&) {
//    return -std::pow(x[0], 4.0) * x[1] + std::pow(1.0-x[0], 4.0) * (1.0-x[1]);
    return 0.0;
  };

  std::shared_ptr<hhg::PrimitiveStorage> storage = std::make_shared<hhg::PrimitiveStorage>(setupStorage);

  std::shared_ptr<hhg::DGFunction<real_t>> c_old = std::make_shared<hhg::DGFunction<real_t>>("c_old", storage, minLevel, maxLevel);
  std::shared_ptr<hhg::DGFunction<real_t>> c = std::make_shared<hhg::DGFunction<real_t>>("c", storage, minLevel, maxLevel);
  std::shared_ptr<hhg::P1Function<real_t>> u = std::make_shared<hhg::P1Function<real_t>>("u", storage, minLevel, maxLevel);
  std::shared_ptr<hhg::P1Function<real_t>> v = std::make_shared<hhg::P1Function<real_t>>("v", storage, minLevel, maxLevel);

  std::array<std::shared_ptr<hhg::P1Function<real_t>>, 2> velocity{{u,v}};

  hhg::DGUpwindOperator<hhg::P1Function<real_t>> N(storage, velocity, minLevel, maxLevel);

  u->interpolate(vel_x, maxLevel);
  v->interpolate(vel_y, maxLevel);
  c_old->interpolate(initialConcentration,{}, maxLevel);

  hhg::VTKOutput vtkOutput( "../output", "dg0_transport" );

  vtkOutput.add( u );
  vtkOutput.add( v );
  vtkOutput.add( c_old );
  vtkOutput.add( c );

  vtkOutput.write( maxLevel );

  for(uint_t i = 1; i <= timesteps; i++) {
    N.apply(*c_old, *c, maxLevel, hhg::Inner, Replace);
    c->assign({1.0, -dt}, {c_old.get(), c.get()}, maxLevel, hhg::Inner);

    vtkOutput.write( maxLevel, i );

    c_old.swap(c);
  }

  return EXIT_SUCCESS;
}
