#include <core/timing/Timer.h>
#include <tinyhhg_core/tinyhhg.hpp>

using walberla::real_t;

int main(int argc, char* argv[])
{
  walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
  walberla::MPIManager::instance()->useWorldComm();
  WALBERLA_LOG_INFO_ON_ROOT("TinyHHG CG Test\n");

  std::string meshFileName = "../data/meshes/quad_4el.msh";

  hhg::MeshInfo meshInfo = hhg::MeshInfo::fromGmshFile( meshFileName );
  hhg::SetupPrimitiveStorage setupStorage( meshInfo, walberla::uint_c ( walberla::mpi::MPIManager::instance()->numProcesses() ) );

  hhg::RoundRobin loadbalancer;
  setupStorage.balanceLoad( loadbalancer, 0.0 );

  size_t minLevel = 2;
  size_t maxLevel = 2;
  size_t maxiter = 10000;

  std::shared_ptr<hhg::PrimitiveStorage> storage = std::make_shared<hhg::PrimitiveStorage>(setupStorage);

  hhg::P1StokesFunction r("r", storage, minLevel, maxLevel);
  hhg::P1StokesFunction f("f", storage, minLevel, maxLevel);
  hhg::P1StokesFunction u("u", storage, minLevel, maxLevel);
  hhg::P1StokesFunction u_exact("u_exact", storage, minLevel, maxLevel);
  hhg::P1StokesFunction err("err", storage, minLevel, maxLevel);
  hhg::P1Function npoints_helper("npoints_helper", storage, minLevel, maxLevel);

  hhg::P1BlockLaplaceOperator L(storage, minLevel, maxLevel);

  std::function<real_t(const hhg::Point3D&)> exact = [](const hhg::Point3D& xx) { return xx[0]*xx[0] - xx[1]*xx[1]; };
  std::function<real_t(const hhg::Point3D&)> rhs   = [](const hhg::Point3D&) { return 0.0; };
  std::function<real_t(const hhg::Point3D&)> ones  = [](const hhg::Point3D&) { return 1.0; };

  u.u.interpolate(exact, maxLevel, hhg::DirichletBoundary);
  u.v.interpolate(exact, maxLevel, hhg::DirichletBoundary);
  u.p.interpolate(exact, maxLevel, hhg::DirichletBoundary);

  u_exact.u.interpolate(exact, maxLevel);
  u_exact.v.interpolate(exact, maxLevel);
  u_exact.p.interpolate(exact, maxLevel);

  auto solver = hhg::CGSolver<hhg::P1StokesFunction, hhg::P1BlockLaplaceOperator>(storage, minLevel, maxLevel);
  walberla::WcTimer timer;
  solver.solve(L, u, f, r, maxLevel, 1e-8, maxiter, hhg::Inner, true);
  timer.end();
  fmt::printf("time was: %e\n",timer.last());
  err.assign({1.0, -1.0}, {&u, &u_exact}, maxLevel);

  npoints_helper.interpolate(ones, maxLevel);

  real_t npoints = npoints_helper.dot(npoints_helper, maxLevel);

  real_t discr_l2_err = std::sqrt(err.dot(err, maxLevel) / npoints);

  WALBERLA_LOG_INFO_ON_ROOT("discrete L2 error = " << discr_l2_err);

//  hhg::VTKWriter({ &u, &u_exact, &f, &r, &err }, maxLevel, "../output", "test");
  return 0;
}
