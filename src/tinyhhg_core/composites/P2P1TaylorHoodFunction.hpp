#pragma once

#include "tinyhhg_core/p1functionspace/P1Function.hpp"
#include "tinyhhg_core/p2functionspace/P2Function.hpp"
#include "tinyhhg_core/FunctionTraits.hpp"

namespace hhg
{

template <typename ValueType>
class P2P1TaylorHoodFunction
{
public:

  typedef P2Function< ValueType > VelocityFunction_T;
  typedef P1Function< ValueType > PressureFunction_T;
  typedef typename FunctionTrait< P2P1TaylorHoodFunction< ValueType > >::Tag Tag;

  P2P1TaylorHoodFunction(const std::string& _name, const std::shared_ptr< PrimitiveStorage > & storage, size_t minLevel, size_t maxLevel)
    : u(_name+"_u", storage, minLevel, maxLevel),
      v(_name+"_v", storage, minLevel, maxLevel),
      w( storage->hasGlobalCells() ? P2Function< ValueType >( _name+"_w", storage, minLevel, maxLevel ) :  P2Function< ValueType >( _name+"_w_dummy", storage )),
      p(_name+"_p", storage, minLevel, maxLevel, BoundaryCondition::createAllInnerBC() )
  {}

  P2P1TaylorHoodFunction(const std::string& _name, const std::shared_ptr< PrimitiveStorage > & storage, size_t minLevel, size_t maxLevel, BoundaryCondition velocityBC)
  : u(_name+"_u", storage, minLevel, maxLevel, velocityBC),
    v(_name+"_v", storage, minLevel, maxLevel, velocityBC),
    w( storage->hasGlobalCells() ? P2Function< ValueType >( _name+"_w", storage, minLevel, maxLevel ) :  P2Function< ValueType >( _name+"_w_dummy", storage )),
    p(_name+"_p", storage, minLevel, maxLevel, BoundaryCondition::createAllInnerBC() )
  {}

  void interpolate(std::function<real_t(const hhg::Point3D&)>& expr, size_t level, DoFType flag = All)
  {
    u.interpolate(expr, level, flag);
    v.interpolate(expr, level, flag);
    w.interpolate(expr, level, flag);
    p.interpolate(expr, level, flag);
  }

  void assign(const std::vector<walberla::real_t> scalars, const std::vector<P2P1TaylorHoodFunction<ValueType>*> functions, size_t level, DoFType flag = All)
  {
    std::vector< VelocityFunction_T * > functions_u;
    std::vector< VelocityFunction_T * > functions_v;
    std::vector< VelocityFunction_T * > functions_w;
    std::vector< PressureFunction_T * > functions_p;

    for (auto& function : functions)
    {
      functions_u.push_back(&function->u);
      functions_v.push_back(&function->v);
      functions_w.push_back(&function->w);
      functions_p.push_back(&function->p);
    }

    u.assign(scalars, functions_u, level, flag);
    v.assign(scalars, functions_v, level, flag);
    w.assign(scalars, functions_w, level, flag);
    p.assign(scalars, functions_p, level, flag);
  }

  void add(const std::vector<walberla::real_t> scalars, const std::vector<P2P1TaylorHoodFunction<ValueType>*> functions, size_t level, DoFType flag = All)
  {
    std::vector< VelocityFunction_T * > functions_u;
    std::vector< VelocityFunction_T * > functions_v;
    std::vector< VelocityFunction_T * > functions_w;
    std::vector< PressureFunction_T * > functions_p;

    for (auto& function : functions)
    {
      functions_u.push_back(&function->u);
      functions_v.push_back(&function->v);
      functions_w.push_back(&function->w);
      functions_p.push_back(&function->p);
    }

    u.add(scalars, functions_u, level, flag);
    v.add(scalars, functions_v, level, flag);
    w.add(scalars, functions_w, level, flag);
    p.add(scalars, functions_p, level, flag);
  }

  walberla::real_t dotGlobal(P2P1TaylorHoodFunction<ValueType>& rhs, size_t level, DoFType flag = All)
  {
    walberla::real_t sum = u.dotLocal(rhs.u, level, flag);
    sum += v.dotLocal(rhs.v, level, flag);
    sum += w.dotLocal(rhs.w, level, flag);
    sum += p.dotLocal(rhs.p, level, flag | DirichletBoundary);
    walberla::mpi::allReduceInplace( sum, walberla::mpi::SUM, walberla::mpi::MPIManager::instance()->comm() );
    return sum;
  }

  void prolongate(size_t level, DoFType flag = All)
  {
    u.prolongate(level, flag);
    v.prolongate(level, flag);
    w.prolongate(level, flag);
    p.prolongate(level, flag);
  }

  void restrict(size_t level, DoFType flag = All)
  {
    u.restrict(level, flag);
    v.restrict(level, flag);
    w.restrict(level, flag);
    p.restrict(level, flag);
  }

  void enableTiming( const std::shared_ptr< walberla::WcTimingTree > & timingTree )
  {
    u.enableTiming(timingTree);
    v.enableTiming(timingTree);
    w.enableTiming(timingTree);
    p.enableTiming(timingTree);
  }

  void enumerate( uint_t level )
  {
    u.enumerate( level );
    v.enumerate( level );
    w.enumerate( level );
    p.enumerate( level );
  }


  VelocityFunction_T u;
  VelocityFunction_T v;
  VelocityFunction_T w;
  PressureFunction_T p;
};

}
