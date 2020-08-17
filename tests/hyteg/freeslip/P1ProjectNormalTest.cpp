/*
 * Copyright (c) 2020 Daniel Drzisga.
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
#include "hyteg/p2functionspace/P2ProjectNormalOperator.hpp"

#include "core/Environment.h"
#include "core/logging/Logging.h"
#include "core/math/Random.h"

#include "hyteg/dataexport/VTKOutput.hpp"
#include "hyteg/geometry/AnnulusMap.hpp"
#include "hyteg/geometry/IcosahedralShellMap.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p1functionspace/P1ProjectNormalOperator.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/Visualization.hpp"

using walberla::real_c;
using walberla::real_t;

using namespace hyteg;


template < typename StokesFunctionType, typename ProjectNormalOperatorType >
static void testProjectNormal2D( )
{
   const int level = 3;

   const auto  meshInfo = MeshInfo::meshAnnulus(0.5, 1.0, MeshInfo::CRISS, 6, 6);
   SetupPrimitiveStorage setupStorage( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   setupStorage.setMeshBoundaryFlagsOnBoundary( 3, 0, true );
   AnnulusMap::setMap( setupStorage );
   const auto storage = std::make_shared< PrimitiveStorage >( setupStorage );

   auto normalInterpolant = [] ( const Point3D & p ) {
     real_t norm = p.norm();
     real_t sign = (norm > 0.75) ? 1.0 : -1.0;
     return sign/norm * p;
   };

   auto normalFunction = [=]( const Point3D& p, Point3D& n ) -> void {
      n = normalInterpolant( p );
   };

   ProjectNormalOperatorType projectNormalOperator( storage, level, level, normalFunction );

   StokesFunctionType u( "u", storage, level, level );

   // we check if a radial function gets set to zero on the free slip boundary
   u.u.interpolate( [=](auto & p){ return normalInterpolant(p)[0]; }, level );
   u.v.interpolate( [=](auto & p){ return normalInterpolant(p)[1]; }, level );
   WALBERLA_CHECK_GREATER( u.dotGlobal(u, level, FreeslipBoundary), 1 );
   projectNormalOperator.apply( u, level, FreeslipBoundary );
   WALBERLA_CHECK_LESS( u.dotGlobal(u, level, FreeslipBoundary), 1e-14 );

   // we check if a tangential function is not changed by the projection operator
   StokesFunctionType uTan( "uTan", storage, level, level );
   uTan.u.interpolate( [=](auto & p){ return -p[1]; }, level );
   uTan.v.interpolate( [=](auto & p){ return p[0]; }, level );
   u.assign({1}, {uTan}, level, All);
   WALBERLA_CHECK_GREATER( u.dotGlobal(u, level, FreeslipBoundary), 1 );
   projectNormalOperator.apply( u, level, FreeslipBoundary );
   StokesFunctionType diff( "diff", storage, level, level );
   diff.assign( {1, -1}, {u, uTan}, level, All );
   WALBERLA_CHECK_LESS( diff.dotGlobal(diff, level, All), 1e-14 );
}

static void testProjectNormal3D( )
{
   const bool   writeVTK   = true;
   const real_t errorLimit = 1e-13;
   const int level = 3;

   auto meshInfo = MeshInfo::meshSphericalShell( 5, 2, 0.5, 1.0 );
   SetupPrimitiveStorage setupStorage( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   setupStorage.setMeshBoundaryFlagsOnBoundary( 3, 0, true );
   IcosahedralShellMap::setMap( setupStorage );
   const auto storage = std::make_shared< PrimitiveStorage >( setupStorage );

   if ( writeVTK )
      writeDomainPartitioningVTK( storage, "../../output", "P1ProjectNormalTest3D_Domain" );

   auto normal_function = []( const Point3D& p, Point3D& n ) -> void {
     real_t norm = p.norm();
     real_t sign = (norm > 0.75) ? 1.0 : -1.0;

     n = sign/norm * p;
   };

   P1ProjectNormalOperator projectNormalOperator( storage, level, level, normal_function );

   P1StokesFunction< real_t > u( "u", storage, level, level );

   VTKOutput vtkOutput( "../../output", "P1ProjectNormalTest3D", storage );
   vtkOutput.add( u );

   u.interpolate( 1, level );
   projectNormalOperator.apply( u, level, FreeslipBoundary );

   if ( writeVTK )
      vtkOutput.write( level, 0 );
}

int main( int argc, char* argv[] )
{
   walberla::Environment walberlaEnv( argc, argv );
   walberla::logging::Logging::instance()->setLogLevel( walberla::logging::Logging::PROGRESS );
   walberla::MPIManager::instance()->useWorldComm();

   WALBERLA_LOG_INFO_ON_ROOT("normal projection P1-P1 in 2D");
   testProjectNormal2D< P1StokesFunction< real_t >, P1ProjectNormalOperator >( );
   WALBERLA_LOG_INFO_ON_ROOT("normal projection P2-P1-TH in 2D");
   testProjectNormal2D< P2P1TaylorHoodFunction< real_t >, P2ProjectNormalOperator >( );

   return 0;
}
