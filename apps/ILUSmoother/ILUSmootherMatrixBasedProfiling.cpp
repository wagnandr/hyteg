/*
 * Copyright (c) 2022 Andreas Wagner.
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
#include <unordered_map>

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/math/Constants.h"
#include "core/math/Random.h"
#include "core/mpi/MPIManager.h"

#include "hyteg/LikwidWrapper.hpp"
#include "hyteg/elementwiseoperators/P1ElementwiseOperator.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p1functionspace/P1VariableOperator.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/StoragePermutator.hpp"

#include "block_smoother/GSEdgeSmoother.hpp"
#include "block_smoother/P1LDLTInplaceCellSmoother.hpp"
#include "utils/create_domain.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;
using walberla::math::pi;

using namespace hyteg;

int main( int argc, char** argv )
{
   LIKWID_MARKER_INIT;

   walberla::Environment env( argc, argv );
   walberla::mpi::MPIManager::instance()->useWorldComm();

   auto cfg = std::make_shared< walberla::config::Config >();
   if ( env.config() == nullptr )
   {
      cfg->readParameterFile( "./ILUSmootherProfiling.prm" );
   }
   else
   {
      cfg = env.config();
   }
   walberla::Config::BlockHandle parameters = cfg->getOneBlock( "Parameters" );
   parameters.listParameters();

   const uint_t level                = parameters.getParameter< uint_t >( "level" );
   const uint_t numberOfIterations   = parameters.getParameter< uint_t >( "number_of_iterations" );
   const uint_t numSubdivision       = parameters.getParameter< uint_t >( "number_of_subdivisions" );

   hyteg::MeshInfo meshInfo =
       hyteg::MeshInfo::meshCuboid( hyteg::Point3D( { 0, 0, 0 } ), hyteg::Point3D( { 1, 1, 1 } ), numSubdivision, numSubdivision, numSubdivision );
   auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
       meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
   WALBERLA_LOG_INFO_ON_ROOT( "num processors " << walberla::mpi::MPIManager::instance()->numProcesses() );
   setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );
   const auto storage = std::make_shared< PrimitiveStorage >( *setupStorage );

   WALBERLA_LOG_INFO( "num cells (global)" << storage->getNumberOfGlobalCells() );
   WALBERLA_LOG_INFO( "num cells (local)" << storage->getNumberOfLocalCells() );

   using FormType     = forms::p1_div_k_grad_blending_q3;
   using OperatorType = P1ElementwiseBlendingDivKGradOperator;

   std::function< real_t( const Point3D& ) > kappa2d = []( const Point3D& ) { return 1.; };
   std::function< real_t( const Point3D& ) > kappa3d = []( const Point3D& ) { return 1.; };

   FormType form( kappa3d, kappa2d );
   FormType form_const( []( auto ) { return 1.; }, []( auto ) { return 1.; } );

   hyteg::P1Function< real_t > src1( "src1", storage, level, level );
   hyteg::P1Function< real_t > src2( "src2", storage, level, level );
   src1.interpolate( []( const Point3D& p ) { return p[0] + -0.5 * p[1] + 2 * p[2]; }, level, All );
   src2.interpolate( []( const Point3D& p ) { return p[0] + -0.5 * p[1] + 2 * p[2]; }, level, All );
   hyteg::P1Function< real_t > dst( "dst", storage, level, level );
   dst.interpolate( 0, level, All );

   hyteg::P1Function< real_t > tmp( "tmp", storage, level, level );

   std::vector< std::array< real_t, 7 > > stencils_l{};
   for ( uint_t i = 0; i < levelinfo::num_microvertices_per_cell( level ); i += 1 )
   {
      stencils_l.push_back( { walberla::math::realRandom( 0., 1. ),
                              walberla::math::realRandom( 0., 1. ),
                              walberla::math::realRandom( 0., 1. ),
                              walberla::math::realRandom( 0., 1. ),
                              walberla::math::realRandom( 0., 1. ),
                              walberla::math::realRandom( 0., 1. ) } );
   }

   std::vector< std::array< real_t, 7 > > stencils_lt{};
   for ( uint_t i = 0; i < levelinfo::num_microvertices_per_cell( level ); i += 1 )
   {
      stencils_lt.push_back( { walberla::math::realRandom( 0., 1. ),
                               walberla::math::realRandom( 0., 1. ),
                               walberla::math::realRandom( 0., 1. ),
                               walberla::math::realRandom( 0., 1. ),
                               walberla::math::realRandom( 0., 1. ),
                               walberla::math::realRandom( 0., 1. ) } );
   }

   std::vector< std::array< real_t, 1 > > stencils_d{};
   for ( uint_t i = 0; i < levelinfo::num_microvertices_per_cell( level ); i += 1 )
   {
      stencils_d.push_back( { walberla::math::realRandom( 0., 1. ) } );
   }

   for ( uint_t i = 0; i < numberOfIterations; i += 1 )
   {
      for ( auto cit : storage->getCells() )
      {
         Cell& cell = *cit.second;

         ldlt::p1::dim3::ConstantStencilNew opStencilProviderNew( level, cell, form, ldlt::p1::dim3::allDirections );

         ldlt::p1::dim3::apply_full_surrogate_ilu_smoothing_step_matrix< hyteg::P1Function< real_t >,
                                                                         ldlt::p1::dim3::ConstantStencilNew< FormType, 15 > >(
             opStencilProviderNew, stencils_l, stencils_lt, stencils_d, level, cell, src2, tmp, dst );

         if ( cell.getData( src2.getCellDataID() )->getPointer( level )[0] > 1000000 )
            WALBERLA_LOG_INFO_ON_ROOT( "op3 " << src2.getMaxMagnitude( level, All, true ) );
      }

      for ( auto cit : storage->getCells() )
      {
         Cell& cell = *cit.second;

         ldlt::p1::dim3::ConstantStencilNew opStencilProviderNew( level, cell, form, ldlt::p1::dim3::allDirections );

         ldlt::p1::dim3::apply_full_surrogate_ilu_smoothing_step_constant_matrix< hyteg::P1Function< real_t >,
             ldlt::p1::dim3::ConstantStencilNew< FormType, 15 > >(
             opStencilProviderNew, stencils_l.front(), stencils_lt.front(), stencils_d.front(), level, cell, src2, tmp, dst );

         if ( cell.getData( src2.getCellDataID() )->getPointer( level )[0] > 1000000 )
         WALBERLA_LOG_INFO_ON_ROOT( "op3 " << src2.getMaxMagnitude( level, All, true ) );
      }
   }
   LIKWID_MARKER_CLOSE;
}
