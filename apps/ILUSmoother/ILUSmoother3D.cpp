/*
 * Copyright (c) 2020 Andreas Wagner, Daniel Drzisga
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
#include <Eigen/SparseLU>
#include <cmath>
#include <random>
#include <unordered_map>
#include <utility>

#include "core/Format.hpp"
#include "core/math/Random.h"
#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/math/Constants.h"
#include "core/mpi/MPIManager.h"

#include "hyteg/dataexport/VTKOutput.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearProlongation.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearRestriction.hpp"
#include "hyteg/p1functionspace/P1ConstantOperator.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p1functionspace/P1VariableOperator.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/solvers/CGSolver.hpp"
#include "hyteg/solvers/SORSmoother.hpp"
#include "hyteg/solvers/GeometricMultigridSolver.hpp"

#include "block_smoother/HybridPrimitiveSmoother.hpp"
#include "block_smoother/P1LDLTInplaceCellSmoother.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;
using walberla::math::pi;

using namespace hyteg;

std::shared_ptr< SetupPrimitiveStorage > createDomain( walberla::Config::BlockHandle& parameters )
{
   const std::string domain = parameters.getParameter< std::string >( "domain" );

   if ( domain == "tetrahedron" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      const double top_x = 0.0;
      const double top_y = 0.0;
      const double top_z = 0.1;

      const Point3D p0( { 0, 0, 0 } );
      const Point3D p1( { 1.0, 0, 0 } );
      const Point3D p2( { 0.0, 1.0, 0 } );
      const Point3D p3( { top_x, top_y, top_z } );

      // we permutate the vertices to study performance for different orientations:
      const uint_t permutationNumber = parameters.getParameter< uint_t >( "tet_permutation" );
      std::vector< uint_t > order { 0, 1, 2, 3 };
      for ( uint_t i = 0; i < permutationNumber; ++i )
         std::next_permutation( std::begin( order ), std::end( order ) );

      std::array< Point3D, 4 > vertices;
      vertices[order[0]] = p0;
      vertices[order[1]] = p1;
      vertices[order[2]] = p2;
      vertices[order[3]] = p3;

      MeshInfo meshInfo = MeshInfo::singleTetrahedron( vertices );

      auto setupStorage =
          std::make_shared< SetupPrimitiveStorage >( meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );

      return setupStorage;
   }
   else
   {
      WALBERLA_ABORT( "unknown domain" );
   }
}

template < typename OperatorType, typename FormType >
std::shared_ptr< hyteg::Solver< OperatorType > > createSmoother3D( walberla::config::Config::BlockHandle& parameters,
                                                                   OperatorType&                          op )
{
   const std::string smoother_type = parameters.getParameter< std::string >( "smoother_type" );

   if ( smoother_type == "inplace_ldlt" )
   {
      auto eigen_smoother = std::make_shared< hyteg::HybridPrimitiveSmoother< OperatorType > >(
          op.getStorage(), op.getMinLevel(), op.getMaxLevel() );

      // cell ilu
      P1Function< int > numerator( "numerator", op.getStorage(), op.getMinLevel(), op.getMaxLevel() );
      for ( size_t level = op.getMinLevel(); level <= op.getMaxLevel(); level += 1 )
         numerator.enumerate( level );
      auto cell_smoother = std::make_shared< hyteg::P1LDLTInplaceCellSmoother< OperatorType, FormType > >(
          op.getStorage(), op.getMinLevel(), op.getMaxLevel() );
      cell_smoother->init( op );
      eigen_smoother->setCellSmoother( cell_smoother );

      return eigen_smoother;
   }
   WALBERLA_ABORT( "smoother type " + smoother_type + " is unknown." );
}

int main( int argc, char** argv )
{
   walberla::Environment env( argc, argv );
   walberla::mpi::MPIManager::instance()->useWorldComm();

   auto cfg = std::make_shared< walberla::config::Config >();
   if ( env.config() == nullptr )
   {
      cfg->readParameterFile( "./ILUSmoother3D.prm" );
   }
   else
   {
      cfg = env.config();
   }
   walberla::Config::BlockHandle parameters = cfg->getOneBlock( "Parameters" );
   parameters.listParameters();

   const bool   powermethod      = parameters.getParameter< bool >( "powermethod" );
   const uint_t minLevel         = parameters.getParameter< uint_t >( "minLevel" );
   const uint_t maxLevel         = parameters.getParameter< uint_t >( "maxLevel" );
   const uint_t max_outer_iter   = parameters.getParameter< uint_t >( "max_outer_iter" );
   const uint_t max_coarse_iter  = parameters.getParameter< uint_t >( "max_coarse_iter" );
   const real_t mg_tolerance     = parameters.getParameter< real_t >( "mg_tolerance" );
   const real_t coarse_tolerance = parameters.getParameter< real_t >( "coarse_tolerance" );

   const uint_t asymptoticConvergenceStartIter = parameters.getParameter< uint_t >( "asymptotic_convergence_start_iter" );

   const uint_t num_smoothing_steps = parameters.getParameter< uint_t >( "num_smoothing_steps" );

   const std::string smoother_type = parameters.getParameter< std::string >( "smoother_type" );
   const real_t      relax         = parameters.getParameter< real_t >( "relax" );

   const std::string solution_type = parameters.getParameter< std::string >( "solution_type" );

   const auto setupStorage = createDomain( parameters );

   const auto storage = std::make_shared< PrimitiveStorage >( *setupStorage );

   hyteg::P1Function< real_t > residual( "residual", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > error( "error", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > f( "f", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > u( "u", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > u_tmp( "u", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > solution( "solution", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > laplaceTimesFunction( "laplaceTimesFunction", storage, minLevel, maxLevel );
   hyteg::P1Function< real_t > tmp( "tmp", storage, minLevel, maxLevel );

   std::function< real_t( const hyteg::Point3D& ) > boundaryConditions;
   std::function< real_t( const hyteg::Point3D& ) > rhsFunctional;

   if ( solution_type == "sines" && !powermethod )
   {
      constexpr uint_t k_x = 4;
      constexpr uint_t k_y = 2;

      boundaryConditions = []( const hyteg::Point3D& x ) {
         return std::sin( 2 * k_x * M_PI * x[0] ) * std::cos( 2 * k_y * M_PI * x[1] );
      };

      rhsFunctional = []( const hyteg::Point3D& x ) {
         return ( k_x * k_x + k_y * k_y ) * 4 * M_PI * M_PI * std::sin( 2 * k_x * M_PI * x[0] ) *
                std::cos( 2 * k_y * M_PI * x[1] );
      };
   }
   else if ( solution_type == "zero" || powermethod )
   {
      boundaryConditions = []( const hyteg::Point3D& ) { return 0; };

      rhsFunctional = []( const hyteg::Point3D& ) { return 0; };
   }
   else
   {
      WALBERLA_ABORT( "unknown solution type" );
   }

   u.interpolate( boundaryConditions, maxLevel, hyteg::DirichletBoundary );
   solution.interpolate( boundaryConditions, maxLevel, hyteg::All );

   tmp.interpolate( rhsFunctional, maxLevel, hyteg::All );
   hyteg::P1BlendingMassOperator M( storage, minLevel, maxLevel );
   M.apply( tmp, f, maxLevel, hyteg::All );

   if ( parameters.getParameter< bool >( "randomInitialGuess" ) || powermethod )
   {
      u.interpolate( []( const Point3D& ) -> real_t { return walberla::math::realRandom( real_t( -10.0 ), real_t( 10.0 ) ); },
                     maxLevel,
                     All ^ DirichletBoundary );
   }

   // using OperatorType = hyteg::P1ElementwiseLaplaceOperator;
   using OperatorType = hyteg::P1ConstantLaplaceOperator;
   using FormType = hyteg::forms::p1_diffusion_blending_q1;

   OperatorType laplaceOperator( storage, minLevel, maxLevel );

   std::shared_ptr< hyteg::Solver< OperatorType > > smoother = createSmoother3D< OperatorType, FormType > ( parameters, laplaceOperator );

   auto coarseGridSolver =
       std::make_shared< hyteg::CGSolver< OperatorType > >( storage, minLevel, minLevel, max_coarse_iter, coarse_tolerance );
   auto restrictionOperator  = std::make_shared< hyteg::P1toP1LinearRestriction >();
   auto prolongationOperator = std::make_shared< hyteg::P1toP1LinearProlongation >();

   auto multiGridSolver = std::make_shared< hyteg::GeometricMultigridSolver< OperatorType > >(
       storage, smoother, coarseGridSolver, restrictionOperator, prolongationOperator, minLevel, maxLevel );
   multiGridSolver->setSmoothingSteps( num_smoothing_steps, num_smoothing_steps );

   auto fineGridSolver = std::make_shared< hyteg::CGSolver< OperatorType > >(
       storage, minLevel, maxLevel, max_outer_iter, mg_tolerance, multiGridSolver );

   WALBERLA_LOG_INFO_ON_ROOT( "Starting V cycles" );
   WALBERLA_LOG_INFO_ON_ROOT( walberla::format( "%6s|%10s|%10s|%10s", "iter", "abs_res", "rel_res", "conv" ) );

   std::vector< real_t > list_abs_res;
   std::vector< real_t > list_rel_res;
   std::vector< real_t > list_res_rate;

   laplaceOperator.apply( u, laplaceTimesFunction, maxLevel, hyteg::Inner );
   residual.assign( { 1.0, -1.0 }, { f, laplaceTimesFunction }, maxLevel, hyteg::Inner );
   real_t       last_abs_res  = std::sqrt( residual.dotGlobal( residual, maxLevel, hyteg::Inner ) );
   const real_t begin_abs_res = last_abs_res;

   WALBERLA_LOG_INFO_ON_ROOT( walberla::format( "%6d|%10.2e|%10.2e|%10.2e", 0, begin_abs_res, 1, std::nan( "" ) ) );

   list_abs_res.push_back( begin_abs_res );
   list_rel_res.push_back( 1 );
   list_res_rate.push_back( std::nan( "" ) );

   real_t asymptoticConvergenceRate = 0;

   double eigenvalue;

   if ( parameters.getParameter< bool >( "usePCG" ) )
   {
      fineGridSolver->setPrintInfo( true );
      fineGridSolver->solve( laplaceOperator, u, f, maxLevel );
   }
   else
   {
      uint_t outerIter = 0;
      for ( outerIter = 0; outerIter < max_outer_iter; ++outerIter )
      {
         if ( powermethod )
         {
            u_tmp.assign( { 1.0 }, { u }, maxLevel, All );
         }

         multiGridSolver->solve( laplaceOperator, u, f, maxLevel );

         if ( powermethod )
         {
            real_t new_ = u.dotGlobal( u_tmp, maxLevel, Inner | NeumannBoundary );
            real_t old_ = u_tmp.dotGlobal( u_tmp, maxLevel, Inner | NeumannBoundary );

            eigenvalue = new_ / old_;

            std::cout << "eigenvalue = " << eigenvalue << std::endl;

            real_t uNorm = std::sqrt( u.dotGlobal( u, maxLevel, Inner | NeumannBoundary ) );
            u.assign( { 1.0 / uNorm }, { u }, maxLevel, All );
         }
         else
         {
            laplaceOperator.apply( u, laplaceTimesFunction, maxLevel, hyteg::Inner );
            residual.assign( { 1.0, -1.0 }, { f, laplaceTimesFunction }, maxLevel, hyteg::Inner );
            const real_t abs_res  = std::sqrt( residual.dotGlobal( residual, maxLevel, hyteg::Inner ) );
            const real_t rel_res  = abs_res / begin_abs_res;
            const real_t res_rate = abs_res / last_abs_res;
            list_abs_res.push_back( abs_res );
            list_rel_res.push_back( rel_res );
            list_res_rate.push_back( res_rate );

            WALBERLA_LOG_INFO_ON_ROOT(
                walberla::format( "%6d|%10.2e|%10.2e|%10.2e", outerIter + 1, abs_res, rel_res, res_rate ) );

            last_abs_res = abs_res;

            if ( outerIter >= asymptoticConvergenceStartIter )
            {
               asymptoticConvergenceRate += res_rate;
            }

            if ( res_rate >= 1.0 )
            {
               asymptoticConvergenceRate = 1e300;
               break;
            }

            if ( rel_res < mg_tolerance )
               break;
         }
      }

      if ( powermethod )
      {
         WALBERLA_LOG_INFO_ON_ROOT( "Final eigenvalue: " << eigenvalue );
      }

      error.assign( { 1.0, -1.0 }, { u, solution }, maxLevel, hyteg::Inner );
      const real_t discr_l2_err = std::sqrt( error.dotGlobal( error, maxLevel, hyteg::Inner ) );
      WALBERLA_LOG_INFO_ON_ROOT( "L2 error: " << discr_l2_err );

      asymptoticConvergenceRate /= real_c( outerIter + 1 - asymptoticConvergenceStartIter );
      WALBERLA_LOG_INFO_ON_ROOT( "Asymptotic onvergence rate: " << std::scientific << asymptoticConvergenceRate );
   }

   if ( parameters.getParameter< bool >( "vtkOutput" ) )
   {
      hyteg::VTKOutput vtkOutput( "./output", "ILUSmoother3D", storage );
      vtkOutput.add( u );
      vtkOutput.add( residual );
      vtkOutput.add( f );
      vtkOutput.add( solution );
      vtkOutput.write( maxLevel, 0 );
   }

   if ( parameters.getParameter< bool >( "csvOutput" ) )
   {
      std::stringstream filename;
      filename << "csv/data_" << maxLevel << "_";
      filename << smoother_type << "_";
      filename << walberla::format( "%.2e", relax ) << "_";
      const std::string domain = parameters.getParameter< std::string >( "domain" );
      filename << domain.c_str();
      filename << ".csv";

      std::ofstream ofs( filename.str() );

      ofs << walberla::format( "%6s,%25s,%25s,%25s", "iter", "abs_res", "rel_res", "res_rate" );
      ofs << std::endl;

      for ( uint_t i = 0; i < list_abs_res.size(); i += 1 )
      {
         ofs << walberla::format(
             "%6d,%25.16e,%25.16e,%25.16e", i, list_abs_res.at( i ), list_rel_res.at( i ), list_res_rate.at( i ) );
         ofs << std::endl;
      }
   }
}
