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
#include <Eigen/SparseLU>
#include <cmath>
#include <hyteg/primitivestorage/Visualization.hpp>
#include <random>
#include <unordered_map>
#include <utility>

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/Format.hpp"
#include "core/math/Constants.h"
#include "core/math/Random.h"
#include "core/mpi/MPIManager.h"

#include "hyteg/dataexport/VTKOutput.hpp"
#include "hyteg/elementwiseoperators/P1ElementwiseOperator.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearProlongation.hpp"
#include "hyteg/gridtransferoperators/P1toP1LinearRestriction.hpp"
#include "hyteg/p1functionspace/P1ConstantOperator.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p1functionspace/P1VariableOperator.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"
#include "hyteg/primitivestorage/StoragePermutator.hpp"
#include "hyteg/solvers/CGSolver.hpp"
#include "hyteg/solvers/GaussSeidelSmoother.hpp"
#include "hyteg/solvers/GeometricMultigridSolver.hpp"
#include "hyteg/solvers/SORSmoother.hpp"

#include "block_smoother/GSCellSmoother.hpp"
#include "block_smoother/GSEdgeSmoother.hpp"
#include "block_smoother/GSFaceSmoother.hpp"
#include "block_smoother/GSVertexSmoother.hpp"
#include "block_smoother/HybridPrimitiveSmoother.hpp"
#include "block_smoother/P1LDLTInplaceCellSmoother.hpp"
#include "utils/create_domain.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;
using walberla::math::pi;

using namespace hyteg;

template < typename OperatorType, typename FormType >
std::shared_ptr< hyteg::Solver< OperatorType > >
    createSmoother3D( walberla::config::Config::BlockHandle& parameters, OperatorType& op, FormType& form )
{
   const std::string smoother_type = parameters.getParameter< std::string >( "smoother_type" );

   if ( smoother_type == "inplace_ldlt" || smoother_type == "surrogate_ldlt" || smoother_type == "cell_gs" )
   {
      auto eigen_smoother = std::make_shared< hyteg::HybridPrimitiveSmoother< OperatorType > >(
          op.getStorage(), op.getMinLevel(), op.getMaxLevel() );

      // cell ilu
      if ( smoother_type == "inplace_ldlt" )
      {
         auto cell_smoother = std::make_shared< hyteg::P1LDLTInplaceCellSmoother< OperatorType, FormType > >(
             op.getStorage(), op.getMinLevel(), op.getMaxLevel(), form );
         cell_smoother->init();
         eigen_smoother->setCellSmoother( cell_smoother );
      }
      else if ( smoother_type == "surrogate_ldlt" )
      {
         const uint_t opDegreeX     = parameters.getParameter< uint_t >( "op_surrogate_degree_x" );
         const uint_t opDegreeY     = parameters.getParameter< uint_t >( "op_surrogate_degree_y" );
         const uint_t opDegreeZ     = parameters.getParameter< uint_t >( "op_surrogate_degree_z" );
         const uint_t assemblyLevel = parameters.getParameter< uint_t >( "op_surrogate_assembly_level" );
         const bool   symmetry      = parameters.getParameter< bool >( "op_surrogate_use_symmetry" );

         const std::array< uint_t, 3 > opDegrees = { opDegreeX, opDegreeY, opDegreeZ };

         const uint_t iluDegreeX = parameters.getParameter< uint_t >( "ilu_surrogate_degree_x" );
         const uint_t iluDegreeY = parameters.getParameter< uint_t >( "ilu_surrogate_degree_y" );
         const uint_t iluDegreeZ = parameters.getParameter< uint_t >( "ilu_surrogate_degree_z" );
         const uint_t skipLevel  = parameters.getParameter< uint_t >( "ilu_surrogate_skip_level" );

         const std::array< uint_t, 3 > iluDegrees = { iluDegreeX, iluDegreeY, iluDegreeZ };

         // cell ilu
         auto cell_smoother = std::make_shared< hyteg::P1LDLTSurrogateCellSmoother< OperatorType, FormType > >(
             op.getStorage(), op.getMinLevel(), op.getMaxLevel(), opDegrees, iluDegrees, symmetry, form );
         cell_smoother->init( assemblyLevel, skipLevel );
         eigen_smoother->setCellSmoother( cell_smoother );
      }
      else if ( smoother_type == "cell_gs" )
      {
         auto cell_smoother = std::make_shared< hyteg::GSCellSmoother< OperatorType, FormType > >(
             op.getStorage(), op.getMinLevel(), op.getMaxLevel(), form );
         eigen_smoother->setCellSmoother( cell_smoother );
         eigen_smoother->setConsecutiveBackwardsSmoothingStepsOnCells( 1 );
      }
      else
      {
         WALBERLA_ABORT( "unknown smoother type" );
      }

      auto sm_steps_lower_primitives = parameters.getParameter< uint_t >( "sm_steps_lower_primitives" );
      eigen_smoother->setConsecutiveSmoothingStepsOnEdges( sm_steps_lower_primitives );
      eigen_smoother->setConsecutiveSmoothingStepsOnFaces( sm_steps_lower_primitives );
      eigen_smoother->setConsecutiveSmoothingStepsOnVertices( sm_steps_lower_primitives );

      auto sm_steps_lower_primitives_bw = parameters.getParameter< uint_t >( "sm_steps_lower_primitives_backward" );
      eigen_smoother->setConsecutiveBackwardsSmoothingStepsOnEdges( sm_steps_lower_primitives_bw );
      eigen_smoother->setConsecutiveBackwardsSmoothingStepsOnFaces( sm_steps_lower_primitives_bw );
      eigen_smoother->setConsecutiveBackwardsSmoothingStepsOnVertices( sm_steps_lower_primitives_bw );

      auto face_smoother = std::make_shared< hyteg::GSFaceSmoother< OperatorType, FormType > >(
          op.getStorage(), op.getMinLevel(), op.getMaxLevel(), form );
      eigen_smoother->setFaceSmoother( face_smoother );

      auto edge_smoother = std::make_shared< hyteg::GSEdgeSmoother< OperatorType, FormType > >(
          op.getStorage(), op.getMinLevel(), op.getMaxLevel(), form );
      eigen_smoother->setEdgeSmoother( edge_smoother );

      auto vertex_smoother = std::make_shared< hyteg::GSVertexSmoother< OperatorType, FormType > >(
          op.getStorage(), op.getMinLevel(), op.getMaxLevel(), form );
      eigen_smoother->setVertexSmoother( vertex_smoother );

      return eigen_smoother;
   }
   else if ( smoother_type == "gs" )
   {
      return std::make_shared< hyteg::GaussSeidelSmoother< OperatorType > >();
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

   printPermutations();

   const bool        powermethod      = parameters.getParameter< bool >( "powermethod" );
   const uint_t      minLevel         = parameters.getParameter< uint_t >( "minLevel" );
   const uint_t      maxLevel         = parameters.getParameter< uint_t >( "maxLevel" );
   const uint_t      max_outer_iter   = parameters.getParameter< uint_t >( "max_outer_iter" );
   const uint_t      max_coarse_iter  = parameters.getParameter< uint_t >( "max_coarse_iter" );
   const real_t      mg_tolerance     = parameters.getParameter< real_t >( "mg_tolerance" );
   const real_t      coarse_tolerance = parameters.getParameter< real_t >( "coarse_tolerance" );
   const std::string domain           = parameters.getParameter< std::string >( "domain" );

   const uint_t asymptoticConvergenceStartIter = parameters.getParameter< uint_t >( "asymptotic_convergence_start_iter" );

   const uint_t num_smoothing_steps = parameters.getParameter< uint_t >( "num_smoothing_steps" );

   const std::string smoother_type = parameters.getParameter< std::string >( "smoother_type" );
   const real_t      relax         = parameters.getParameter< real_t >( "relax" );

   const std::string solution_type = parameters.getParameter< std::string >( "solution_type" );

   const std::string kappa_type = parameters.getParameter< std::string >( "kappa_type" );

   const auto setupStorage = createDomain( parameters );

   if ( parameters.getParameter< bool >( "auto_permutation" ) )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "applying auto permutation" );
      StoragePermutator permutator;
      permutator.permutate_ilu( *setupStorage );
   }

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

   std::function< real_t( const Point3D& ) > kappa2d = []( const Point3D& p ) { return 1.; };
   std::function< real_t( const Point3D& ) > kappa3d = []( const Point3D& p ) { return 1.; };

   if ( kappa_type == "constant" )
   {
      kappa2d = []( const Point3D& p ) { return 1.; };
      kappa3d = []( const Point3D& p ) { return 1001.; };
   }
   else if ( kappa_type == "linear" )
   {
      kappa2d = []( const Point3D& p ) { return 10. + 10 * p[0] - 5 * p[1]; };
      kappa3d = []( const Point3D& p ) { return 1. + 10 * ( p[0] + p[1] + p[2] ); };
   }
   else if ( kappa_type == "quadratic" )
   {
      kappa2d = []( const Point3D& p ) { return 10. + 10 * std::pow( p[0], 2 ) - 5 * p[1] * p[0]; };
      kappa3d = []( const Point3D& p ) { return 1. + 10 * ( std::pow( p[0], 2 ) + std::pow( p[1], 2 ) + std::pow( p[2], 2 ) ); };
   }
   else if ( kappa_type == "cubic" )
   {
      kappa2d = []( const Point3D& p ) { return 10. + 10 * std::pow( p[0], 2 ) - 5 * p[1] * p[0]; };
      kappa3d = []( const Point3D& p ) { return 1. + 10 * ( std::pow( p[0], 3 ) + std::pow( p[1], 3 ) + std::pow( p[2], 3 ) ); };
   }
   else if ( kappa_type == "unspecified" )
   {
      // DO NOTHING
   }
   else
   {
      WALBERLA_ABORT( "unknown kappa type" );
   }

   if ( solution_type == "sines" && !powermethod && domain != "two_layer_cube" && domain != "two_layer_cube_v2" )
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
   else if ( domain == "two_layer_cube" && !powermethod )
   {
      const real_t kappa_lower = parameters.getParameter< real_t >( "kappa_lower" );
      const real_t kappa_upper = parameters.getParameter< real_t >( "kappa_upper" );
      const real_t height      = parameters.getParameter< real_t >( "tetrahedron_height" );

      const auto height_upper = 1 - height;
      const auto height_lower = height;

      const double z_m = kappa_upper / height_upper / ( kappa_lower / height_lower + kappa_upper / height_upper );

      boundaryConditions = [height_lower, height_upper, z_m]( const hyteg::Point3D& x ) {
         const double z = x[2];
         if ( z < height_lower )
         {
            return z_m / height_lower * z;
         }
         else
         {
            return ( 1 - z_m ) / height_upper * ( z - height_lower ) + z_m;
         }
      };

      rhsFunctional = []( const hyteg::Point3D& x ) { return 0; };
   }
   else if ( domain == "two_layer_cube_v2" && !powermethod )
   {
      const real_t kappa_lower = parameters.getParameter< real_t >( "kappa_lower" );
      const real_t kappa_upper = parameters.getParameter< real_t >( "kappa_upper" );
      const real_t height      = parameters.getParameter< real_t >( "tetrahedron_height" );

      const auto height_upper = 1 + height;
      const auto height_lower = 1.;

      const double z_m = kappa_upper / height_upper / ( kappa_lower / height_lower + kappa_upper / height_upper );

      boundaryConditions = [height_lower, height_upper, z_m]( const hyteg::Point3D& x ) {
         const double z = x[2];
         if ( z < height_lower )
         {
            return z_m / height_lower * z;
         }
         else
         {
            return ( 1 - z_m ) / height_upper * ( z - height_lower ) + z_m;
         }
      };

      rhsFunctional = []( const hyteg::Point3D& x ) { return 0; };
   }
   else if ( solution_type == "linear" && !powermethod )
   {
      boundaryConditions = []( const hyteg::Point3D& p ) { return 2 * p[0] - p[1] + 0.5 * p[2]; };

      kappa3d       = []( const hyteg::Point3D& p ) { return p[0] + 5 * p[1] + 9 * p[2] + 1; };
      rhsFunctional = []( const hyteg::Point3D& ) { return -( 2 * 1 - 1 * 5 + 0.5 * 9 ); };

      // kappa3d = []( const hyteg::Point3D& p ) { return 1.; };
      // rhsFunctional = []( const hyteg::Point3D& ) { return 0.; };
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

   //using OperatorType = hyteg::P1ConstantLaplaceOperator;
   // using OperatorType = hyteg::P1BlendingLaplaceOperator;
   // using FormType = hyteg::forms::p1_diffusion_blending_q1;
   //using FormType = P1FenicsForm< p1_diffusion_cell_integral_0_otherwise, p1_tet_diffusion_cell_integral_0_otherwise >;
   //FormType     form;
   //OperatorType laplaceOperator( storage, minLevel, maxLevel );

   // using OperatorType = hyteg::P1BlendingLaplaceOperator;
   // using OperatorType = hyteg::P1AffineDivkGradOperator;
   using OperatorType = P1ElementwiseBlendingDivKGradOperator;
   using FormType     = forms::p1_div_k_grad_blending_q3;
   //using FormType = forms::p1_div_k_grad_affine_q3;

   if ( domain == "two_layer_cube" )
   {
      const real_t kappa_lower = parameters.getParameter< real_t >( "kappa_lower" );
      const real_t kappa_upper = parameters.getParameter< real_t >( "kappa_upper" );
      const real_t height      = parameters.getParameter< real_t >( "tetrahedron_height" );

      kappa3d = [=]( const Point3D& p ) {
         if ( p[2] < height )
            return kappa_lower;
         else if ( p[2] > height )
            return kappa_upper;
         else
            WALBERLA_ABORT( "not defined" )
      };
   }
   if ( domain == "two_layer_cube_v2" )
   {
      const real_t kappa_lower = parameters.getParameter< real_t >( "kappa_lower" );
      const real_t kappa_upper = parameters.getParameter< real_t >( "kappa_upper" );
      const real_t height      = parameters.getParameter< real_t >( "tetrahedron_height" );

      kappa3d = [=]( const Point3D& p ) {
         if ( p[2] < 1. )
            return kappa_lower;
         else if ( p[2] > 1. )
            return kappa_upper;
         else
            WALBERLA_ABORT( "not defined" );
      };
   }

   FormType     form( kappa3d, kappa2d );
   OperatorType laplaceOperator( storage, minLevel, maxLevel, form );

   std::shared_ptr< hyteg::Solver< OperatorType > > smoother =
       createSmoother3D< OperatorType, FormType >( parameters, laplaceOperator, form );

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
      const real_t h            = 1 / std::pow( 2, maxLevel );
      WALBERLA_LOG_INFO_ON_ROOT( "L2 error: " << discr_l2_err * std::pow( h, 3. / 2. ) );

      asymptoticConvergenceRate /= real_c( outerIter + 1 - asymptoticConvergenceStartIter );
      WALBERLA_LOG_INFO_ON_ROOT( "Asymptotic onvergence rate: " << std::scientific << asymptoticConvergenceRate );
   }

   if ( parameters.getParameter< bool >( "vtkOutput" ) )
   {
      hyteg::VTKOutput vtkOutput( "./output", "ILUSmoother3D", storage );
      vtkOutput.add( u );
      vtkOutput.add( residual );
      vtkOutput.add( error );
      vtkOutput.add( f );
      vtkOutput.add( solution );
      vtkOutput.write( maxLevel, 0 );

      writeDomainPartitioningVTK( storage, "./output", "ILUSmoother3D" );
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
