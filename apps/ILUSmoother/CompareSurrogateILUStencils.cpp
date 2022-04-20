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
#include <hyteg/p1functionspace/P1ConstantOperator.hpp>
#include <random>
#include <unordered_map>

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/mpi/MPIManager.h"

#include "hyteg/dataexport/VTKOutput.hpp"
#include "hyteg/p1functionspace/P1Function.hpp"
#include "hyteg/p1functionspace/P1VariableOperator.hpp"
#include "hyteg/primitivestorage/PrimitiveStorage.hpp"
#include "hyteg/elementwiseoperators/P1ElementwiseOperator.hpp"

#include "block_smoother/P1LDLTInplaceCellSmoother.hpp"
#include "utils/create_domain.hpp"

int main( int argc, char** argv )
{
   walberla::Environment env( argc, argv );
   walberla::mpi::MPIManager::instance()->useWorldComm();

   auto cfg = std::make_shared< walberla::config::Config >();
   if ( env.config() == nullptr )
   {
      cfg->readParameterFile( "./CompareSurrogateILUStencils.prm" );
   }
   else
   {
      cfg = env.config();
   }
   walberla::Config::BlockHandle parameters = cfg->getOneBlock( "Parameters" );
   parameters.listParameters();

   const uint_t level = parameters.getParameter< uint_t >( "level" );

   const uint_t degree    = parameters.getParameter< uint_t >( "degree" );
   const uint_t skipLevel = parameters.getParameter< uint_t >( "skip_level" );

   const std::string filename = parameters.getParameter< std::string >( "filename" );
   const bool        writeVtk = parameters.getParameter< bool >( "vtk_output" );

   auto setupStorage = createDomain( parameters );

   const auto storage = std::make_shared< hyteg::PrimitiveStorage >( *setupStorage );

   using OperatorType = hyteg::P1ElementwiseBlendingDivKGradOperator;
   using FormType     = hyteg::forms::p1_div_k_grad_blending_q3;
   FormType     form ([](auto& ){ return 1.; }, [](auto& ){ return 1.; });
   OperatorType laplaceOperator( storage, level, level, form );

   const std::array< uint_t, 3 >                                      degrees = { degree, degree, degree };
   hyteg::P1LDLTSurrogateCellSmoother< OperatorType, FormType, true > surrogateSmoother(
       storage, level, level, degrees, degrees, true, form );
   hyteg::P1LDLTInplaceCellSmoother< OperatorType, FormType > inplaceSmoother( storage, level, level, form );

   surrogateSmoother.init( skipLevel, skipLevel );
   inplaceSmoother.init();

   std::vector< hyteg::P1Function< real_t > > u_surr;
   std::vector< hyteg::P1Function< real_t > > u_inpl;
   std::vector< hyteg::P1Function< real_t > > u_error;
   for ( auto d : hyteg::ldlt::p1::dim3::lowerDirectionsAndCenter )
   {
      u_surr.emplace_back( "u_" + hyteg::stencilDirectionToStr[d], storage, level, level );
      u_inpl.emplace_back( "u_" + hyteg::stencilDirectionToStr[d], storage, level, level );
      u_error.emplace_back( "u_" + hyteg::stencilDirectionToStr[d], storage, level, level );

      surrogateSmoother.interpolate_stencil_direction( level, d, u_surr.back() );
      inplaceSmoother.interpolate_stencil_direction( level, d, u_inpl.back() );
      u_error.back().assign( { +1, -1 }, { u_inpl.back(), u_surr.back() }, level );

      real_t error_value = std::sqrt( u_error.back().dotGlobal( u_error.back(), level, hyteg::All ) );

      u_error.back().getMaxMagnitude( level, hyteg::All );

      WALBERLA_LOG_INFO_ON_ROOT( "l2 error " + hyteg::stencilDirectionToStr[d] << " " << std::scientific << error_value );
   }

   /*
   {
      // hack to display the tetrahedron undistorted:
      hyteg::Cell& c      = *( storage->getCells().begin() )->second;
      auto&        coords = const_cast< std::array< hyteg::Point3D, 4 >& >( c.getCoordinates() );
      if ( coords[3][2] < 1 )
         coords[3][2] = 1.;
   }
    */

   if ( writeVtk )
   {
      hyteg::VTKOutput vtkOutput( "./output", filename + "_surr", storage );
      for ( uint_t i = 0; i < hyteg::ldlt::p1::dim3::lowerDirectionsAndCenter.size(); i += 1 )
      {
         vtkOutput.add( u_surr[i] );
      }
      vtkOutput.write( level, 0 );
   }

   if ( writeVtk )
   {
      hyteg::VTKOutput vtkOutput( "./output", filename + "_inpl", storage );
      for ( uint_t i = 0; i < hyteg::ldlt::p1::dim3::lowerDirectionsAndCenter.size(); i += 1 )
      {
         vtkOutput.add( u_inpl[i] );
      }
      vtkOutput.write( level, 0 );
   }

   if ( writeVtk )
   {
      hyteg::VTKOutput vtkOutput( "./output", filename + "_error", storage );
      for ( uint_t i = 0; i < hyteg::ldlt::p1::dim3::lowerDirectionsAndCenter.size(); i += 1 )
      {
         vtkOutput.add( u_error[i] );
      }
      vtkOutput.write( level, 0 );
   }
}
