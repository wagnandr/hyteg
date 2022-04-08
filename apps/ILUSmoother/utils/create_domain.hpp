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
#pragma once

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/math/Constants.h"

#include "hyteg/primitivestorage/SetupPrimitiveStorage.hpp"

using walberla::real_t;
using walberla::uint_c;
using walberla::uint_t;

std::shared_ptr< hyteg::SetupPrimitiveStorage > createDomain( walberla::Config::BlockHandle& parameters )
{
   const std::string domain = parameters.getParameter< std::string >( "domain" );

   if ( domain == "tetrahedron" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      const double top_x = 0.0;
      const double top_y = 0.0;
      const double top_z = parameters.getParameter< real_t >( "tetrahedron_height" );

      const hyteg::Point3D p0( { 0, 0, 0 } );
      const hyteg::Point3D p1( { 1.0, 0, 0 } );
      const hyteg::Point3D p2( { 0.0, 1.0, 0 } );
      const hyteg::Point3D p3( { top_x, top_y, top_z } );

      // we permutate the vertices to study performance for different orientations:
      const uint_t          permutationNumber = parameters.getParameter< uint_t >( "tetrahedron_permutation" );
      std::vector< uint_t > order{ 0, 1, 2, 3 };
      for ( uint_t i = 0; i < permutationNumber; ++i )
         std::next_permutation( std::begin( order ), std::end( order ) );

      std::array< hyteg::Point3D, 4 > vertices;
      vertices[order[0]] = p0;
      vertices[order[1]] = p1;
      vertices[order[2]] = p2;
      vertices[order[3]] = p3;

      hyteg::MeshInfo meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );

      return setupStorage;
   }
   else if ( domain == "squished_cube" )
   {
      const double    top_z = parameters.getParameter< real_t >( "tetrahedron_height" );
      hyteg::MeshInfo meshInfo =
         hyteg::MeshInfo::meshCuboid( hyteg::Point3D( { 0, 0, 0 } ), hyteg::Point3D( { 1, 1, top_z } ), 1, 1, 1 );
         // hyteg::MeshInfo::meshSymmetricCuboid( hyteg::Point3D( { 0, 0, 0 } ), hyteg::Point3D( { 1, 1, top_z } ), 1, 1, 1 );
      // hyteg::MeshInfo::fromGmshFile("../../data/meshes/3D/pyramid_2el.msh");
      // hyteg::MeshInfo::fromGmshFile("../../data/meshes/3D/tet_1el.msh");

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );

      return setupStorage;
   }
   else
   {
      WALBERLA_ABORT( "unknown domain" );
   }
}
