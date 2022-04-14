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

std::vector< uint_t > createPermutation( uint_t permutationNumber )
{
   std::vector< uint_t > order{ 0, 1, 2, 3 };
   for ( uint_t i = 0; i < permutationNumber; ++i )
      std::next_permutation( std::begin( order ), std::end( order ) );
   return order;
}

void printPermutations()
{
   for ( uint_t permutationNumber = 0; permutationNumber < 24; ++permutationNumber )
   {
      auto order = createPermutation(permutationNumber);
      WALBERLA_LOG_INFO_ON_ROOT(permutationNumber << " " << order[0] << " " << order[1] << " " << order[2] << " " << order[3]);
   }
}

void executePermutation( std::array< hyteg::Point3D, 4 >& vertices, const std::vector< uint_t >& order )
{
   const auto p0 = vertices[0];
   const auto p1 = vertices[1];
   const auto p2 = vertices[2];
   const auto p3 = vertices[3];

   vertices[order[0]] = p0;
   vertices[order[1]] = p1;
   vertices[order[2]] = p2;
   vertices[order[3]] = p3;
}

std::shared_ptr< hyteg::SetupPrimitiveStorage > createDomain( walberla::Config::BlockHandle& parameters )
{
   const std::string domain = parameters.getParameter< std::string >( "domain" );

   if ( domain == "tetrahedron" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      const double top_x = 0.0;
      const double top_y = 0.0;
      const double top_z = parameters.getParameter< real_t >( "tetrahedron_height" );

      std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0.0, 0.0, 0.0 } ),
                                                   hyteg::Point3D( { 1.0, 0.0, 0.0 } ),
                                                   hyteg::Point3D( { 0.0, 1.0, 0.0 } ),
                                                   hyteg::Point3D( { top_x, top_y, top_z } ) };

      // we permutate the vertices to study performance for different orientations:
      auto order = createPermutation( parameters.getParameter< uint_t >( "tetrahedron_permutation" ) );
      executePermutation( vertices, order );

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
   else if ( domain == "two_layer_cube" )
   {
      const double    top_z = parameters.getParameter< real_t >( "tetrahedron_height" );
      hyteg::MeshInfo meshInfo =
          hyteg::MeshInfo::meshCuboid( hyteg::Point3D( { 0, 0, 0 } ), hyteg::Point3D( { 1, 1, 1. } ), 1, 1, 2 );
      for ( auto& vIt : meshInfo.getVertices() )
      {
         hyteg::MeshInfo::Vertex& v = vIt.second;
         auto                     c = v.getCoordinates();

         if ( c[2] >= 0.5 - 1e-14 && c[2] <= 0.5 + 1e-14 )
         {
            v.setCoordinates( hyteg::Point3D( { c[0], c[1], top_z } ) );
         }
      }

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );
      setupStorage->setMeshBoundaryFlagsByCentroidLocation( 2, [](const hyteg::Point3D & p){
         return ((p[0] <= 1e-16 || p[0] >= 1-1e-16) || (p[1] <= 1e-16 || p[1] >= 1-1e-16) ) && (p[2] > 1e-16 && p[2] < 1 - 1e-16);
      });

      return setupStorage;
   }
   else if ( domain == "two_layer_cube_v2" )
   {
      const double    top_z = parameters.getParameter< real_t >( "tetrahedron_height" );
      hyteg::MeshInfo meshInfo =
          hyteg::MeshInfo::meshCuboid( hyteg::Point3D( { 0, 0, 0 } ), hyteg::Point3D( { 1, 1, 2. } ), 1, 1, 2 );
      for ( auto& vIt : meshInfo.getVertices() )
      {
         hyteg::MeshInfo::Vertex& v = vIt.second;
         auto                     c = v.getCoordinates();

         if ( c[2] >= 2. - 1e-14 && c[2] <= 2. + 1e-14 )
         {
            v.setCoordinates( hyteg::Point3D( { c[0], c[1], 1+top_z } ) );
         }
      }

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );
      setupStorage->setMeshBoundaryFlagsByCentroidLocation( 2, [top_z](const hyteg::Point3D & p){
        return ((p[0] <= 1e-16 || p[0] >= 1-1e-16) || (p[1] <= 1e-16 || p[1] >= 1-1e-16) ) && (p[2] > 1e-16 && p[2] < 1+top_z - 1e-16);
      });

      return setupStorage;
   }
   else if ( domain == "tetrahedron_spindle" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0.0, 0.0, 0.5 } ),
                                                   hyteg::Point3D( { 0.0, 0.0, -0.5 } ),
                                                   hyteg::Point3D( { 0.5, 9.974968671630002, 0.0 } ),
                                                   hyteg::Point3D( { -0.5, 9.974968671630002, 0.0 } ) };
      WALBERLA_LOG_INFO_ON_ROOT( "unpermuted points: " << vertices[0] << " " << vertices[1] << vertices[2] << vertices[3] );

      auto order = createPermutation( parameters.getParameter< uint_t >( "tetrahedron_permutation" ) );
      WALBERLA_LOG_INFO_ON_ROOT( "permutation: " << order[0] << " " << order[1] << " " << order[2] << " " << order[3] );
      executePermutation( vertices, order );
      hyteg::MeshInfo meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );

      return setupStorage;
   }
   else if ( domain == "tetrahedron_cap" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0, 0, 0 } ),
                                                   hyteg::Point3D( { 1, 0, 0 } ),
                                                   hyteg::Point3D( { 0.5, 0.8660254037844386, 0.0 } ),
                                                   hyteg::Point3D( { 0.5, 0.28867513459481287, 0.09301739017887074 } ) };
      WALBERLA_LOG_INFO_ON_ROOT( "unpermuted points: " << vertices[0] << " " << vertices[1] << vertices[2] << vertices[3] );

      auto order = createPermutation( parameters.getParameter< uint_t >( "tetrahedron_permutation" ) );
      WALBERLA_LOG_INFO_ON_ROOT( "permutation: " << order[0] << " " << order[1] << " " << order[2] << " " << order[3] );
      executePermutation( vertices, order );
      hyteg::MeshInfo meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );

      return setupStorage;
   }
   else if ( domain == "tetrahedron_spade" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0, 0, 0 } ),
                                                   hyteg::Point3D( { 1.0, -0.666666, 0.0 } ),
                                                   hyteg::Point3D( { 1.0, 0.666666, 0.0 } ),
                                                   hyteg::Point3D( { 1.0, 0.0, 0.44293183140981923 } ) };
      WALBERLA_LOG_INFO_ON_ROOT( "unpermuted points: " << vertices[0] << " " << vertices[1] << vertices[2] << vertices[3] );

      auto order = createPermutation( parameters.getParameter< uint_t >( "tetrahedron_permutation" ) );
      WALBERLA_LOG_INFO_ON_ROOT( "permutation: " << order[0] << " " << order[1] << " " << order[2] << " " << order[3] );
      executePermutation( vertices, order );
      hyteg::MeshInfo meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

      auto setupStorage = std::make_shared< hyteg::SetupPrimitiveStorage >(
          meshInfo, uint_c( walberla::mpi::MPIManager::instance()->numProcesses() ) );
      setupStorage->setMeshBoundaryFlagsOnBoundary( 1, 0, true );

      return setupStorage;
   }
   else if ( domain == "tetrahedron_regular" )
   {
      WALBERLA_LOG_INFO_ON_ROOT( "Preparing " << domain << " domain." );

      std::array< hyteg::Point3D, 4 > vertices = { hyteg::Point3D( { 0, 0, 0 } ),
                                                   hyteg::Point3D( { 1, 0, 0 } ),
                                                   hyteg::Point3D( { 0.5, 0.8660254037844386, 0.0 } ),
                                                   hyteg::Point3D( { 0.5, 0.28867513459481287, 0.816496580927726 } ) };
      WALBERLA_LOG_INFO_ON_ROOT( "unpermuted points: " << vertices[0] << " " << vertices[1] << vertices[2] << vertices[3] );

      auto order = createPermutation( parameters.getParameter< uint_t >( "tetrahedron_permutation" ) );
      WALBERLA_LOG_INFO_ON_ROOT( "permutation: " << order[0] << " " << order[1] << " " << order[2] << " " << order[3] );
      executePermutation( vertices, order );
      hyteg::MeshInfo meshInfo = hyteg::MeshInfo::singleTetrahedron( vertices );

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
