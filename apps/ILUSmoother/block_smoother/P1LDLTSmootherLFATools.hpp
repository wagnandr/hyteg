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
#include "hyteg/StencilDirections.hpp"

#include "P1LDLTInplaceCellSmoother.hpp"

namespace hyteg {
namespace ldlt {
namespace p1 {
namespace dim3 {

using SD = stencilDirection;

std::vector< uint_t > createPermutation( uint_t permutationNumber )
{
   std::vector< uint_t > order{ 0, 1, 2, 3 };
   for ( uint_t i = 0; i < permutationNumber; ++i )
      std::next_permutation( std::begin( order ), std::end( order ) );
   return order;
}

std::map< SD, real_t > calculateAsymptoticLDLTStencil( std::map< SD, real_t >& a_stencil )
{
   std::map< stencilDirection, real_t > l_stencil_prev;
   std::map< stencilDirection, real_t > l_stencil_next;
   for ( auto d : hyteg::ldlt::p1::dim3::allDirections )
   {
      l_stencil_prev[d] = 0;
      l_stencil_next[d] = 0;
   }
   l_stencil_prev[SD::VERTEX_C] = 1.;
   l_stencil_next[SD::VERTEX_C] = 1.;

   // asymptotic iteration:
   real_t       last_diff = 0;
   const uint_t maxIter   = 10000;
   for ( uint_t i = 0; i < maxIter; i += 1 )
   {
      const real_t a_bc  = a_stencil[SD::VERTEX_BC];
      const real_t a_s   = a_stencil[SD::VERTEX_S];
      const real_t a_bnw = a_stencil[SD::VERTEX_BNW];
      const real_t a_be  = a_stencil[SD::VERTEX_BE];
      const real_t a_w   = a_stencil[SD::VERTEX_W];
      const real_t a_bn  = a_stencil[SD::VERTEX_BN];
      const real_t a_se  = a_stencil[SD::VERTEX_SE];
      const real_t a_c   = a_stencil[SD::VERTEX_C];

      // beta_bc:
      real_t beta_bc = a_bc / l_stencil_prev[SD::VERTEX_C];
      // beta_s:
      real_t beta_s = a_s;
      beta_s -= beta_bc * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_BN];
      beta_s /= l_stencil_prev[SD::VERTEX_C];
      // beta_bnw:
      real_t beta_bnw = a_bnw;
      beta_bnw -= beta_bc * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_SE];
      beta_bnw /= l_stencil_prev[SD::VERTEX_C];
      // beta_be:
      real_t beta_be = a_be;
      beta_be -= beta_bc * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_W];
      beta_be /= l_stencil_prev[SD::VERTEX_C];
      // beta_w:
      real_t beta_w = a_w;
      beta_w -= beta_bc * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_BE];
      beta_w -= beta_bnw * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_BN];
      beta_w -= beta_s * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_SE];
      beta_w /= l_stencil_prev[SD::VERTEX_C];
      // beta_bn:
      real_t beta_bn = a_bn;
      beta_bn -= beta_bc * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_S];
      beta_bn -= beta_be * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_SE];
      beta_bn -= beta_bnw * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_W];
      beta_bn /= l_stencil_prev[SD::VERTEX_C];
      // beta_se:
      real_t beta_se = a_se;
      beta_se -= beta_bc * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_BNW];
      beta_se -= beta_be * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_BN];
      beta_se -= beta_s * l_stencil_prev[SD::VERTEX_C] * l_stencil_prev[SD::VERTEX_W];
      beta_se /= l_stencil_prev[SD::VERTEX_C];
      // beta_c:
      real_t beta_c = a_c;
      beta_c -= beta_bc * beta_bc * l_stencil_prev[SD::VERTEX_C];
      beta_c -= beta_be * beta_be * l_stencil_prev[SD::VERTEX_C];
      beta_c -= beta_bnw * beta_bnw * l_stencil_prev[SD::VERTEX_C];
      beta_c -= beta_bn * beta_bn * l_stencil_prev[SD::VERTEX_C];
      beta_c -= beta_se * beta_se * l_stencil_prev[SD::VERTEX_C];
      beta_c -= beta_s * beta_s * l_stencil_prev[SD::VERTEX_C];
      beta_c -= beta_w * beta_w * l_stencil_prev[SD::VERTEX_C];

      l_stencil_next[SD::VERTEX_BC]  = beta_bc;
      l_stencil_next[SD::VERTEX_S]   = beta_s;
      l_stencil_next[SD::VERTEX_BNW] = beta_bnw;
      l_stencil_next[SD::VERTEX_BE]  = beta_be;
      l_stencil_next[SD::VERTEX_W]   = beta_w;
      l_stencil_next[SD::VERTEX_BN]  = beta_bn;
      l_stencil_next[SD::VERTEX_SE]  = beta_se;
      l_stencil_next[SD::VERTEX_C]   = beta_c;

      last_diff = 0;
      for ( auto d : hyteg::ldlt::p1::dim3::lowerDirectionsAndCenter )
      {
         last_diff += std::abs( l_stencil_prev[d] - l_stencil_next[d] );
      }

      //WALBERLA_LOG_INFO_ON_ROOT( "difference in iteration " << i << " is " << last_diff );
      //for ( auto d : ldlt::p1::dim3::lowerDirectionsAndCenter )
      //   WALBERLA_LOG_INFO(  stencilDirectionToStr[d] << " " << l_stencil_next[d] );

      l_stencil_prev = l_stencil_next;

      if ( last_diff < 1e-12 )
      {
         return l_stencil_prev;
      }
   }
   WALBERLA_ABORT( "Could not find asymptotic stencil. Last difference was " << last_diff << " after " << maxIter
                                                                             << " iterations." );
}

std::complex< real_t >
calculateSymbol( std::map< SD, real_t >& stencil, const std::array< real_t, 3 >& theta, const std::array< real_t, 3 >& h )
{
   using namespace std::complex_literals;
   std::complex< real_t > z_a = 0;
   for ( auto d : ldlt::p1::dim3::allDirections )
   {
      auto k = ldlt::p1::dim3::toIndex( d );
      z_a += std::exp( ( k[0] * theta[0] * h[0] + k[1] * theta[1] * h[1] + k[2] * theta[2] * h[2] ) * 1i ) * stencil[d];
   }
   return z_a;
}

std::map< SD, real_t > prepare_l_stencil( std::map< SD, real_t >& asymptotic_stencil )
{
   using namespace std::complex_literals;
   std::map< stencilDirection, real_t > l_stencil;
   for ( auto d : hyteg::ldlt::p1::dim3::allDirections )
      l_stencil[d] = 0;
   for ( auto d : hyteg::ldlt::p1::dim3::lowerDirections )
      l_stencil[d] = asymptotic_stencil[d];
   l_stencil[SD::VERTEX_C] = 1.;
   return l_stencil;
}

std::map< SD, real_t > prepare_lt_stencil( std::map< SD, real_t >& asymptotic_stencil )
{
   std::map< stencilDirection, real_t > lt_stencil;
   for ( auto d : hyteg::ldlt::p1::dim3::allDirections )
      lt_stencil[d] = 0;
   for ( auto d : hyteg::ldlt::p1::dim3::lowerDirections )
      lt_stencil[ldlt::p1::dim3::opposite( d )] = asymptotic_stencil[d];
   lt_stencil[SD::VERTEX_C] = 1.;
   return lt_stencil;
}

std::map< SD, real_t > prepare_d_stencil( std::map< SD, real_t >& asymptotic_stencil )
{
   std::map< stencilDirection, real_t > d_stencil;
   for ( auto d : hyteg::ldlt::p1::dim3::allDirections )
      d_stencil[d] = 0;
   d_stencil[SD::VERTEX_C] = asymptotic_stencil[SD::VERTEX_C];
   return d_stencil;
}

std::array< real_t, 3 > getMicroEdgeWidths( const std::array< hyteg::Point3D, 4 >& coordinates, uint_t level )
{
   auto                    N  = real_c( levelinfo::num_microedges_per_edge( level ) );
   real_t                  h1 = ( coordinates[1] - coordinates[0] ).norm() / N;
   real_t                  h2 = ( coordinates[2] - coordinates[0] ).norm() / N;
   real_t                  h3 = ( coordinates[3] - coordinates[0] ).norm() / N;
   std::array< real_t, 3 > h  = { h1, h2, h3 };
   return h;
}

template < typename FormType >
real_t estimateAsymptoticSmootherRateGS( const Cell& cell, uint_t level, FormType form )
{
   form.setGeometryMap( cell.getGeometryMap() );
   auto a_stencil  = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 2, 2, 2 }, cell, level, form );
   auto ld_stencil = a_stencil;
   auto u_stencil  = a_stencil;

   for ( auto d : lowerDirections )
      ld_stencil[opposite( d )] = 0;

   for ( auto d : lowerDirectionsAndCenter )
      u_stencil[d] = 0;

   const auto h = ldlt::p1::dim3::getMicroEdgeWidths( cell.getCoordinates(), level );

   // calculate symbols
   /*
      WALBERLA_LOG_INFO_ON_ROOT( "a stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << a_stencil[d] );
      WALBERLA_LOG_INFO_ON_ROOT( "l stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << l_stencil[d] );
      WALBERLA_LOG_INFO_ON_ROOT( "d stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << d_stencil[d] );
      WALBERLA_LOG_INFO_ON_ROOT( "lt stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << lt_stencil[d] );
      */

   int    num_samples_half    = 4;
   int    num_samples         = 2 * num_samples_half;
   real_t max_symbol          = 0;
   real_t mean_symbol         = 0;
   int    total_sample_number = 0;
   for ( int ix = -( num_samples - 1 ); ix < num_samples; ix += 1 )
   {
      for ( int iy = -( num_samples - 1 ); iy < num_samples; iy += 1 )
      {
         for ( int iz = -( num_samples - 1 ); iz < num_samples; iz += 1 )
         {
            real_t sx = ( real_c( ix ) / num_samples );
            real_t sy = ( real_c( iy ) / num_samples );
            real_t sz = ( real_c( iz ) / num_samples );

            using walberla::math::pi;
            real_t                  thetax = pi * sx / h[0];
            real_t                  thetay = pi * sy / h[1];
            real_t                  thetaz = pi * sz / h[2];
            std::array< real_t, 3 > theta  = { thetax, thetay, thetaz };

            bool ix_smooth = ( -pi / 2. / h[0] <= thetax ) && ( thetax <= pi / 2. / h[0] );
            bool iy_smooth = ( -pi / 2. / h[1] <= thetay ) && ( thetay <= pi / 2. / h[1] );
            bool iz_smooth = ( -pi / 2. / h[2] <= thetaz ) && ( thetaz <= pi / 2. / h[2] );

            if ( ix_smooth && iy_smooth && iz_smooth )
               continue;

            auto symbol_a  = ldlt::p1::dim3::calculateSymbol( a_stencil, theta, h );
            auto symbol_ld = ldlt::p1::dim3::calculateSymbol( ld_stencil, theta, h );
            auto symbol_u  = ldlt::p1::dim3::calculateSymbol( u_stencil, theta, h );

            auto symbol = std::abs( ( symbol_ld - symbol_a ) / symbol_ld );

            max_symbol = std::max( symbol, max_symbol );
            mean_symbol += symbol;
            total_sample_number += 1;

            /*
               WALBERLA_LOG_INFO_ON_ROOT( sx << " " << sy << " " << sz );
               WALBERLA_LOG_INFO_ON_ROOT( ix << " " << iy << " " << iz << ", " << theta[0] * h[0] << " " << theta[1] * h[1] << " "
                                             << theta[2] * h[2] << ", " << symbol << " " << max_symbol )
               WALBERLA_LOG_INFO_ON_ROOT( "a stencil symbol " << symbol_a );
               WALBERLA_LOG_INFO_ON_ROOT( "l stencil symbol " << symbol_l );
               WALBERLA_LOG_INFO_ON_ROOT( "lt stencil symbol " << symbol_lt );
               WALBERLA_LOG_INFO_ON_ROOT( "d stencil symbol " << symbol_d );
               WALBERLA_LOG_INFO_ON_ROOT( "ldt stencil symbol " << symbol_ldlt );
               WALBERLA_LOG_INFO_ON_ROOT( "symbol " << symbol );
               WALBERLA_LOG_INFO_ON_ROOT( "----------" )
               */
         }
      }
   }

   /*
      WALBERLA_LOG_INFO_ON_ROOT( "symbol max " << max_symbol );
      WALBERLA_LOG_INFO_ON_ROOT( "symbol mean " << mean_symbol / total_sample_number );
      */

   return max_symbol;
}

template < typename FormType >
real_t estimateAsymptoticSmootherRate( const std::array< Point3D, 4 >& coordinates, uint_t level, FormType form )
{
   auto a_stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 2, 2, 2 }, coordinates, level, form );
   auto l_stencil_asymptotic = ldlt::p1::dim3::calculateAsymptoticLDLTStencil( a_stencil );

   const auto h = ldlt::p1::dim3::getMicroEdgeWidths( coordinates, level );

   // calculate symbols
   auto l_stencil  = ldlt::p1::dim3::prepare_l_stencil( l_stencil_asymptotic );
   auto lt_stencil = ldlt::p1::dim3::prepare_lt_stencil( l_stencil_asymptotic );
   auto d_stencil  = ldlt::p1::dim3::prepare_d_stencil( l_stencil_asymptotic );

   /*
      WALBERLA_LOG_INFO_ON_ROOT( "a stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << a_stencil[d] );
      WALBERLA_LOG_INFO_ON_ROOT( "l stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << l_stencil[d] );
      WALBERLA_LOG_INFO_ON_ROOT( "d stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << d_stencil[d] );
      WALBERLA_LOG_INFO_ON_ROOT( "lt stencil:" );
      for ( auto d : ldlt::p1::dim3::allDirections )
         WALBERLA_LOG_INFO( " " << stencilDirectionToStr[d] << " " << lt_stencil[d] );
      */

   int    num_samples_half    = 8;
   int    num_samples         = 2 * num_samples_half;
   real_t max_symbol          = 0;
   real_t mean_symbol         = 0;
   int    total_sample_number = 0;
   for ( int ix = -( num_samples - 1 ); ix < num_samples; ix += 1 )
   {
      for ( int iy = -( num_samples - 1 ); iy < num_samples; iy += 1 )
      {
         for ( int iz = -( num_samples - 1 ); iz < num_samples; iz += 1 )
         {
            real_t sx = ( real_c( ix ) / num_samples );
            real_t sy = ( real_c( iy ) / num_samples );
            real_t sz = ( real_c( iz ) / num_samples );

            using walberla::math::pi;
            real_t                  thetax = pi * sx / h[0];
            real_t                  thetay = pi * sy / h[1];
            real_t                  thetaz = pi * sz / h[2];
            std::array< real_t, 3 > theta  = { thetax, thetay, thetaz };

            bool ix_smooth = ( -pi / 2. / h[0] <= thetax ) && ( thetax <= pi / 2. / h[0] );
            bool iy_smooth = ( -pi / 2. / h[1] <= thetay ) && ( thetay <= pi / 2. / h[1] );
            bool iz_smooth = ( -pi / 2. / h[2] <= thetaz ) && ( thetaz <= pi / 2. / h[2] );

            if ( ix_smooth && iy_smooth && iz_smooth )
               continue;

            auto symbol_a    = ldlt::p1::dim3::calculateSymbol( a_stencil, theta, h );
            auto symbol_l    = ldlt::p1::dim3::calculateSymbol( l_stencil, theta, h );
            auto symbol_lt   = ldlt::p1::dim3::calculateSymbol( lt_stencil, theta, h );
            auto symbol_d    = ldlt::p1::dim3::calculateSymbol( d_stencil, theta, h );
            auto symbol_ldlt = symbol_l * symbol_d * symbol_lt;

            auto symbol = std::abs( ( symbol_ldlt - symbol_a ) / symbol_ldlt );

            max_symbol = std::max( symbol, max_symbol );
            mean_symbol += symbol;
            total_sample_number += 1;

            /*
               WALBERLA_LOG_INFO_ON_ROOT( sx << " " << sy << " " << sz );
               WALBERLA_LOG_INFO_ON_ROOT( ix << " " << iy << " " << iz << ", " << theta[0] * h[0] << " " << theta[1] * h[1] << " "
                                             << theta[2] * h[2] << ", " << symbol << " " << max_symbol )
               WALBERLA_LOG_INFO_ON_ROOT( "a stencil symbol " << symbol_a );
               WALBERLA_LOG_INFO_ON_ROOT( "l stencil symbol " << symbol_l );
               WALBERLA_LOG_INFO_ON_ROOT( "lt stencil symbol " << symbol_lt );
               WALBERLA_LOG_INFO_ON_ROOT( "d stencil symbol " << symbol_d );
               WALBERLA_LOG_INFO_ON_ROOT( "ldt stencil symbol " << symbol_ldlt );
               WALBERLA_LOG_INFO_ON_ROOT( "symbol " << symbol );
               WALBERLA_LOG_INFO_ON_ROOT( "----------" )
               */
         }
      }
   }

   /*
      WALBERLA_LOG_INFO_ON_ROOT( "symbol max " << max_symbol );
      WALBERLA_LOG_INFO_ON_ROOT( "symbol mean " << mean_symbol / total_sample_number );
      */

   return max_symbol;
}

std::map< SD, real_t > prepare_restriction_stencil()
{
   std::map< stencilDirection, real_t > stencil;
   for ( auto d : hyteg::ldlt::p1::dim3::allDirections )
      stencil[d] = 0.5;
   stencil[SD::VERTEX_C] = 1.;
   return stencil;
}

std::array< real_t, 3 >
getTheta( const std::array< real_t, 3 >& theta, const std::array< uint_t, 3 >& alpha, const std::array< real_t, 3 >& h )
{
   using walberla::math::pi;
   std::array< real_t, 3 > theta_alpha{ 0, 0, 0 };
   for ( uint_t i = 0; i < 3; i += 1 )
   {
      auto sgn       = ( theta[i] > 0 ) ? +1. : -1.;
      theta_alpha[i] = theta[i] - real_c( alpha[i] ) * sgn * pi / h[i];
   }
   return theta_alpha;
}

Eigen::MatrixXcd getRestrictionTwoGridMatrix( const std::array< real_t, 3 > theta, const std::array< real_t, 3 >& h )
{
   auto resStencil = prepare_restriction_stencil();

   Eigen::MatrixXcd symRestriction( 1, 8 );
   int              dof = 0;
   for ( uint_t x = 0; x <= 1; x += 1 )
   {
      for ( uint_t y = 0; y <= 1; y += 1 )
      {
         for ( uint_t z = 0; z <= 1; z += 1 )
         {
            std::array< uint_t, 3 > alpha{ x, y, z };
            auto                    thetaAlpha = getTheta( theta, alpha, h );
            auto                    symbol     = calculateSymbol( resStencil, thetaAlpha, h );
            symRestriction( 0, dof )           = symbol;
            dof += 1;
         }
      }
   }

   return symRestriction;
}

Eigen::MatrixXcd
getOpTwoGridMatrix( std::map< SD, real_t > op_sencil, const std::array< real_t, 3 > theta, const std::array< real_t, 3 >& h )
{
   Eigen::MatrixXcd matrix( 8, 8 );
   matrix.setZero();
   int dof = 0;
   for ( uint_t x = 0; x <= 1; x += 1 )
   {
      for ( uint_t y = 0; y <= 1; y += 1 )
      {
         for ( uint_t z = 0; z <= 1; z += 1 )
         {
            std::array< uint_t, 3 > alpha{ x, y, z };
            auto                    thetaAlpha = getTheta( theta, alpha, h );
            matrix( dof, dof )                 = calculateSymbol( op_sencil, thetaAlpha, h );

            dof += 1;
         }
      }
   }

   return matrix;
}

bool isFeasible( std::map< SD, real_t > op_sencil, const std::array< real_t, 3 > theta, const std::array< real_t, 3 >& h )
{
   for ( uint_t x = 0; x <= 1; x += 1 )
   {
      for ( uint_t y = 0; y <= 1; y += 1 )
      {
         for ( uint_t z = 0; z <= 1; z += 1 )
         {
            std::array< uint_t, 3 > alpha{ x, y, z };
            auto                    thetaAlpha = getTheta( theta, alpha, h );
            auto                    symbol     = calculateSymbol( op_sencil, thetaAlpha, h );

            if ( std::abs( symbol ) < 1e-14 )
               return false;
         }
      }
   }

   return true;
}

Eigen::MatrixXcd getSmootherTwoGridMatrix( std::map< SD, real_t >         a_stencil,
                                           const std::array< real_t, 3 >  theta,
                                           const std::array< real_t, 3 >& h,
                                           real_t                         nu )
{
   auto l_stencil_asymptotic = ldlt::p1::dim3::calculateAsymptoticLDLTStencil( a_stencil );

   auto l_stencil  = ldlt::p1::dim3::prepare_l_stencil( l_stencil_asymptotic );
   auto lt_stencil = ldlt::p1::dim3::prepare_lt_stencil( l_stencil_asymptotic );
   auto d_stencil  = ldlt::p1::dim3::prepare_d_stencil( l_stencil_asymptotic );

   Eigen::MatrixXcd matrix( 8, 8 );
   matrix.setZero();
   int dof = 0;
   for ( uint_t x = 0; x <= 1; x += 1 )
   {
      for ( uint_t y = 0; y <= 1; y += 1 )
      {
         for ( uint_t z = 0; z <= 1; z += 1 )
         {
            std::array< uint_t, 3 > alpha{ x, y, z };
            auto                    thetaAlpha = getTheta( theta, alpha, h );

            auto symbol_a    = ldlt::p1::dim3::calculateSymbol( a_stencil, thetaAlpha, h );
            auto symbol_l    = ldlt::p1::dim3::calculateSymbol( l_stencil, thetaAlpha, h );
            auto symbol_lt   = ldlt::p1::dim3::calculateSymbol( lt_stencil, thetaAlpha, h );
            auto symbol_d    = ldlt::p1::dim3::calculateSymbol( d_stencil, thetaAlpha, h );
            auto symbol_ldlt = symbol_l * symbol_d * symbol_lt;

            auto symbol = ( symbol_ldlt - symbol_a ) / symbol_ldlt;

            matrix( dof, dof ) = std::pow( symbol, nu );

            dof += 1;
         }
      }
   }

   return matrix;
}

Eigen::MatrixXcd getSmootherTwoGridMatrixGS( std::map< SD, real_t >         a_stencil,
                                             const std::array< real_t, 3 >  theta,
                                             const std::array< real_t, 3 >& h,
                                             real_t                         nu )
{
   auto ld_stencil = a_stencil;
   for ( auto d : lowerDirections )
      ld_stencil[opposite( d )] = 0;

   Eigen::MatrixXcd matrix( 8, 8 );
   matrix.setZero();
   int dof = 0;
   for ( uint_t x = 0; x <= 1; x += 1 )
   {
      for ( uint_t y = 0; y <= 1; y += 1 )
      {
         for ( uint_t z = 0; z <= 1; z += 1 )
         {
            std::array< uint_t, 3 > alpha{ x, y, z };
            auto                    thetaAlpha = getTheta( theta, alpha, h );

            auto symbol_a  = ldlt::p1::dim3::calculateSymbol( a_stencil, thetaAlpha, h );
            auto symbol_ld = ldlt::p1::dim3::calculateSymbol( ld_stencil, thetaAlpha, h );

            auto symbol = ( symbol_ld - symbol_a ) / symbol_ld;

            matrix( dof, dof ) = std::pow( symbol, nu );

            dof += 1;
         }
      }
   }

   return matrix;
}

template < typename FormType >
real_t estimateAsymptoticTwoGridRate( const Cell& cell, uint_t level, FormType form, bool useGS )
{
   auto a_stencil_fine   = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 1, 1, 1 }, cell, level, form );
   auto a_stencil_coarse = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 1, 1, 1 }, cell, level - 1, form );

   auto asymptotic_ldlt_stencil = calculateAsymptoticLDLTStencil( a_stencil_fine );
   auto l_stencil               = ldlt::p1::dim3::prepare_l_stencil( asymptotic_ldlt_stencil );
   auto lt_stencil              = ldlt::p1::dim3::prepare_lt_stencil( asymptotic_ldlt_stencil );
   auto d_stencil               = ldlt::p1::dim3::prepare_d_stencil( asymptotic_ldlt_stencil );

   // TODO
   auto h        = getMicroEdgeWidths( cell.getCoordinates(), level );
   auto h_coarse = getMicroEdgeWidths( cell.getCoordinates(), level - 1 );

   int    num_samples_half    = 16;
   int    num_samples         = 2 * num_samples_half;
   real_t max_symbol          = 0;
   real_t mean_symbol         = 0;
   int    total_sample_number = 0;
   for ( int ix = -( num_samples - 1 ); ix < num_samples; ix += 1 )
   {
      for ( int iy = -( num_samples - 1 ); iy < num_samples; iy += 1 )
      {
         for ( int iz = -( num_samples - 1 ); iz < num_samples; iz += 1 )
         {
            real_t sx = ( real_c( ix ) / num_samples );
            real_t sy = ( real_c( iy ) / num_samples );
            real_t sz = ( real_c( iz ) / num_samples );
            // real_t sx = -0.25;
            // real_t sy = -0.25;
            //real_t sz = -0.25;

            using walberla::math::pi;
            real_t                  thetax = pi * sx / h[0] / 2.;
            real_t                  thetay = pi * sy / h[1] / 2.;
            real_t                  thetaz = pi * sz / h[2] / 2.;
            std::array< real_t, 3 > theta  = { thetax, thetay, thetaz };

            std::array< real_t, 3 > theta_coarse = { 2 * thetax, 2 * thetay, 2 * thetaz };

            if ( !isFeasible( a_stencil_fine, theta, h ) )
               continue;

            Eigen::MatrixXcd restriction = getRestrictionTwoGridMatrix( theta, h );
            // WALBERLA_LOG_INFO_ON_ROOT("restriction " << restriction);
            Eigen::MatrixXcd interpolation = ( 1 / 8. ) * restriction.transpose().conjugate();
            // Eigen::MatrixXcd interpolation = 8. * restriction.transpose().conjugate();
            // Eigen::MatrixXcd interpolation = restriction.transpose().conjugate();
            // WALBERLA_LOG_INFO_ON_ROOT("interpolation " << interpolation);
            std::complex< real_t > op_coarse = calculateSymbol( a_stencil_coarse, theta_coarse, h_coarse );
            // WALBERLA_LOG_INFO_ON_ROOT("coarse " << op_coarse);
            Eigen::MatrixXcd op_fine = getOpTwoGridMatrix( a_stencil_fine, theta, h );
            // WALBERLA_LOG_INFO_ON_ROOT("coarse " << op_fine);
            Eigen::MatrixXcd smoother;
            if ( useGS )
            {
               smoother = getSmootherTwoGridMatrixGS( a_stencil_fine, theta, h, 3 );
            }
            else
            {
               smoother = getSmootherTwoGridMatrix( a_stencil_fine, theta, h, 3 );
            }
            // WALBERLA_LOG_INFO_ON_ROOT("smoother " << smoother);
            Eigen::MatrixXcd id( 8, 8 );
            id.setIdentity();
            // WALBERLA_LOG_INFO_ON_ROOT("id " << id);
            Eigen::MatrixXcd C = ( id - ( 1. / op_coarse ) * interpolation * restriction * op_fine );
            Eigen::MatrixXcd S = smoother * C * smoother;
            // Eigen::MatrixXcd S = smoother * C * smoother * smoother * C * smoother;
            // WALBERLA_LOG_INFO_ON_ROOT( ( 1. / op_coarse ) );

            if ( std::abs( op_coarse ) < 1e-14 )
               continue;

            if ( ix == 0 && iy == 0 && iz == 0 )
               continue;

            Eigen::SelfAdjointEigenSolver< Eigen::MatrixXcd > eigensolver( S );
            if ( eigensolver.info() != Eigen::Success )
               WALBERLA_ABORT( "eigensolver failed" );
            real_t symbol = eigensolver.eigenvalues().cwiseAbs().maxCoeff();
            // real_t symbol = S.operatorNorm();

            // WALBERLA_LOG_INFO_ON_ROOT(sx << " " << sy << " " << sz << " " << symbol);

            /*
            if (symbol > 1)
            {
               WALBERLA_LOG_INFO_ON_ROOT("restriction \n" << restriction);
               WALBERLA_LOG_INFO_ON_ROOT("interpolation \n" << interpolation);
               WALBERLA_LOG_INFO_ON_ROOT("coarse \n" << op_coarse);
               WALBERLA_LOG_INFO_ON_ROOT("fine \n" << op_fine);
               WALBERLA_LOG_INFO_ON_ROOT("smoother \n" << smoother);
               WALBERLA_LOG_INFO_ON_ROOT("id \n" << id);
               WALBERLA_LOG_INFO_ON_ROOT("C \n" << C);
               for (uint_t x = 0; x <= 1; x+=1)
                  for (uint_t y = 0; y <= 1; y+=1)
                     for (uint_t z = 0; z <= 1; z+=1)
                        WALBERLA_LOG_INFO_ON_ROOT(x << " " << y << " " << z);

               Eigen::MatrixXcd        restriction2   = getRestrictionTwoGridMatrix( theta, h );
            }
            */

            if ( ix == 0 && iy == 0 && iz == 0 )
            {
               continue;
            }

            max_symbol = std::max( max_symbol, symbol );
         }
      }
   }

   return max_symbol;
}

template < typename FormType >
class ILUPermutator
{
 public:
   ILUPermutator( uint_t level, FormType& form )
       : level_( level )
       , form_( form )
   {}

   std::array< uint_t, 4 > operator()( const Cell& cell )
   {
      auto id = std::make_shared< IdentityMap >();
      form_.setGeometryMap( id );

      std::vector< uint_t > bestPermutation;
      real_t                minSymbol = std::numeric_limits< real_t >::max();

      for ( uint_t permutationId = 0; permutationId < 24; permutationId += 1 )
      {
         auto                     permutation           = createPermutation( permutationId );
         std::array< Point3D, 4 > permutatedCoordinates = {
             cell.getCoordinates()[permutation[0]],
             cell.getCoordinates()[permutation[1]],
             cell.getCoordinates()[permutation[2]],
             cell.getCoordinates()[permutation[3]],
         };

         auto symbol = ldlt::p1::dim3::estimateAsymptoticSmootherRate( cell.getCoordinates(), level_, form_ );

         if ( symbol < minSymbol )
         {
            minSymbol       = symbol;
            bestPermutation = permutation;
         }
      }

      return std::array< uint_t, 4 >( {
                                          bestPermutation[0],
                                          bestPermutation[1],
                                          bestPermutation[2],
                                          bestPermutation[3],
                                      } );
   }

 private:
   uint_t   level_;
   FormType form_;
};

} // namespace dim3
} // namespace p1
} // namespace ldlt
} // namespace hyteg

