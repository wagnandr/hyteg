#pragma once

#include "hyteg/polynomial/LSQPInterpolator.hpp"
#include "hyteg/polynomial/QuadrilateralBasis.hpp"
#include "hyteg/polynomial/QuadrilateralLSQPInterpolator.hpp"
#include "hyteg/polynomial/QuadrilateralPolynomial.hpp"
#include "hyteg/polynomial/QuadrilateralPolynomialEvaluator.hpp"

#include "PrimitiveSmoothers.hpp"

namespace hyteg {

namespace ldlt {
namespace p1 {
namespace dim3 {

using SD = stencilDirection;

constexpr std::array< SD, 7 > lowerDirections =
    { SD::VERTEX_W, SD::VERTEX_S, SD::VERTEX_SE, SD::VERTEX_BNW, SD::VERTEX_BN, SD::VERTEX_BC, SD::VERTEX_BE };

constexpr std::array< SD, 8 > lowerDirectionsAndCenter =
    { SD::VERTEX_W, SD::VERTEX_S, SD::VERTEX_SE, SD::VERTEX_BNW, SD::VERTEX_BN, SD::VERTEX_BC, SD::VERTEX_BE, SD::VERTEX_C };

constexpr std::array< SD, 7 > upperDirections =
    { SD::VERTEX_E, SD::VERTEX_N, SD::VERTEX_NW, SD::VERTEX_TSE, SD::VERTEX_TS, SD::VERTEX_TC, SD::VERTEX_TW };

SD opposite( SD dir )
{
   switch ( dir )
   {
   case SD::VERTEX_W:
      return SD::VERTEX_E;
   case SD::VERTEX_S:
      return SD::VERTEX_N;
   case SD::VERTEX_SE:
      return SD::VERTEX_NW;
   case SD::VERTEX_BNW:
      return SD::VERTEX_TSE;
   case SD::VERTEX_BN:
      return SD::VERTEX_TS;
   case SD::VERTEX_BC:
      return SD::VERTEX_TC;
   case SD::VERTEX_BE:
      return SD::VERTEX_TW;
   case SD::VERTEX_C:
      return SD::VERTEX_C;
   case SD::VERTEX_E:
      return SD::VERTEX_W;
   case SD::VERTEX_N:
      return SD::VERTEX_S;
   case SD::VERTEX_NW:
      return SD::VERTEX_SE;
   case SD::VERTEX_TSE:
      return SD::VERTEX_BNW;
   case SD::VERTEX_TS:
      return SD::VERTEX_BN;
   case SD::VERTEX_TC:
      return SD::VERTEX_BC;
   case SD::VERTEX_TW:
      return SD::VERTEX_BE;
   default:
      break;
   };
   WALBERLA_ABORT( "not implemented" );
}

void print_stencil( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil )
{
   for ( auto d : ldlt::p1::dim3::lowerDirectionsAndCenter )
      WALBERLA_LOG_INFO( "x " << x << " y " << y << " z " << z << " " << stencilDirectionToStr[d] << " " << stencil[d] );
}

bool on_west_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   WALBERLA_UNUSED( y );
   WALBERLA_UNUSED( z );
   WALBERLA_UNUSED( N );
   return x == 1;
}

bool on_diagonal_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   return x + y + z == N - 2;
}

bool on_south_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   WALBERLA_UNUSED( x );
   WALBERLA_UNUSED( z );
   WALBERLA_UNUSED( N );
   return y == 1;
}

bool on_bottom_boundary( uint_t x, uint_t y, uint_t z, uint_t N )
{
   WALBERLA_UNUSED( x );
   WALBERLA_UNUSED( y );
   WALBERLA_UNUSED( N );
   return z == 1;
}

void apply_boundary_corrections( uint_t x, uint_t y, uint_t z, uint_t N, std::map< SD, real_t >& stencil )
{
   if ( x == 1 )
   {
      stencil[stencilDirection::VERTEX_NW]  = 0;
      stencil[stencilDirection::VERTEX_W]   = 0;
      stencil[stencilDirection::VERTEX_TW]  = 0;
      stencil[stencilDirection::VERTEX_BNW] = 0;
   }
   if ( x + y + z == N - 2 )
   {
      stencil[stencilDirection::VERTEX_E]   = 0;
      stencil[stencilDirection::VERTEX_N]   = 0;
      stencil[stencilDirection::VERTEX_TC]  = 0;
      stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( y == 1 )
   {
      stencil[stencilDirection::VERTEX_S]   = 0;
      stencil[stencilDirection::VERTEX_SE]  = 0;
      stencil[stencilDirection::VERTEX_TS]  = 0;
      stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( z == 1 )
   {
      stencil[stencilDirection::VERTEX_BC]  = 0;
      stencil[stencilDirection::VERTEX_BE]  = 0;
      stencil[stencilDirection::VERTEX_BN]  = 0;
      stencil[stencilDirection::VERTEX_BNW] = 0;
   }
}

bool on_cell_boundary( uint_t x, uint_t y, uint_t z, uint_t size_boundary, uint_t num_dof_on_edge )
{
   const bool on_east_boundary     = x < size_boundary;
   const bool on_south_boundary    = y < size_boundary;
   const bool on_bottom_boundary   = z < size_boundary;
   const bool on_diagonal_boundary = x + y + z > num_dof_on_edge - 1 - size_boundary;
   return on_east_boundary || on_south_boundary || on_bottom_boundary || on_diagonal_boundary;
}

real_t apply_boundary_corrections_to_scalar( uint_t x, uint_t y, uint_t z, uint_t N, SD d, real_t value )
{
   // if we are on the outmost boundary we return the unit stencil:
   if ( on_cell_boundary( x, y, z, 1, N ) )
   {
      if ( d == SD::VERTEX_C )
         return 1.;
      return 0.;
   }

   const bool on_west_boundary     = x == 1;
   const bool on_south_boundary    = y == 1;
   const bool on_bottom_boundary   = z == 1;
   const bool on_diagonal_boundary = x + y + z == N - 2;

   const bool is_western_stencil = ( d == stencilDirection::VERTEX_NW ) || ( d == stencilDirection::VERTEX_W ) ||
                                   ( d == stencilDirection::VERTEX_TW ) || ( d == stencilDirection::VERTEX_BNW );

   const bool is_southern_stencil = ( d == stencilDirection::VERTEX_S ) || ( d == stencilDirection::VERTEX_SE ) ||
                                    ( d == stencilDirection::VERTEX_TS ) || ( d == stencilDirection::VERTEX_TSE );

   const bool is_bottom_stencil = ( d == stencilDirection::VERTEX_BC ) || ( d == stencilDirection::VERTEX_BE ) ||
                                  ( d == stencilDirection::VERTEX_BN ) || ( d == stencilDirection::VERTEX_BNW );

   const bool is_diagonal_stencil = ( d == stencilDirection::VERTEX_E ) || ( d == stencilDirection::VERTEX_N ) ||
                                    ( d == stencilDirection::VERTEX_TC ) || ( d == stencilDirection::VERTEX_TSE );

   if ( on_west_boundary && is_western_stencil )
      return 0;

   if ( on_south_boundary && is_southern_stencil )
      return 0;

   if ( on_bottom_boundary && is_bottom_stencil )
      return 0;

   if ( on_diagonal_boundary && is_diagonal_stencil )
      return 0;

   // if we do not belong to any boundary, we return the value
   return value;
}

void apply_boundary_corrections_on_backward_stencil( uint_t x, uint_t y, uint_t z, uint_t N, std::map< SD, real_t >& stencil )
{
   stencil[SD::VERTEX_W]  = apply_boundary_corrections_to_scalar( x + 1, y, z, N, SD::VERTEX_W, stencil[SD::VERTEX_W] );
   stencil[SD::VERTEX_S]  = apply_boundary_corrections_to_scalar( x, y + 1, z, N, SD::VERTEX_S, stencil[SD::VERTEX_S] );
   stencil[SD::VERTEX_SE] = apply_boundary_corrections_to_scalar( x - 1, y + 1, z, N, SD::VERTEX_SE, stencil[SD::VERTEX_SE] );
   stencil[SD::VERTEX_BNW] =
       apply_boundary_corrections_to_scalar( x + 1, y - 1, z + 1, N, SD::VERTEX_BNW, stencil[SD::VERTEX_BNW] );
   stencil[SD::VERTEX_BN] = apply_boundary_corrections_to_scalar( x, y - 1, z + 1, N, SD::VERTEX_BN, stencil[SD::VERTEX_BN] );
   stencil[SD::VERTEX_BC] = apply_boundary_corrections_to_scalar( x, y, z + 1, N, SD::VERTEX_BC, stencil[SD::VERTEX_BC] );
   stencil[SD::VERTEX_BE] = apply_boundary_corrections_to_scalar( x - 1, y, z + 1, N, SD::VERTEX_BE, stencil[SD::VERTEX_BE] );
}

template < typename FormType, typename FactorizationCallback >
void factorize_matrix( FormType& form, uint_t level, Cell& cell, const FactorizationCallback& cb )
{
   // face index on level
   const auto fidx = [=]( uint_t x, uint_t y ) { return vertexdof::macroface::indexFromVertex( level, x, y, SD::VERTEX_C ); };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );
   const auto N_face = levelinfo::num_microvertices_per_face( level );

   // the current and lower row in the factorization:
   std::map< SD, std::vector< real_t > > beta;
   std::map< SD, std::vector< real_t > > gamma;

   for ( auto d : lowerDirections )
   {
      beta[d]  = std::vector< real_t >( N_face, 0 );
      gamma[d] = std::vector< real_t >( N_face, 0 );
   }
   beta[SD::VERTEX_C]  = std::vector< real_t >( N_face, 1 );
   gamma[SD::VERTEX_C] = std::vector< real_t >( N_face, 1 );

   // l stencil
   std::map< SD, real_t > l_stencil;

   // add blending:
   form.setGeometryMap( cell.getGeometryMap() );

   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            auto a_stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm( { x, y, z }, cell, level, form );

            apply_boundary_corrections( x, y, z, N_edge, a_stencil );

            const real_t a_bc  = a_stencil[SD::VERTEX_BC];
            const real_t a_s   = a_stencil[SD::VERTEX_S];
            const real_t a_bnw = a_stencil[SD::VERTEX_BNW];
            const real_t a_be  = a_stencil[SD::VERTEX_BE];
            const real_t a_w   = a_stencil[SD::VERTEX_W];
            const real_t a_bn  = a_stencil[SD::VERTEX_BN];
            const real_t a_se  = a_stencil[SD::VERTEX_SE];
            const real_t a_c   = a_stencil[SD::VERTEX_C];

//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_bc, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y - 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_s, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y + 1, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_bnw, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_be, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_w, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y + 1, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_bn, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y - 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_se, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( a_c, 0. );

            // beta_bc:
            real_t beta_bc = a_bc / gamma[SD::VERTEX_C][fidx( x, y )];
            // beta_s:
            real_t beta_s = a_s;
            beta_s -= beta_bc * gamma[SD::VERTEX_C][fidx( x, y )] * beta[SD::VERTEX_BN][fidx( x, y - 1 )];
            beta_s /= beta[SD::VERTEX_C][fidx( x, y - 1 )];
            // beta_bnw:
            real_t beta_bnw = a_bnw;
            beta_bnw -= beta_bc * gamma[SD::VERTEX_C][fidx( x, y )] * gamma[SD::VERTEX_SE][fidx( x - 1, y + 1 )];
            beta_bnw /= gamma[SD::VERTEX_C][fidx( x - 1, y + 1 )];
            // beta_be:
            real_t beta_be = a_be;
            beta_be -= beta_bc * gamma[SD::VERTEX_C][fidx( x, y )] * gamma[SD::VERTEX_W][fidx( x + 1, y )];
            beta_be /= gamma[SD::VERTEX_C][fidx( x + 1, y )];
            // beta_w:
            real_t beta_w = a_w;
            beta_w -= beta_bc * gamma[SD::VERTEX_C][fidx( x, y )] * beta[SD::VERTEX_BE][fidx( x - 1, y )];
            beta_w -= beta_bnw * gamma[SD::VERTEX_C][fidx( x - 1, y + 1 )] * beta[SD::VERTEX_BN][fidx( x - 1, y )];
            beta_w -= beta_s * beta[SD::VERTEX_C][fidx( x, y - 1 )] * beta[SD::VERTEX_SE][fidx( x - 1, y )];
            beta_w /= beta[SD::VERTEX_C][fidx( x - 1, y )];
            // beta_bn:
            real_t beta_bn = a_bn;
            beta_bn -= beta_bc * gamma[SD::VERTEX_C][fidx( x, y )] * gamma[SD::VERTEX_S][fidx( x, y + 1 )];
            beta_bn -= beta_be * gamma[SD::VERTEX_C][fidx( x + 1, y )] * gamma[SD::VERTEX_SE][fidx( x, y + 1 )];
            beta_bn -= beta_bnw * gamma[SD::VERTEX_C][fidx( x - 1, y + 1 )] * gamma[SD::VERTEX_W][fidx( x, y + 1 )];
            beta_bn /= gamma[SD::VERTEX_C][fidx( x, y + 1 )];
            // beta_se:
            real_t beta_se = a_se;
            beta_se -= beta_bc * gamma[SD::VERTEX_C][fidx( x, y )] * beta[SD::VERTEX_BNW][fidx( x + 1, y - 1 )];
            beta_se -= beta_be * gamma[SD::VERTEX_C][fidx( x + 1, y )] * beta[SD::VERTEX_BN][fidx( x + 1, y - 1 )];
            beta_se -= beta_s * beta[SD::VERTEX_C][fidx( x, y - 1 )] * beta[SD::VERTEX_W][fidx( x + 1, y - 1 )];
            beta_se /= beta[SD::VERTEX_C][fidx( x + 1, y - 1 )];
            // beta_c:
            real_t beta_c = a_c;
            beta_c -= beta_bc * beta_bc * gamma[SD::VERTEX_C][fidx( x, y )];
            beta_c -= beta_be * beta_be * gamma[SD::VERTEX_C][fidx( x + 1, y )];
            beta_c -= beta_bnw * beta_bnw * gamma[SD::VERTEX_C][fidx( x - 1, y + 1 )];
            beta_c -= beta_bn * beta_bn * gamma[SD::VERTEX_C][fidx( x, y + 1 )];
            beta_c -= beta_se * beta_se * beta[SD::VERTEX_C][fidx( x + 1, y - 1 )];
            beta_c -= beta_s * beta_s * beta[SD::VERTEX_C][fidx( x, y - 1 )];
            beta_c -= beta_w * beta_w * beta[SD::VERTEX_C][fidx( x - 1, y )];

//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_bc, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y - 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_s, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y + 1, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_bnw, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_be, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_w, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y + 1, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_bn, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y - 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_se, 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( beta_c, 0. );

            // write back into beta:
            beta[SD::VERTEX_BC][fidx( x, y )]  = beta_bc;
            beta[SD::VERTEX_S][fidx( x, y )]   = beta_s;
            beta[SD::VERTEX_BNW][fidx( x, y )] = beta_bnw;
            beta[SD::VERTEX_BE][fidx( x, y )]  = beta_be;
            beta[SD::VERTEX_W][fidx( x, y )]   = beta_w;
            beta[SD::VERTEX_BN][fidx( x, y )]  = beta_bn;
            beta[SD::VERTEX_SE][fidx( x, y )]  = beta_se;
            beta[SD::VERTEX_C][fidx( x, y )]   = beta_c;

            // copy into stencil:
            l_stencil[SD::VERTEX_BC]  = beta_bc;
            l_stencil[SD::VERTEX_S]   = beta_s;
            l_stencil[SD::VERTEX_BNW] = beta_bnw;
            l_stencil[SD::VERTEX_BE]  = beta_be;
            l_stencil[SD::VERTEX_W]   = beta_w;
            l_stencil[SD::VERTEX_BN]  = beta_bn;
            l_stencil[SD::VERTEX_SE]  = beta_se;
            l_stencil[SD::VERTEX_C]   = beta_c;

            cb( x, y, z, l_stencil );
         } // end x loop
      }    // end y loop

      // copy gamma <- beta and initialize beta
      for ( uint_t i = 0; i < N_face; i += 1 )
      {
         for ( auto d : lowerDirections )
         {
            gamma[d][i] = beta[d][i];
            beta[d][i]  = 0;
         }
         gamma[SD::VERTEX_C][i] = beta[SD::VERTEX_C][i];
         beta[SD::VERTEX_C][i]  = 1;
      }
   } // end z lopp
}

template < typename LStencilProvider, typename FunctionType >
void apply_substitutions( LStencilProvider&   get_l_stencil,
                          uint_t              level,
                          Cell&               cell,
                          const FunctionType& u_function,
                          const FunctionType& b_function )
{
   const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
      return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
   };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   std::map< SD, real_t > l_stencil;

   // unpack u and b
   auto u = cell.getData( u_function.getCellDataID() )->getPointer( level );
   auto b = cell.getData( b_function.getCellDataID() )->getPointer( level );

   // forward substitution:
   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            get_l_stencil( x, y, z, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] = b[cidx( x, y, z, SD::VERTEX_C )];

//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_BC], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y - 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_S], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y + 1, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_BNW], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_BE], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_W], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y + 1, z - 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_BN], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y - 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_SE], 0. );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[SD::VERTEX_C], 0. );

            for ( auto d : lowerDirections )
            {
               u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[d] * u[cidx( x, y, z, d )];
            }
         }
      }
   }

   // diagonal:
   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            get_l_stencil( x, y, z, l_stencil );
            //print_stencil(x,y,z,l_stencil);
            u[cidx( x, y, z, SD::VERTEX_C )] /= l_stencil[SD::VERTEX_C];
         }
      }
   }

   // backward substitution:
   for ( uint_t z = N_edge - 2; z >= 1; z -= 1 )
   {
      for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
      {
         for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
         {
            // E
            get_l_stencil( x + 1, y, z, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_E )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_E )] * u[cidx( x, y, z, SD::VERTEX_E )];

            // N
            get_l_stencil( x, y + 1, z, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y + 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_N )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_N )] * u[cidx( x, y, z, SD::VERTEX_N )];

            // NW
            get_l_stencil( x - 1, y + 1, z, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y + 1, z ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_NW )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_NW )] * u[cidx( x, y, z, SD::VERTEX_NW )];

            // TSE
            get_l_stencil( x + 1, y - 1, z + 1, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x + 1, y - 1, z + 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_TSE )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TSE )] * u[cidx( x, y, z, SD::VERTEX_TSE )];

            // TS
            get_l_stencil( x, y - 1, z + 1, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y - 1, z + 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_TS )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TS )] * u[cidx( x, y, z, SD::VERTEX_TS )];

            // TC
            get_l_stencil( x, y, z + 1, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x, y, z + 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_TC )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TC )] * u[cidx( x, y, z, SD::VERTEX_TC )];

            // TW
            get_l_stencil( x - 1, y, z + 1, l_stencil );
//            if ( vertexdof::macrocell::isVertexOnBoundary( level, indexing::Index( x - 1, y, z + 1 ) ) )
//               WALBERLA_CHECK_FLOAT_EQUAL( l_stencil[opposite( SD::VERTEX_TW )], 0. );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TW )] * u[cidx( x, y, z, SD::VERTEX_TW )];
         }
      }
   }
}

class LDLTBoundaryStencils
{
 public:
   using SD = stencilDirection;

   using StencilType = std::map< SD, real_t >;

   void add( uint_t x, uint_t y, uint_t z, StencilType& stencil )
   {
      if ( stencils_.count( x ) == 0 )
         stencils_[x] = {};
      if ( stencils_[x].count( y ) == 0 )
         stencils_[x][y] = {};
      if ( stencils_[x][y].count( z ) != 0 )
         WALBERLA_LOG_WARNING( "boundary stencils gets added twice" )
      stencils_[x][y][z] = stencil;
   }

   StencilType& get( uint_t x, uint_t y, uint_t z )
   {
      if ( stencils_.count( x ) == 0 || stencils_[x].count( y ) == 0 || stencils_[x][y].count( z ) == 0 )
         WALBERLA_ABORT( "stencil not found" );
      return stencils_[x][y][z];
   }

 private:
   // x, y, z, direction
   std::map< uint_t, std::map< uint_t, std::map< uint_t, std::map< SD, real_t > > > > stencils_;
};

class LDLTHierarchicalBoundaryStencils
{
 public:
   LDLTHierarchicalBoundaryStencils( uint_t minLevel, uint_t maxLevel )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , collections( maxLevel - minLevel + 1 )
   {}

   LDLTBoundaryStencils& getLevel( uint_t level )
   {
      WALBERLA_CHECK_LESS_EQUAL( minLevel_, level );
      WALBERLA_CHECK_GREATER_EQUAL( maxLevel_, level );
      return collections[level - minLevel_];
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;

   std::vector< LDLTBoundaryStencils > collections;
};

class LDLTHierarchicalBoundaryStencilsMemoryDataHandling
: public hyteg::OnlyInitializeDataHandling< LDLTHierarchicalBoundaryStencils, Cell >
{
 public:
   explicit LDLTHierarchicalBoundaryStencilsMemoryDataHandling( const uint_t& minLevel, const uint_t& maxLevel )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   {}

   std::shared_ptr< LDLTHierarchicalBoundaryStencils > initialize( const Cell* const ) const override
   {
      return std::make_shared< LDLTHierarchicalBoundaryStencils >( minLevel_, maxLevel_ );
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;
};

class LDLTPolynomials
{
 public:
   using Basis      = QuadrilateralBasis3D;
   using Polynomial = QuadrilateralPolynomial3D;

   LDLTPolynomials( uint_t degreeX, uint_t degreeY, uint_t degreeZ )
   : basis_( degreeX, degreeY, degreeZ )
   {
      for ( auto d : lowerDirectionsAndCenter )
         polynomials_.emplace( d, basis_ );
   }

   inline Polynomial& getPolynomial( stencilDirection direction ) { return polynomials_.at( direction ); }

   inline const Polynomial& operator[]( stencilDirection direction ) const { return polynomials_.at( direction ); }

   [[nodiscard]] inline std::array< uint_t, 3 > getDegrees() const { return basis_.getDegrees(); }

 private:
   Basis basis_;

   mutable std::map< SD, Polynomial > polynomials_;
};

class LDLTHierachicalPolynomials
{
 public:
   LDLTHierachicalPolynomials( uint_t minLevel, uint_t maxLevel, uint_t degreeX, uint_t degreeY, uint_t degreeZ )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , collections( maxLevel - minLevel + 1, LDLTPolynomials( degreeX, degreeY, degreeZ ) )
   {}

   LDLTPolynomials& getLevel( uint_t level )
   {
      WALBERLA_CHECK_LESS_EQUAL( minLevel_, level );
      WALBERLA_CHECK_GREATER_EQUAL( maxLevel_, level );
      return collections[level - minLevel_];
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;

   std::vector< LDLTPolynomials > collections;
};

class LDLTHierachicalPolynomialsDataHandling : public hyteg::OnlyInitializeDataHandling< LDLTHierachicalPolynomials, Cell >
{
 public:
   explicit LDLTHierachicalPolynomialsDataHandling( const uint_t& minLevel,
                                                    const uint_t& maxLevel,
                                                    const uint_t& degreeX,
                                                    const uint_t& degreeY,
                                                    const uint_t& degreeZ )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , degreeX_( degreeX )
   , degreeY_( degreeY )
   , degreeZ_( degreeZ )
   {}

   std::shared_ptr< LDLTHierachicalPolynomials > initialize( const Cell* const ) const override
   {
      auto collection = std::make_shared< LDLTHierachicalPolynomials >( minLevel_, maxLevel_, degreeX_, degreeY_, degreeZ_ );
      return collection;
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;
   uint_t degreeX_;
   uint_t degreeY_;
   uint_t degreeZ_;
};

class Interpolators
{
 public:
   using Basis          = QuadrilateralBasis3D;
   using Polynomial     = QuadrilateralPolynomial< 3, Point3D, Basis >;
   using Interpolator3D = VariableQuadrilateralLSQPInterpolator< QuadrilateralBasis3D, Polynomial, Point3D >;

   Interpolators( uint_t degreeX, uint_t degreeY, uint_t degreeZ )
   {
      Basis basis( degreeX, degreeY, degreeZ );
      for ( auto d : lowerDirectionsAndCenter )
         interpolators.emplace( d, basis );
   }

   Interpolator3D& operator()( SD direction ) { return interpolators.at( direction ); }

   void addStencil( const Point3D& p, const std::map< SD, real_t >& stencil )
   {
      for ( auto d : lowerDirectionsAndCenter )
         interpolators.at( d ).addInterpolationPoint( p, stencil.at( d ) );
   }

   void addValue( const Point3D& p, SD d, real_t v ) { interpolators.at( d ).addInterpolationPoint( p, v ); }

   void interpolate( LDLTPolynomials& poly )
   {
      for ( auto d : lowerDirectionsAndCenter )
         interpolators.at( d ).interpolate( poly.getPolynomial( d ) );
   }

 private:
   std::map< SD, Interpolator3D > interpolators;
};

template < size_t num_directions >
class PolyStencil
{
 public:
   PolyStencil( const std::array< uint_t, 3 > degrees, const std::array< SD, num_directions >& directions )
   : directions_( directions )
   {
      for ( uint_t i = 0; i < num_directions; i += 1 )
      {
         polynomials_.emplace_back( degrees[0], degrees[1], degrees[2] );
         offsetsX_.push_back( 0 );
         offsetsY_.push_back( 0 );
         offsetsZ_.push_back( 0 );
      }
   }

   template < typename PolyListType >
   void setPolynomial( PolyListType& polylist )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         polynomials_[i].setPolynomial( polylist[directions_[i]] );
      }
   }

   void setY( real_t y )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         polynomials_[i].setY( y + offsetsY_[i] );
   }

   void setZ( real_t z )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         polynomials_[i].setZ( z + offsetsZ_[i] );
   }

   void setStartX( real_t x, real_t h, std::map< SD, real_t >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[directions_[i]] = polynomials_[i].setStartX( x + offsetsX_[i], h );
   }

   void incrementEval( std::map< SD, real_t >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[directions_[i]] = polynomials_[i].incrementEval();
   };

   void setOffset( SD d, real_t x, real_t y, real_t z )
   {
      auto it = std::find( directions_.begin(), directions_.end(), d );
      if ( it == directions_.end() )
         WALBERLA_ABORT( "direction not defined" );
      auto idx       = it - directions_.begin();
      offsetsX_[idx] = x;
      offsetsY_[idx] = y;
      offsetsZ_[idx] = z;
   }

 private:
   std::array< SD, num_directions > directions_;

   std::vector< QuadrilateralPolynomial3DEvaluator > polynomials_;

   std::vector< real_t > offsetsX_;
   std::vector< real_t > offsetsY_;
   std::vector< real_t > offsetsZ_;
};

void compare_stencil_vs_polynomial( uint_t                  x,
                                    uint_t                  y,
                                    uint_t                  z,
                                    real_t                  h,
                                    std::map< SD, real_t >& stencil,
                                    LDLTPolynomials&        polynomials,
                                    real_t                  tol )
{
   Point3D p( {
       h * static_cast< real_t >( x ),
       h * static_cast< real_t >( y ),
       h * static_cast< real_t >( z ),
   } );
   for ( auto d : lowerDirections )
   {
      double diff = std::abs( stencil[d] - polynomials.getPolynomial( d ).eval( p ) );
      if ( diff > tol )
         WALBERLA_LOG_INFO( x << " " << y << " " << z << " " << diff << " too large difference" )
   }
}

Point3D to_point( uint_t x, uint_t y, uint_t z, real_t h )
{
   Point3D p( {
       h * static_cast< real_t >( x ),
       h * static_cast< real_t >( y ),
       h * static_cast< real_t >( z ),
   } );
   return p;
}

template < typename FunctionType, bool useBoundaryCorrection = false, bool useMatrixBoundaryValuesInInnerRegion = false >
void apply_surrogate_substitutions( LDLTBoundaryStencils& boundaryStencils,
                                    LDLTPolynomials&      polynomials,
                                    uint_t                level,
                                    Cell&                 cell,
                                    const FunctionType&   u_function,
                                    const FunctionType&   b_function )
{
   const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
      return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
   };

   auto get_l_stencil = [&boundaryStencils, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
      stencil = boundaryStencils.get( x, y, z );
   };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   real_t h = 1. / static_cast< real_t >( levelinfo::num_microedges_per_edge( level ) );

   // unpack u and b
   auto u = cell.getData( u_function.getCellDataID() )->getPointer( level );
   auto b = cell.getData( b_function.getCellDataID() )->getPointer( level );

   const size_t boundarySize = 1;

   auto apply_forward_substitution =
       [cidx]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil, real_t const* const b_dat, real_t* u_dat ) {
          u_dat[cidx( x, y, z, SD::VERTEX_C )] = b_dat[cidx( x, y, z, SD::VERTEX_C )];
          for ( auto d : lowerDirections )
             u_dat[cidx( x, y, z, SD::VERTEX_C )] -= stencil[d] * u_dat[cidx( x, y, z, d )];
       };

   auto               conv = []( uint_t x, uint_t y, uint_t z ) { return 1000000 * z + 1000 * y + x; };
   std::set< uint_t > contains;

   auto apply_diagonal_scaling =
       [cidx, &conv, &contains](
           uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil, real_t const* const, real_t* u_dat ) {
          u_dat[cidx( x, y, z, SD::VERTEX_C )] /= stencil[SD::VERTEX_C];
          if ( contains.find( conv( x, y, z ) ) != contains.end() )
             WALBERLA_LOG_INFO( x << " " << y << " " << z << " twice!" );
          contains.insert( conv( x, y, z ) );
       };

   auto get_lt_stencil =
       [&boundaryStencils, &polynomials, N_edge, h, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
          if ( ldlt::p1::dim3::on_cell_boundary( x + 1, y, z, 2, N_edge ) )
             stencil[SD::VERTEX_W] = boundaryStencils.get( x + 1, y, z )[SD::VERTEX_W];
          else
             stencil[SD::VERTEX_W] = polynomials.getPolynomial( SD::VERTEX_W ).eval( to_point( x + 1, y, z, h ) );

          if ( ldlt::p1::dim3::on_cell_boundary( x, y + 1, z, 2, N_edge ) )
             stencil[SD::VERTEX_S] = boundaryStencils.get( x, y + 1, z )[SD::VERTEX_S];
          else
             stencil[SD::VERTEX_S] = polynomials.getPolynomial( SD::VERTEX_S ).eval( to_point( x, y + 1, z, h ) );

          if ( ldlt::p1::dim3::on_cell_boundary( x - 1, y + 1, z, 2, N_edge ) )
             stencil[SD::VERTEX_SE] = boundaryStencils.get( x - 1, y + 1, z )[SD::VERTEX_SE];
          else
             stencil[SD::VERTEX_SE] = polynomials.getPolynomial( SD::VERTEX_SE ).eval( to_point( x - 1, y + 1, z, h ) );

          if ( ldlt::p1::dim3::on_cell_boundary( x + 1, y - 1, z + 1, 2, N_edge ) )
             stencil[SD::VERTEX_BNW] = boundaryStencils.get( x + 1, y - 1, z + 1 )[SD::VERTEX_BNW];
          else
             stencil[SD::VERTEX_BNW] = polynomials.getPolynomial( SD::VERTEX_BNW ).eval( to_point( x + 1, y - 1, z + 1, h ) );

          if ( ldlt::p1::dim3::on_cell_boundary( x, y - 1, z + 1, 2, N_edge ) )
             stencil[SD::VERTEX_BN] = boundaryStencils.get( x, y - 1, z + 1 )[SD::VERTEX_BN];
          else
             stencil[SD::VERTEX_BN] = polynomials.getPolynomial( SD::VERTEX_BN ).eval( to_point( x, y - 1, z + 1, h ) );

          if ( ldlt::p1::dim3::on_cell_boundary( x, y, z + 1, 2, N_edge ) )
             stencil[SD::VERTEX_BC] = boundaryStencils.get( x, y, z + 1 )[SD::VERTEX_BC];
          else
             stencil[SD::VERTEX_BC] = polynomials.getPolynomial( SD::VERTEX_BC ).eval( to_point( x, y, z + 1, h ) );

          if ( ldlt::p1::dim3::on_cell_boundary( x - 1, y, z + 1, 2, N_edge ) )
             stencil[SD::VERTEX_BE] = boundaryStencils.get( x - 1, y, z + 1 )[SD::VERTEX_BE];
          else
             stencil[SD::VERTEX_BE] = polynomials.getPolynomial( SD::VERTEX_BE ).eval( to_point( x - 1, y, z + 1, h ) );
       };

   /*
   auto get_lt_stencil_poly = [&polynomials, level, h]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
      stencil[SD::VERTEX_W]   = polynomials.getPolynomial( SD::VERTEX_W ).eval( to_point( x + 1, y, z, h ) );
      stencil[SD::VERTEX_S]   = polynomials.getPolynomial( SD::VERTEX_S ).eval( to_point( x, y + 1, z, h ) );
      stencil[SD::VERTEX_SE]  = polynomials.getPolynomial( SD::VERTEX_SE ).eval( to_point( x - 1, y + 1, z, h ) );
      stencil[SD::VERTEX_BNW] = polynomials.getPolynomial( SD::VERTEX_BNW ).eval( to_point( x + 1, y - 1, z + 1, h ) );
      stencil[SD::VERTEX_BN]  = polynomials.getPolynomial( SD::VERTEX_BN ).eval( to_point( x, y - 1, z + 1, h ) );
      stencil[SD::VERTEX_BC]  = polynomials.getPolynomial( SD::VERTEX_BC ).eval( to_point( x, y, z + 1, h ) );
      stencil[SD::VERTEX_BE]  = polynomials.getPolynomial( SD::VERTEX_BE ).eval( to_point( x - 1, y, z + 1, h ) );
   };
   */

   auto apply_backward_substitution =
       [cidx]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil, real_t const* const, real_t* u_dat ) {
          for ( auto d : upperDirections )
             u_dat[cidx( x, y, z, SD::VERTEX_C )] -= stencil[opposite( d )] * u_dat[cidx( x, y, z, d )];
       };

   // ---------------------
   // forward substitution:
   // ---------------------
   {
      PolyStencil< 7 > poly_stencil_lower( polynomials.getDegrees(), lowerDirections );
      poly_stencil_lower.setPolynomial( polynomials );

      std::map< SD, real_t > l_stencil;

      // z bottom:
      for ( uint_t z = 1; z < 1 + boundarySize; z += 1 )
      {
         poly_stencil_lower.setZ( h * static_cast< real_t >( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z inner:
      for ( uint_t z = 1 + boundarySize; z <= N_edge - 2 - boundarySize; z += 1 )
      {
         poly_stencil_lower.setZ( h * static_cast< real_t >( z ) );
         // y south:
         for ( uint_t y = 1; y < 1 + boundarySize; y += 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }
         }

         // y inner:
         for ( uint_t y = 1 + boundarySize; y <= N_edge - 2 - boundarySize - z; y += 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            // x west:
            for ( uint_t x = 1; x < 1 + boundarySize; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }

            // x inner:
            for ( uint_t x = 1 + boundarySize; x <= N_edge - 2 - boundarySize - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }

            // x east:
            for ( uint_t x = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z - y ); x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }
         }

         // y north:
         for ( uint_t y = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z ); y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z bottom:
      for ( uint_t z = std::max( 1 + boundarySize, N_edge - 1 - boundarySize ); z <= N_edge - 2; z += 1 )
      {
         poly_stencil_lower.setZ( h * static_cast< real_t >( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }
   }

   // ---------
   // diagonal:
   // ---------
   {
      PolyStencil< 1 > poly_stencil_diagonal( polynomials.getDegrees(), { SD::VERTEX_C } );
      poly_stencil_diagonal.setPolynomial( polynomials );

      std::map< SD, real_t > l_stencil;

      // z bottom:
      for ( uint_t z = 1; z < 1 + boundarySize; z += 1 )
      {
         poly_stencil_diagonal.setZ( h * static_cast< real_t >( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_diagonal.setY( h * static_cast< real_t >( y ) );
            poly_stencil_diagonal.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( !useBoundaryCorrection )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z inner:
      for ( uint_t z = 1 + boundarySize; z <= N_edge - 2 - boundarySize; z += 1 )
      {
         poly_stencil_diagonal.setZ( h * static_cast< real_t >( z ) );
         // y south:
         for ( uint_t y = 1; y < 1 + boundarySize; y += 1 )
         {
            poly_stencil_diagonal.setY( h * static_cast< real_t >( y ) );
            poly_stencil_diagonal.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( !useBoundaryCorrection )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }
         }

         // y inner:
         for ( uint_t y = 1 + boundarySize; y <= N_edge - 2 - boundarySize - z; y += 1 )
         {
            poly_stencil_diagonal.setY( h * static_cast< real_t >( y ) );
            poly_stencil_diagonal.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            // x west:
            for ( uint_t x = 1; x < 1 + boundarySize; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( !useBoundaryCorrection )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }

            // x inner:
            for ( uint_t x = 1 + boundarySize; x <= N_edge - 2 - boundarySize - z - y; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }

            // x east:
            for ( uint_t x = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z - y ); x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( !useBoundaryCorrection )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }
         }

         // y north:
         for ( uint_t y = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z ); y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_diagonal.setY( h * static_cast< real_t >( y ) );
            poly_stencil_diagonal.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( !useBoundaryCorrection )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z bottom:
      for ( uint_t z = std::max( 1 + boundarySize, N_edge - 1 - boundarySize ); z <= N_edge - 2; z += 1 )
      {
         poly_stencil_diagonal.setZ( h * static_cast< real_t >( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_diagonal.setY( h * static_cast< real_t >( y ) );
            poly_stencil_diagonal.setStartX( h * static_cast< real_t >( 1 - 1 ), h, l_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               poly_stencil_diagonal.incrementEval( l_stencil );
               if ( !useBoundaryCorrection )
                  get_l_stencil( x, y, z, l_stencil );
               apply_diagonal_scaling( x, y, z, l_stencil, b, u );
            }
         }
      }
   }

   // ----------------------
   // backward substitution:
   // ----------------------
   {
      PolyStencil< 7 > poly_stencil_lower( polynomials.getDegrees(), lowerDirections );
      poly_stencil_lower.setPolynomial( polynomials );
      poly_stencil_lower.setOffset( SD::VERTEX_W, h, 0, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_S, 0, h, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_SE, -h, h, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_BNW, +h, -h, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BN, 0, -h, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BC, 0, 0, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BE, -h, 0, +h );

      std::map< SD, real_t > l_stencil;
      std::map< SD, real_t > l_stencil_test;

      // z top
      for ( uint_t z = N_edge - 2; z > N_edge - 2 - boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * static_cast< real_t >( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z inner
      for ( uint_t z = N_edge - 2 - boundarySize; z >= 1 + boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * static_cast< real_t >( z ) );
         // y north:
         for ( uint_t y = N_edge - 2 - z; y > N_edge - 2 - boundarySize - z; y -= 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }

         // y inner:
         for ( uint_t y = N_edge - 2 - boundarySize - z; y >= 1 + boundarySize; y -= 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            // x west:
            for ( uint_t x = N_edge - 2 - z - y; x > N_edge - 2 - boundarySize - z - y; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }

            // x inner:
            for ( uint_t x = N_edge - 2 - boundarySize - z - y; x >= 1 + boundarySize; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }

            // x east:
            for ( uint_t x = std::min( boundarySize, N_edge - 2 - boundarySize - z - y ); x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }

         // y south:
         for ( uint_t y = std::min( boundarySize, N_edge - 2 - boundarySize - z ); y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z bottom
      for ( uint_t z = std::min( boundarySize, N_edge - 3 - boundarySize ); z >= 1; z -= 1 )
      {
         poly_stencil_lower.setZ( h * static_cast< real_t >( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * static_cast< real_t >( y ) );
            poly_stencil_lower.setStartX( h * static_cast< real_t >( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               // poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
                  get_lt_stencil( x, y, z, l_stencil );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }
   }
}

} // namespace dim3
} // namespace p1
} // namespace ldlt

template < class OperatorType, class FormType >
class P1LDLTSurrogateCellSmoother : public CellSmoother< OperatorType >
{
 public:
   using FunctionType = typename OperatorType::srcType;

   P1LDLTSurrogateCellSmoother( std::shared_ptr< PrimitiveStorage > storage,
                                uint_t                              minLevel,
                                uint_t                              maxLevel,
                                uint_t                              degreeX,
                                uint_t                              degreeY,
                                uint_t                              degreeZ,
                                FormType                            form )
   : storage_( std::move( storage ) )
   , tmp1_( "tmp", storage_, minLevel, maxLevel )
   , tmp2_( "tmp", storage_, minLevel, maxLevel )
   , form_( form )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , polyDegreeX_( degreeX )
   , polyDegreeY_( degreeY )
   , polyDegreeZ_( degreeZ )
   {
      // storage for surrogate operator
      auto polyDataHandling = std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >(
          minLevel_, maxLevel_, polyDegreeX_, polyDegreeY_, polyDegreeZ_ );
      storage_->addCellData( polynomialsID_, polyDataHandling, "P1LDLTSurrogateCellSmootherPolynomials" );

      // storage for the boundary stencils
      auto boundaryDataHandling =
          std::make_shared< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencilsMemoryDataHandling >( minLevel_, maxLevel_ );
      storage_->addCellData( boundaryID_, boundaryDataHandling, "P1LDLTSurrogateCellSmootherBoundary" );
   }

   using SD = stencilDirection;

   static constexpr auto cindex = vertexdof::macrocell::indexFromVertex;

   void init( uint_t skipLevel )
   {
      for ( auto& it : storage_->getCells() )
      {
         Cell& cell = *it.second;
         for ( uint_t level = minLevel_; level <= maxLevel_; ++level )
         {
            factorize_matrix_inplace( level, cell, form_, skipLevel );
         }
      }
   }

   const static bool useBoundaryCorrection = false;

   void preSmooth( const OperatorType&                   A,
                   uint_t                                level,
                   const typename OperatorType::srcType& u,
                   const typename OperatorType::dstType& b ) override
   {
      tmp1_.assign( { 1. }, { u }, level, DirichletBoundary );
      A.apply( u, tmp1_, level, flag_ );
      tmp1_.assign( { 1., -1. }, { b, tmp1_ }, level, flag_ );

      tmp1_.template communicate< Vertex, Edge >( level );
      tmp1_.template communicate< Edge, Face >( level );
      tmp1_.template communicate< Face, Cell >( level );
   }

   void postSmooth( const OperatorType&,
                    uint_t                                level,
                    const typename OperatorType::srcType& u,
                    const typename OperatorType::dstType& ) override
   {
      u.assign( { 1., 1. }, { tmp2_, u }, level, flag_ );
   }

   void smooth( const OperatorType&                   A,
                uint_t                                level,
                Cell&                                 cell,
                const typename OperatorType::srcType&,
                const typename OperatorType::dstType& ) override
   {
      smooth_apply( A, level, cell, tmp2_, tmp1_ );
   }

   void factorize_matrix_inplace( uint_t level, Cell& cell, FormType& form, uint_t skipLevel )
   {
      auto& polynomials  = cell.getData( polynomialsID_ )->getLevel( level );
      auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );

      real_t h          = 1. / static_cast< real_t >( levelinfo::num_microedges_per_edge( level ) );
      real_t H          = 1. / static_cast< real_t >( levelinfo::num_microedges_per_edge( skipLevel ) );
      auto   skipLength = static_cast< uint_t >( std::max( 1., std::round( H / h ) ) );

      auto is_interpolation_point = [skipLength, this]( uint_t x, uint_t y, uint_t z, uint_t offx, uint_t offy, uint_t offz ) {
         auto x_b = x - 1 - offx;
         auto y_b = y - 1 - offy;
         auto z_b = z - 1 - offz;
         return ( x_b % skipLength == 0 ) && ( y_b % skipLength == 0 ) && ( z_b % skipLength == 0 );
      };

      ldlt::p1::dim3::Interpolators interpolators( polyDegreeX_, polyDegreeY_, polyDegreeZ_ );

      // initialize boundary data:
      std::map< SD, real_t > unit_stencil;
      for ( auto d : ldlt::p1::dim3::lowerDirections )
         unit_stencil[d] = 0;
      unit_stencil[SD::VERTEX_C] = 1;

      const uint_t N_edge = levelinfo::num_microvertices_per_edge( level );

      for ( uint_t z = 0; z <= N_edge - 1; z += 1 )
      {
         for ( uint_t y = 0; y <= N_edge - 1 - z; y += 1 )
         {
            for ( uint_t x = 0; x <= N_edge - 1 - z - y; x += 1 )
            {
               if ( ldlt::p1::dim3::on_cell_boundary( x, y, z, 1, N_edge ) )
                  boundaryData.add( x, y, z, unit_stencil );
            }
         }
      }

      auto factorization = [&boundaryData, level, N_edge, is_interpolation_point, h, &interpolators](
                               uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
         Point3D p( { h * static_cast< real_t >( x ), h * static_cast< real_t >( y ), h * static_cast< real_t >( z ) } );

         if ( is_interpolation_point( x, y, z, 0, 0, 0 ) )
            interpolators.addValue( p, SD::VERTEX_C, stencil[SD::VERTEX_C] );
         if ( ( !ldlt::p1::dim3::on_west_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 1, 0, 0 ) )
            interpolators.addValue( p, SD::VERTEX_W, stencil[SD::VERTEX_W] );
         if ( ( !ldlt::p1::dim3::on_south_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 0, 1, 0 ) )
            interpolators.addValue( p, SD::VERTEX_S, stencil[SD::VERTEX_S] );
         if ( ( !ldlt::p1::dim3::on_south_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 0, 1, 0 ) )
            interpolators.addValue( p, SD::VERTEX_SE, stencil[SD::VERTEX_SE] );
         if ( ( !ldlt::p1::dim3::on_west_boundary( x, y, z, N_edge ) ) &&
              ( !ldlt::p1::dim3::on_bottom_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 1, 0, 1 ) )
            interpolators.addValue( p, SD::VERTEX_BNW, stencil[SD::VERTEX_BNW] );
         if ( ( !ldlt::p1::dim3::on_bottom_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 0, 0, 1 ) )
            interpolators.addValue( p, SD::VERTEX_BN, stencil[SD::VERTEX_BN] );
         if ( ( !ldlt::p1::dim3::on_bottom_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 0, 0, 1 ) )
            interpolators.addValue( p, SD::VERTEX_BC, stencil[SD::VERTEX_BC] );
         if ( ( !ldlt::p1::dim3::on_bottom_boundary( x, y, z, N_edge ) ) && is_interpolation_point( x, y, z, 0, 0, 1 ) )
            interpolators.addValue( p, SD::VERTEX_BE, stencil[SD::VERTEX_BE] );

         if ( !useBoundaryCorrection )
         {
            if ( ldlt::p1::dim3::on_cell_boundary( x, y, z, 2, N_edge ) )
            {
               boundaryData.add( x, y, z, stencil );
            }
         }
      };

      ldlt::p1::dim3::factorize_matrix( form, level, cell, factorization );

      interpolators.interpolate( polynomials );
   }

   void smooth_apply( const OperatorType&,
                      uint_t                                level,
                      Cell&                                 cell,
                      const typename OperatorType::srcType& u,
                      const typename OperatorType::dstType& b )
   {
      auto& polynomialData = cell.getData( polynomialsID_ )->getLevel( level );
      auto& boundaryData   = cell.getData( boundaryID_ )->getLevel( level );

      ldlt::p1::dim3::apply_surrogate_substitutions< typename OperatorType::srcType, useBoundaryCorrection >(
          boundaryData, polynomialData, level, cell, u, b );
   }

   void smooth_backwards( const OperatorType&                   A,
                          uint_t                                level,
                          Cell&                                 cell,
                          const typename OperatorType::srcType& x,
                          const typename OperatorType::dstType& b ) override
   {
      smooth( A, level, cell, x, b );
   }

   void interpolate_stencil_direction( uint_t level, SD direction, const FunctionType& u )
   {
      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      const real_t h = 1. / static_cast< real_t >( N_edge );

      constexpr auto cidx = vertexdof::macrocell::indexFromVertex;

      std::map< SD, real_t > l_stencil;

      for ( auto it : storage_->getCells() )
      {
         Cell& cell           = *it.second;
         auto& boundaryData   = cell.getData( boundaryID_ )->getLevel( level );
         auto& polynomialData = cell.getData( polynomialsID_ )->getLevel( level );
         auto& polynomial     = polynomialData.getPolynomial( direction );
         auto  u_data         = cell.getData( u.getCellDataID() )->getPointer( level );

         for ( uint_t z = 0; z <= N_edge - 1; z += 1 )
         {
            for ( uint_t y = 0; y <= N_edge - 1 - z; y += 1 )
            {
               for ( uint_t x = 0; x <= N_edge - 1 - z - y; x += 1 )
               {
                  real_t polynomialValue = polynomial.eval( ldlt::p1::dim3::to_point( x, y, z, h ) );
                  u_data[cidx( level, x, y, z, SD::VERTEX_C )] =
                      ldlt::p1::dim3::apply_boundary_corrections_to_scalar( x, y, z, N_edge, direction, polynomialValue );
                  if ( !useBoundaryCorrection && ldlt::p1::dim3::on_cell_boundary( x, y, z, 2, N_edge ) )
                     u_data[cidx( level, x, y, z, SD::VERTEX_C )] = boundaryData.get( x, y, z )[direction];
               }
            }
         }
      }
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;

   FunctionType tmp1_;
   FunctionType tmp2_;

   FormType form_;

   DoFType flag_;

   uint_t minLevel_;
   uint_t maxLevel_;

   uint_t polyDegreeX_;
   uint_t polyDegreeY_;
   uint_t polyDegreeZ_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencils, Cell > boundaryID_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierachicalPolynomials, Cell > polynomialsID_;
};

template < class OperatorType, class FormType >
class P1LDLTInplaceCellSmoother : public CellSmoother< OperatorType >
{
 public:
   using FunctionType = typename OperatorType::srcType;

   P1LDLTInplaceCellSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel, FormType form )
   : storage_( std::move( storage ) )
   , tmp1_( "tmp", storage_, minLevel, maxLevel )
   , tmp2_( "tmp", storage_, minLevel, maxLevel )
   , form_( form )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   {
      auto boundaryDataHandling =
          std::make_shared< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencilsMemoryDataHandling >( minLevel_, maxLevel_ );
      storage_->addCellData( boundaryID_, boundaryDataHandling, "P1LDLTInplaceCellSmootherBoundary" );
   }

   using SD = stencilDirection;

   static constexpr auto cindex = vertexdof::macrocell::indexFromVertex;

   void init()
   {
      for ( auto& it : storage_->getCells() )
      {
         Cell& cell = *it.second;
         for ( uint_t level = minLevel_; level <= maxLevel_; ++level )
         {
            factorize_matrix_inplace( level, cell, form_ );
         }
      }
   }

   void preSmooth( const OperatorType&                   A,
                   uint_t                                level,
                   const typename OperatorType::srcType& u,
                   const typename OperatorType::dstType& b ) override
   {
      tmp1_.assign( { 1. }, { u }, level, DirichletBoundary );
      A.apply( u, tmp1_, level, flag_ );
      tmp1_.assign( { 1., -1. }, { b, tmp1_ }, level, flag_ );

      // tmp1_.template communicate< Vertex, Edge >( level );
      // tmp1_.template communicate< Edge, Face >( level );
      // tmp1_.template communicate< Face, Cell >( level );
   }

   void postSmooth( const OperatorType&,
                    uint_t                                level,
                    const typename OperatorType::srcType& u,
                    const typename OperatorType::dstType& ) override
   {
      u.assign( { 1., 1. }, { tmp2_, u }, level, flag_ );
   }

   void smooth( const OperatorType& A,
                uint_t              level,
                Cell&               cell,
                const typename OperatorType::srcType&,
                const typename OperatorType::dstType& ) override
   {
      smooth_apply( A, level, cell, tmp2_, tmp1_ );
   }

   void factorize_matrix_inplace( uint_t level, Cell& cell, FormType& form )
   {
      auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );

      // initialize boundary data:
      std::map< SD, real_t > unit_stencil;
      for ( auto d : ldlt::p1::dim3::lowerDirections )
         unit_stencil[d] = 0;
      unit_stencil[SD::VERTEX_C] = 1;

      const uint_t N_edge = levelinfo::num_microvertices_per_edge( level );

      for ( uint_t z = 0; z <= N_edge - 1; z += 1 )
      {
         for ( uint_t y = 0; y <= N_edge - 1 - z; y += 1 )
         {
            for ( uint_t x = 0; x <= N_edge - 1 - z - y; x += 1 )
            {
               if ( ldlt::p1::dim3::on_cell_boundary( x, y, z, 1, N_edge ) )
                  boundaryData.add( x, y, z, unit_stencil );
            }
         }
      }

      ldlt::p1::dim3::factorize_matrix(
          form, level, cell, [&boundaryData, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
             boundaryData.add( x, y, z, stencil );
          } );
   }

   void smooth_apply( const OperatorType&,
                      uint_t                                level,
                      Cell&                                 cell,
                      const typename OperatorType::srcType& u,
                      const typename OperatorType::dstType& b )
   {
      auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );

      auto stencil_provider = [&boundaryData, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
         stencil = boundaryData.get( x, y, z );
      };

      ldlt::p1::dim3::apply_substitutions( stencil_provider, level, cell, u, b );
   }

   void smooth_backwards( const OperatorType&                   A,
                          uint_t                                level,
                          Cell&                                 cell,
                          const typename OperatorType::srcType& x,
                          const typename OperatorType::dstType& b ) override
   {
      smooth( A, level, cell, x, b );
   }

   void interpolate_stencil_direction( uint_t level, SD direction, const FunctionType& u )
   {
      const auto N_edge = levelinfo::num_microvertices_per_edge( level );

      constexpr auto cidx = vertexdof::macrocell::indexFromVertex;

      std::map< SD, real_t > l_stencil;

      for ( auto it : storage_->getCells() )
      {
         Cell& cell         = *it.second;
         auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );
         auto  u_data       = cell.getData( u.getCellDataID() )->getPointer( level );

         for ( uint_t z = 0; z <= N_edge - 1; z += 1 )
         {
            for ( uint_t y = 0; y <= N_edge - 1 - z; y += 1 )
            {
               for ( uint_t x = 0; x <= N_edge - 1 - z - y; x += 1 )
               {
                  u_data[cidx( level, x, y, z, SD::VERTEX_C )] = boundaryData.get( x, y, z )[direction];
               }
            }
         }
      }
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;

   FunctionType tmp1_;
   FunctionType tmp2_;

   FormType form_;

   DoFType flag_;

   uint_t minLevel_;
   uint_t maxLevel_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencils, Cell > boundaryID_;
};

} // namespace hyteg