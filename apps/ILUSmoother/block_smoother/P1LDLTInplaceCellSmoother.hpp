#pragma once

// #include "hyteg/LikwidWrapper.hpp"
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

constexpr std::array< SD, 15 > allDirections = { SD::VERTEX_C,
                                                 SD::VERTEX_W,
                                                 SD::VERTEX_S,
                                                 SD::VERTEX_SE,
                                                 SD::VERTEX_BNW,
                                                 SD::VERTEX_BN,
                                                 SD::VERTEX_BC,
                                                 SD::VERTEX_BE,
                                                 SD::VERTEX_E,
                                                 SD::VERTEX_N,
                                                 SD::VERTEX_NW,
                                                 SD::VERTEX_TSE,
                                                 SD::VERTEX_TS,
                                                 SD::VERTEX_TC,
                                                 SD::VERTEX_TW };

indexing::IndexIncrement toIndex( SD dir )
{
   switch ( dir )
   {
   case SD::VERTEX_W:
      return { -1, 0, 0 };
   case SD::VERTEX_S:
      return { 0, -1, 0 };
   case SD::VERTEX_SE:
      return { +1, -1, 0 };
   case SD::VERTEX_BNW:
      return { -1, +1, -1 };
   case SD::VERTEX_BN:
      return { 0, +1, -1 };
   case SD::VERTEX_BC:
      return { 0, 0, -1 };
   case SD::VERTEX_BE:
      return { +1, 0, -1 };
   case SD::VERTEX_C:
      return { 0, 0, 0 };
   case SD::VERTEX_E:
      return { +1, 0, 0 };
   case SD::VERTEX_N:
      return { 0, +1, 0 };
   case SD::VERTEX_NW:
      return { -1, +1, 0 };
   case SD::VERTEX_TSE:
      return { +1, -1, +1 };
   case SD::VERTEX_TS:
      return { 0, -1, +1 };
   case SD::VERTEX_TC:
      return { 0, 0, 1 };
   case SD::VERTEX_TW:
      return { -1, 0, +1 };
   default:
      break;
   };
   WALBERLA_ABORT( "not implemented" );
}

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

void apply_boundary_corrections( uint_t x, uint_t y, uint_t z, uint_t N, std::array< real_t, 7 >& stencil )
{
   if ( x == 1 )
   {
      // stencil[stencilDirection::VERTEX_NW]  = 0;
      stencil[0] = 0;
      // stencil[stencilDirection::VERTEX_TW]  = 0;
      stencil[3] = 0;
   }
   if ( x + y + z == N - 2 )
   {
      // stencil[stencilDirection::VERTEX_E]   = 0;
      // stencil[stencilDirection::VERTEX_N]   = 0;
      // stencil[stencilDirection::VERTEX_TC]  = 0;
      // stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( y == 1 )
   {
      stencil[1] = 0;
      stencil[2] = 0;
      // stencil[stencilDirection::VERTEX_TS]  = 0;
      // stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( z == 1 )
   {
      stencil[5] = 0;
      stencil[6] = 0;
      stencil[4] = 0;
      stencil[3] = 0;
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

void apply_boundary_corrections_on_backward_stencil( uint_t x, uint_t y, uint_t z, uint_t N, std::array< real_t, 7 >& stencil )
{
   stencil[0] = apply_boundary_corrections_to_scalar( x + 1, y, z, N, SD::VERTEX_W, stencil[0] );
   stencil[1] = apply_boundary_corrections_to_scalar( x, y + 1, z, N, SD::VERTEX_S, stencil[1] );
   stencil[2] = apply_boundary_corrections_to_scalar( x - 1, y + 1, z, N, SD::VERTEX_SE, stencil[2] );
   stencil[3] = apply_boundary_corrections_to_scalar( x + 1, y - 1, z + 1, N, SD::VERTEX_BNW, stencil[3] );
   stencil[4] = apply_boundary_corrections_to_scalar( x, y - 1, z + 1, N, SD::VERTEX_BN, stencil[4] );
   stencil[5] = apply_boundary_corrections_to_scalar( x, y, z + 1, N, SD::VERTEX_BC, stencil[5] );
   stencil[6] = apply_boundary_corrections_to_scalar( x - 1, y, z + 1, N, SD::VERTEX_BE, stencil[6] );
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
            auto a_stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { x, y, z }, cell, level, form );

            apply_boundary_corrections( x, y, z, N_edge, a_stencil );

            const real_t a_bc  = a_stencil[SD::VERTEX_BC];
            const real_t a_s   = a_stencil[SD::VERTEX_S];
            const real_t a_bnw = a_stencil[SD::VERTEX_BNW];
            const real_t a_be  = a_stencil[SD::VERTEX_BE];
            const real_t a_w   = a_stencil[SD::VERTEX_W];
            const real_t a_bn  = a_stencil[SD::VERTEX_BN];
            const real_t a_se  = a_stencil[SD::VERTEX_SE];
            const real_t a_c   = a_stencil[SD::VERTEX_C];

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

   template < typename CollectionType >
   LDLTPolynomials( const std::array< uint_t, 3 >& degrees, const CollectionType& directions )
   : basis_( degrees )
   {
      for ( auto d : directions )
         polynomials_.emplace( d, basis_ );
   }

   inline Polynomial& getPolynomial( stencilDirection direction ) { return polynomials_.at( direction ); }

   inline const Polynomial& operator[]( stencilDirection direction ) const { return polynomials_.at( direction ); }

   inline bool contains( stencilDirection direction ) const { return polynomials_.count( direction ); }

   [[nodiscard]] inline std::array< uint_t, 3 > getDegrees() const { return basis_.getDegrees(); }

   void print()
   {
      for ( auto it : polynomials_ )
      {
         std::stringstream buf;
         buf << stencilDirectionToStr[it.first];
         buf << " = [";
         for ( uint_t i = 0; i < it.second.getNumCoefficients(); i += 1 )
         {
            buf << it.second.getCoefficient( i ) << ",";
         }
         buf << "]";

         WALBERLA_LOG_INFO_ON_ROOT( buf.str() );
      }
   }

 private:
   Basis basis_;

   mutable std::map< SD, Polynomial > polynomials_;
};

class LDLTHierachicalPolynomials
{
 public:
   template < typename CollectionType >
   LDLTHierachicalPolynomials( uint_t                         minLevel,
                               uint_t                         maxLevel,
                               const std::array< uint_t, 3 >& degrees,
                               const CollectionType&          directions )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , collections( maxLevel - minLevel + 1, LDLTPolynomials( degrees, directions ) )
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
   template < uint_t numDirections >
   explicit LDLTHierachicalPolynomialsDataHandling( const uint_t&                          minLevel,
                                                    const uint_t&                          maxLevel,
                                                    const std::array< uint_t, 3 >&         degrees,
                                                    const std::array< SD, numDirections >& directions )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , degrees_( degrees )
   , directions_( directions.begin(), directions.end() )
   {}

   std::shared_ptr< LDLTHierachicalPolynomials > initialize( const Cell* const ) const override
   {
      auto collection = std::make_shared< LDLTHierachicalPolynomials >( minLevel_, maxLevel_, degrees_, directions_ );
      return collection;
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;

   std::array< uint_t, 3 > degrees_;

   std::vector< SD > directions_;
};

class Interpolators
{
 public:
   using Basis          = QuadrilateralBasis3D;
   using Polynomial     = QuadrilateralPolynomial< 3, Point3D, Basis >;
   using Interpolator3D = VariableQuadrilateralLSQPInterpolator< QuadrilateralBasis3D, Polynomial, Point3D >;

   template < uint_t num_directions >
   Interpolators( std::array< uint_t, 3 > degrees, const std::array< SD, num_directions >& dir )
   : directions( dir.begin(), dir.end() )
   {
      Basis basis( degrees[0], degrees[1], degrees[2] );
      for ( auto d : directions )
         interpolators.emplace( d, basis );
   }

   Interpolators( std::array< uint_t, 3 > degrees )
   : Interpolators( degrees, lowerDirectionsAndCenter )
   {}

   Interpolator3D& operator()( SD direction ) { return interpolators.at( direction ); }

   void addStencil( const Point3D& p, const std::map< SD, real_t >& stencil )
   {
      for ( auto d : directions )
         interpolators.at( d ).addInterpolationPoint( p, stencil.at( d ) );
   }

   void addValue( const Point3D& p, SD d, real_t v ) { interpolators.at( d ).addInterpolationPoint( p, v ); }

   void interpolate( LDLTPolynomials& poly )
   {
      for ( auto d : directions )
         interpolators.at( d ).interpolate( poly.getPolynomial( d ) );
   }

   [[nodiscard]] std::vector< SD > getAllDirections() const { return directions; }

 private:
   std::vector< SD > directions;

   std::map< SD, Interpolator3D > interpolators;
};

template < size_t num_directions >
class PolyStencilNew
{
 public:
   PolyStencilNew( const std::array< uint_t, 3 > degrees, const std::array< SD, num_directions >& directions )
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

   template < typename PolyListType >
   void setPolynomialSymmetrical( PolyListType& polylist, real_t h )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         if ( polylist.contains( directions_[i] ) )
         {
            polynomials_[i].setPolynomial( polylist[directions_[i]] );
         }
         else
         {
            polynomials_[i].setPolynomial( polylist[opposite( directions_[i] )] );
            auto indexIncrement = toIndex( directions_[i] );
            offsetsX_[i]        = h * indexIncrement[0];
            offsetsY_[i]        = h * indexIncrement[1];
            offsetsZ_[i]        = h * indexIncrement[2];
         }
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

   void setStartX( real_t x, real_t h, std::array< real_t, num_directions >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[i] = polynomials_[i].setStartX( x + offsetsX_[i], h );
   }

   void incrementEval( std::array< real_t, num_directions >& stencil )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
         stencil[i] = polynomials_[i].incrementEval();
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

   template < typename PolyListType >
   void setPolynomialSymmetrical( PolyListType& polylist, real_t h )
   {
      for ( uint_t i = 0; i < directions_.size(); i += 1 )
      {
         if ( polylist.contains( directions_[i] ) )
         {
            polynomials_[i].setPolynomial( polylist[directions_[i]] );
         }
         else
         {
            polynomials_[i].setPolynomial( polylist[opposite( directions_[i] )] );
            auto indexIncrement = toIndex( directions_[i] );
            offsetsX_[i]        = h * indexIncrement[0];
            offsetsY_[i]        = h * indexIncrement[1];
            offsetsZ_[i]        = h * indexIncrement[2];
         }
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

template < typename FormType >
class AssembledStencil
{
 public:
   AssembledStencil( uint_t level, const Cell& cell, const FormType& form )
   : level_( level )
   , point_( 0, 0, 0 )
   , h_( 1. / real_c( levelinfo::num_microedges_per_edge( level ) ) )
   , cell_( cell )
   , form_( form )
   {
      form_.setGeometryMap( cell.getGeometryMap() );
   }

   void setY( real_t y ) { point_[1] = uint_c( std::round( y / h_ ) ); }

   void setZ( real_t z ) { point_[2] = uint_c( std::round( z / h_ ) ); }

   void setStartX( real_t x, real_t h, std::map< SD, real_t >& stencil )
   {
      h_        = h;
      point_[0] = uint_c( std::round( x / h_ ) );
      assemble( stencil );
   }

   void incrementEval( std::map< SD, real_t >& stencil )
   {
      point_[0] += 1;
      assemble( stencil );
   };

   void assemble( std::map< SD, real_t >& stencil )
   {
      stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( point_, cell_, level_, form_ );
   }

 private:
   uint_t level_;

   indexing::Index point_;

   real_t h_;

   const Cell cell_;

   FormType form_;
};

template < typename FormType >
class ConstantStencil
{
 public:
   ConstantStencil( uint_t level, const Cell& cell, const FormType& form )
   : level_( level )
   , cell_( cell )
   , form_( form )
   {
      form_.setGeometryMap( cell.getGeometryMap() );
      stencil_ = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 1, 1, 1 }, cell_, level_, form_ );
   }

   void setY( real_t ) {}

   void setZ( real_t ) {}

   void setStartX( real_t, real_t, std::map< SD, real_t >& stencil ) { assemble( stencil ); }

   void incrementEval( std::map< SD, real_t >& stencil ) { assemble( stencil ); };

   void assemble( std::map< SD, real_t >& stencil ) { stencil = stencil_; }

 private:
   uint_t level_;

   const Cell cell_;

   FormType form_;

   std::map< SD, real_t > stencil_;
};

template < typename FormType, uint_t StencilSize >
class ConstantStencilNew
{
 public:
   ConstantStencilNew( uint_t level, const Cell& cell, const FormType& form, const std::array< SD, StencilSize >& directions )
   : level_( level )
   , cell_( cell )
   , form_( form )
   {
      form_.setGeometryMap( cell.getGeometryMap() );
      auto stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new( { 1, 1, 1 }, cell_, level_, form_ );
      for ( uint_t i = 0; i < directions.size(); i += 1 )
      {
         stencil_[i] = stencil[directions[i]];
      }
   }

   void setY( real_t ) {}

   void setZ( real_t ) {}

   void setStartX( real_t, real_t, std::array< real_t, StencilSize >& stencil ) { assemble( stencil ); }

   void incrementEval( std::array< real_t, StencilSize >& stencil ) { assemble( stencil ); };

   void assemble( std::array< real_t, StencilSize >& stencil ) { stencil = stencil_; }

 private:
   uint_t level_;

   const Cell cell_;

   FormType form_;

   std::array< real_t, StencilSize > stencil_;
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
       h * real_c( x ),
       h * real_c( y ),
       h * real_c( z ),
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
       h * real_c( x ),
       h * real_c( y ),
       h * real_c( z ),
   } );
   return p;
}

template < typename P1Form >
void assemble_surrogate_operator( P1Form& form, const Cell& cell, uint_t coarse_level, uint_t fine_level, Interpolators& inter )
{
   coarse_level = std::min( coarse_level, fine_level );

   const auto N_edge = levelinfo::num_microvertices_per_edge( coarse_level );

   const uint_t level_difference = ( 2 << fine_level ) / ( 2 << coarse_level );

   const real_t h = 1. / levelinfo::num_microedges_per_edge( fine_level );

   form.setGeometryMap( cell.getGeometryMap() );

   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            uint_t x_tilde = ( x - 1 ) * level_difference + 1;
            uint_t y_tilde = ( y - 1 ) * level_difference + 1;
            uint_t z_tilde = ( z - 1 ) * level_difference + 1;

            auto a_stencil = P1Elements::P1Elements3D::calculateStencilInMacroCellForm_new(
                { x_tilde, y_tilde, z_tilde }, cell, fine_level, form );

            Point3D p( { h * real_c( x_tilde ), h * real_c( y_tilde ), h * real_c( z_tilde ) } );

            for ( auto d : inter.getAllDirections() )
               inter.addValue( p, d, a_stencil[d] );
         }
      }
   }
}

template < typename FunctionType >
void apply_surrogate_operator( LDLTPolynomials&    polynomials,
                               uint_t              level,
                               bool                useSymmetry,
                               Cell&               cell,
                               const FunctionType& src_function,
                               const FunctionType& dst_function )
{
   const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
      return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
   };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

   // unpack u and b
   auto src = cell.getData( src_function.getCellDataID() )->getPointer( level );
   auto dst = cell.getData( dst_function.getCellDataID() )->getPointer( level );

   PolyStencil< 15 > poly_stencil( polynomials.getDegrees(), allDirections );
   if ( useSymmetry )
      poly_stencil.setPolynomialSymmetrical( polynomials, h );
   else
      poly_stencil.setPolynomial( polynomials );

   std::map< SD, real_t > a_stencil;

   for ( uint_t z = 1; z <= N_edge - 2; z += 1 )
   {
      poly_stencil.setZ( h * real_c( z ) );
      for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
      {
         poly_stencil.setY( h * real_c( y ) );
         poly_stencil.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
         for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
         {
            poly_stencil.incrementEval( a_stencil );

            dst[cidx( x, y, z, SD::VERTEX_C )] = 0;
            for ( auto d : allDirections )
               dst[cidx( x, y, z, SD::VERTEX_C )] += a_stencil[d] * src[cidx( x, y, z, d )];
         }
      }
   }
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

   auto get_d_stencil = [&boundaryStencils, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
      stencil = boundaryStencils.get( x, y, z );
   };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

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

   auto apply_diagonal_scaling =
       [cidx]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil, real_t const* const, real_t* u_dat ) {
          u_dat[cidx( x, y, z, SD::VERTEX_C )] *= stencil[SD::VERTEX_C];
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
         poly_stencil_lower.setZ( h * real_c( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
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
         poly_stencil_lower.setZ( h * real_c( z ) );
         // y south:
         for ( uint_t y = 1; y < 1 + boundarySize; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
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
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
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

            // LIKWID_MARKER_START( "forward_inner" );
            // x inner:
            for ( uint_t x = 1 + boundarySize; x <= N_edge - 2 - boundarySize - z - y; x += 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, b, u );
            }
            // LIKWID_MARKER_STOP( "forward_inner" );

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
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
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
         poly_stencil_lower.setZ( h * real_c( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
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

   // ---------------------------------
   // diagonal & backward substitution:
   // ---------------------------------
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

      PolyStencil< 1 > poly_stencil_diagonal( polynomials.getDegrees(), { SD::VERTEX_C } );
      poly_stencil_diagonal.setPolynomial( polynomials );

      std::map< SD, real_t > d_stencil;

      // z top
      for ( uint_t z = N_edge - 2; z > N_edge - 2 - boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         poly_stencil_diagonal.setZ( h * real_c( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z inner
      for ( uint_t z = N_edge - 2 - boundarySize; z >= 1 + boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         poly_stencil_diagonal.setZ( h * real_c( z ) );
         // y north:
         for ( uint_t y = N_edge - 2 - z; y > N_edge - 2 - boundarySize - z; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }

         // y inner:
         for ( uint_t y = N_edge - 2 - boundarySize - z; y >= 1 + boundarySize; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            // x west:
            for ( uint_t x = N_edge - 2 - z - y; x > N_edge - 2 - boundarySize - z - y; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }

            // LIKWID_MARKER_START( "backward_inner" );
            // x inner:
            for ( uint_t x = N_edge - 2 - boundarySize - z - y; x >= 1 + boundarySize; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }

               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
            // LIKWID_MARKER_STOP( "backward_inner" );

            // x east:
            for ( uint_t x = std::min( boundarySize, N_edge - 2 - boundarySize - z - y ); x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }

         // y south:
         for ( uint_t y = std::min( boundarySize, N_edge - 2 - boundarySize - z ); y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }

      // z bottom
      for ( uint_t z = std::min( boundarySize, N_edge - 3 - boundarySize ); z >= 1; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, b, u );
               apply_backward_substitution( x, y, z, l_stencil, b, u );
            }
         }
      }
   }
}

template < typename FunctionType, typename OpStencilProviderType >
void apply_full_surrogate_ilu_smoothing_step_new( OpStencilProviderType& opStencilProvider,
                                                  LDLTPolynomials&       polynomials_l,
                                                  uint_t                 level,
                                                  Cell&                  cell,
                                                  const FunctionType&    u_function,
                                                  const FunctionType&    w_function,
                                                  const FunctionType&    b_function )
{
   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   const auto idx = [N_edge]( uint_t x, uint_t y, uint_t z ) { return indexing::macroCellIndex( N_edge, x, y, z ); };

   real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

   // unpack u and b
   auto u = cell.getData( u_function.getCellDataID() )->getPointer( level );
   auto w = cell.getData( w_function.getCellDataID() )->getPointer( level );
   auto b = cell.getData( b_function.getCellDataID() )->getPointer( level );

   const size_t boundarySize = 1;

   auto apply_forward_substitution = [idx]( uint_t x, uint_t y, uint_t z, std::array< real_t, 7 >& l, real_t* w_dat ) {
      real_t sum = 0;

      // SD::VERTEX_W
      sum += l[0] * w_dat[idx( x - 1, y, z )];
      // SD::VERTEX_S
      sum += l[1] * w_dat[idx( x, y - 1, z )];
      // SD::VERTEX_SE
      sum += l[2] * w_dat[idx( x + 1, y - 1, z )];
      // SD::VERTEX_BNW
      sum += l[3] * w_dat[idx( x - 1, y + 1, z - 1 )];
      // SD::VERTEX_BN
      sum += l[4] * w_dat[idx( x, y + 1, z - 1 )];
      // SD::VERTEX_BC
      sum += l[5] * w_dat[idx( x, y, z - 1 )];
      // SD::VERTEX_BE
      sum += l[6] * w_dat[idx( x + 1, y, z - 1 )];

      w_dat[idx( x, y, z )] -= sum;
   };

   auto apply_diagonal_scaling = [idx]( uint_t x, uint_t y, uint_t z, const std::array< real_t, 1 >& stencil, real_t* w_dat ) {
      w_dat[idx( x, y, z )] *= stencil[0];
   };

   auto apply_backward_substitution = [idx]( uint_t x, uint_t y, uint_t z, std::array< real_t, 7 >& l, real_t* w_dat ) {
      real_t sum = 0;
      // SD::VERTEX_E
      sum += l[0] * w_dat[idx( x + 1, y, z )];
      // SD::VERTEX_N
      sum += l[1] * w_dat[idx( x, y + 1, z )];
      // SD::VERTEX_NW
      sum += l[2] * w_dat[idx( x - 1, y + 1, z )];
      // SD::VERTEX_TSE
      sum += l[3] * w_dat[idx( x + 1, y - 1, z + 1 )];
      // SD::VERTEX_TS
      sum += l[4] * w_dat[idx( x, y - 1, z + 1 )];
      // SD::VERTEX_TC
      sum += l[5] * w_dat[idx( x, y, z + 1 )];
      // SD::VERTEX_TW
      sum += l[6] * w_dat[idx( x - 1, y, z + 1 )];
      w_dat[idx( x, y, z )] -= sum;
   };

   auto calc_residual =
       [idx](
           uint_t x, uint_t y, uint_t z, const std::array< real_t, 15 >& a, real_t const* u_d, real_t const* b_d, real_t* w_d ) {
          w_d[idx( x, y, z )] = b_d[idx( x, y, z )];
          real_t tmp          = 0;
          // SD::VERTEX_C
          tmp += a[0] * u_d[idx( x, y, z )];
          // SD::VERTEX_W
          tmp += a[1] * u_d[idx( x - 1, y, z )];
          // SD::VERTEX_S
          tmp += a[2] * u_d[idx( x, y - 1, z )];
          // SD::VERTEX_SE
          tmp += a[3] * u_d[idx( x + 1, y - 1, z )];
          // SD::VERTEX_BNW
          tmp += a[4] * u_d[idx( x - 1, y + 1, z - 1 )];
          // SD::VERTEX_BN
          tmp += a[5] * u_d[idx( x, y + 1, z - 1 )];
          // SD::VERTEX_BC
          tmp += a[6] * u_d[idx( x, y, z - 1 )];
          // SD::VERTEX_BE
          tmp += a[7] * u_d[idx( x + 1, y, z - 1 )];
          // SD::VERTEX_E
          tmp += a[8] * u_d[idx( x + 1, y, z )];
          // SD::VERTEX_N
          tmp += a[9] * u_d[idx( x, y + 1, z )];
          // SD::VERTEX_NW
          tmp += a[10] * u_d[idx( x - 1, y + 1, z )];
          // SD::VERTEX_TSE
          tmp += a[11] * u_d[idx( x + 1, y - 1, z + 1 )];
          // SD::VERTEX_TS
          tmp += a[12] * u_d[idx( x, y - 1, z + 1 )];
          // SD::VERTEX_TC
          tmp += a[13] * u_d[idx( x, y, z + 1 )];
          // SD::VERTEX_TW
          tmp += a[14] * u_d[idx( x - 1, y, z + 1 )];

          w_d[idx( x, y, z )] -= tmp;
       };

   auto add_correction = [idx]( uint_t x, uint_t y, uint_t z, real_t const* w_d, real_t* u_d ) {
      u_d[idx( x, y, z )] += w_d[idx( x, y, z )];
   };

   // ---------------------
   // forward substitution:
   // ---------------------
   {
      PolyStencilNew< 7 > poly_stencil_lower( polynomials_l.getDegrees(), lowerDirections );
      poly_stencil_lower.setPolynomial( polynomials_l );
      std::array< real_t, 7 > l_stencil{};

      std::array< real_t, 15 > a_stencil{};

      // z bottom:
      for ( uint_t z = 1; z < 1 + boundarySize; z += 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         opStencilProvider.setZ( h * real_c( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }
      }

      // z inner:
      for ( uint_t z = 1 + boundarySize; z <= N_edge - 2 - boundarySize; z += 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         opStencilProvider.setZ( h * real_c( z ) );
         // y south:
         for ( uint_t y = 1; y < 1 + boundarySize; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }

         // y inner:
         for ( uint_t y = 1 + boundarySize; y <= N_edge - 2 - boundarySize - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            // x west:
            for ( uint_t x = 1; x < 1 + boundarySize; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }

            LIKWID_MARKER_START( "forward:inner:new" );
            // x inner:
            for ( uint_t x = 1 + boundarySize; x <= N_edge - 2 - boundarySize - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
            LIKWID_MARKER_STOP( "forward:inner:new" );

            // x east:
            for ( uint_t x = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z - y ); x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }

         // y north:
         for ( uint_t y = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z ); y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }
      }

      // z bottom:
      for ( uint_t z = std::max( 1 + boundarySize, N_edge - 1 - boundarySize ); z <= N_edge - 2; z += 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         opStencilProvider.setZ( h * real_c( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }
      }
   }

   // ---------------------------------
   // diagonal & backward substitution:
   // ---------------------------------
   {
      PolyStencilNew< 7 > poly_stencil_lower( polynomials_l.getDegrees(), lowerDirections );
      poly_stencil_lower.setPolynomial( polynomials_l );
      poly_stencil_lower.setOffset( SD::VERTEX_W, h, 0, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_S, 0, h, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_SE, -h, h, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_BNW, +h, -h, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BN, 0, -h, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BC, 0, 0, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BE, -h, 0, +h );

      std::array< real_t, 7 > l_stencil{};

      PolyStencilNew< 1 > poly_stencil_diagonal( polynomials_l.getDegrees(), { SD::VERTEX_C } );
      poly_stencil_diagonal.setPolynomial( polynomials_l );

      std::array< real_t, 1 > d_stencil{};

      // z top
      for ( uint_t z = N_edge - 2; z > N_edge - 2 - boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         poly_stencil_diagonal.setZ( h * real_c( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, d_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }
      }

      // z inner
      for ( uint_t z = N_edge - 2 - boundarySize; z >= 1 + boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         poly_stencil_diagonal.setZ( h * real_c( z ) );
         // y north:
         for ( uint_t y = N_edge - 2 - z; y > N_edge - 2 - boundarySize - z; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, d_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }

         // y inner:
         for ( uint_t y = N_edge - 2 - boundarySize - z; y >= 1 + boundarySize; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, d_stencil );
            // x west:
            for ( uint_t x = N_edge - 2 - z - y; x > N_edge - 2 - boundarySize - z - y; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }

            // x inner:
            LIKWID_MARKER_START( "backward:inner:new" );
            for ( uint_t x = N_edge - 2 - boundarySize - z - y; x >= 1 + boundarySize; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );

               apply_diagonal_scaling( x, y, z, d_stencil, w );

               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
            LIKWID_MARKER_STOP( "backward:inner:new" );

            // x east:
            for ( uint_t x = std::min( boundarySize, N_edge - 2 - boundarySize - z - y ); x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }

         // y south:
         for ( uint_t y = std::min( boundarySize, N_edge - 2 - boundarySize - z ); y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, d_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }
      }

      // z bottom
      for ( uint_t z = std::min( boundarySize, N_edge - 3 - boundarySize ); z >= 1; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, d_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }
      }
   }
}

template < typename FunctionType,
           typename OpStencilProviderType,
           bool useBoundaryCorrection                = false,
           bool useMatrixBoundaryValuesInInnerRegion = false >
void apply_full_surrogate_ilu_smoothing_step( OpStencilProviderType& opStencilProvider,
                                              LDLTPolynomials&       polynomials_l,
                                              LDLTBoundaryStencils&  boundaryStencils,
                                              uint_t                 level,
                                              Cell&                  cell,
                                              const FunctionType&    u_function,
                                              const FunctionType&    w_function,
                                              const FunctionType&    b_function )
{
   const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
      return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
   };

   using StencilT = std::map< SD, real_t >;

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

   // unpack u and b
   auto u = cell.getData( u_function.getCellDataID() )->getPointer( level );
   auto w = cell.getData( w_function.getCellDataID() )->getPointer( level );
   auto b = cell.getData( b_function.getCellDataID() )->getPointer( level );

   const size_t boundarySize = 1;

   auto get_l_stencil = [&boundaryStencils, level]( uint_t x, uint_t y, uint_t z, StencilT& stencil ) {
      stencil = boundaryStencils.get( x, y, z );
   };

   auto get_d_stencil = [&boundaryStencils, level]( uint_t x, uint_t y, uint_t z, StencilT& stencil ) {
      stencil = boundaryStencils.get( x, y, z );
   };

   auto apply_forward_substitution = [cidx]( uint_t x, uint_t y, uint_t z, StencilT& l, real_t* w_dat ) {
      for ( auto d : lowerDirections )
         w_dat[cidx( x, y, z, SD::VERTEX_C )] -= l[d] * w_dat[cidx( x, y, z, d )];
   };

   auto apply_diagonal_scaling = [cidx]( uint_t x, uint_t y, uint_t z, StencilT& stencil, real_t* w_dat ) {
      w_dat[cidx( x, y, z, SD::VERTEX_C )] *= stencil[SD::VERTEX_C];
   };

   auto get_lt_stencil = [&boundaryStencils, &polynomials_l, N_edge, h, level]( uint_t x, uint_t y, uint_t z, StencilT& lt ) {
      if ( ldlt::p1::dim3::on_cell_boundary( x + 1, y, z, 2, N_edge ) )
         lt[SD::VERTEX_W] = boundaryStencils.get( x + 1, y, z )[SD::VERTEX_W];
      else
         lt[SD::VERTEX_W] = polynomials_l.getPolynomial( SD::VERTEX_W ).eval( to_point( x + 1, y, z, h ) );

      if ( ldlt::p1::dim3::on_cell_boundary( x, y + 1, z, 2, N_edge ) )
         lt[SD::VERTEX_S] = boundaryStencils.get( x, y + 1, z )[SD::VERTEX_S];
      else
         lt[SD::VERTEX_S] = polynomials_l.getPolynomial( SD::VERTEX_S ).eval( to_point( x, y + 1, z, h ) );

      if ( ldlt::p1::dim3::on_cell_boundary( x - 1, y + 1, z, 2, N_edge ) )
         lt[SD::VERTEX_SE] = boundaryStencils.get( x - 1, y + 1, z )[SD::VERTEX_SE];
      else
         lt[SD::VERTEX_SE] = polynomials_l.getPolynomial( SD::VERTEX_SE ).eval( to_point( x - 1, y + 1, z, h ) );

      if ( ldlt::p1::dim3::on_cell_boundary( x + 1, y - 1, z + 1, 2, N_edge ) )
         lt[SD::VERTEX_BNW] = boundaryStencils.get( x + 1, y - 1, z + 1 )[SD::VERTEX_BNW];
      else
         lt[SD::VERTEX_BNW] = polynomials_l.getPolynomial( SD::VERTEX_BNW ).eval( to_point( x + 1, y - 1, z + 1, h ) );

      if ( ldlt::p1::dim3::on_cell_boundary( x, y - 1, z + 1, 2, N_edge ) )
         lt[SD::VERTEX_BN] = boundaryStencils.get( x, y - 1, z + 1 )[SD::VERTEX_BN];
      else
         lt[SD::VERTEX_BN] = polynomials_l.getPolynomial( SD::VERTEX_BN ).eval( to_point( x, y - 1, z + 1, h ) );

      if ( ldlt::p1::dim3::on_cell_boundary( x, y, z + 1, 2, N_edge ) )
         lt[SD::VERTEX_BC] = boundaryStencils.get( x, y, z + 1 )[SD::VERTEX_BC];
      else
         lt[SD::VERTEX_BC] = polynomials_l.getPolynomial( SD::VERTEX_BC ).eval( to_point( x, y, z + 1, h ) );

      if ( ldlt::p1::dim3::on_cell_boundary( x - 1, y, z + 1, 2, N_edge ) )
         lt[SD::VERTEX_BE] = boundaryStencils.get( x - 1, y, z + 1 )[SD::VERTEX_BE];
      else
         lt[SD::VERTEX_BE] = polynomials_l.getPolynomial( SD::VERTEX_BE ).eval( to_point( x - 1, y, z + 1, h ) );
   };

   auto apply_backward_substitution = [cidx]( uint_t x, uint_t y, uint_t z, StencilT& l, real_t* w_dat ) {
      for ( auto d : upperDirections )
         w_dat[cidx( x, y, z, SD::VERTEX_C )] -= l[opposite( d )] * w_dat[cidx( x, y, z, d )];
   };

   auto calc_residual = [cidx]( uint_t x, uint_t y, uint_t z, StencilT& a, real_t const* u_d, real_t const* b_d, real_t* w_d ) {
      w_d[cidx( x, y, z, SD::VERTEX_C )] = b_d[cidx( x, y, z, SD::VERTEX_C )];
      real_t tmp                         = 0;
      for ( auto d : allDirections )
         tmp += a[d] * u_d[cidx( x, y, z, d )];
      w_d[cidx( x, y, z, SD::VERTEX_C )] -= tmp;
   };

   auto add_correction = [cidx]( uint_t x, uint_t y, uint_t z, real_t const* w_d, real_t* u_d ) {
      u_d[cidx( x, y, z, SD::VERTEX_C )] += w_d[cidx( x, y, z, SD::VERTEX_C )];
   };

   // ---------------------
   // forward substitution:
   // ---------------------
   {
      PolyStencil< 7 > poly_stencil_lower( polynomials_l.getDegrees(), lowerDirections );
      poly_stencil_lower.setPolynomial( polynomials_l );
      std::map< SD, real_t > l_stencil;

      std::map< SD, real_t > a_stencil;

      // z bottom:
      for ( uint_t z = 1; z < 1 + boundarySize; z += 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         opStencilProvider.setZ( h * real_c( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }
      }

      // z inner:
      for ( uint_t z = 1 + boundarySize; z <= N_edge - 2 - boundarySize; z += 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         opStencilProvider.setZ( h * real_c( z ) );
         // y south:
         for ( uint_t y = 1; y < 1 + boundarySize; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }

         // y inner:
         for ( uint_t y = 1 + boundarySize; y <= N_edge - 2 - boundarySize - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            // x west:
            for ( uint_t x = 1; x < 1 + boundarySize; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }

            LIKWID_MARKER_START( "forward:inner" );
            // x inner:
            for ( uint_t x = 1 + boundarySize; x <= N_edge - 2 - boundarySize - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
            LIKWID_MARKER_STOP( "forward:inner" );

            // x east:
            for ( uint_t x = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z - y ); x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }

         // y north:
         for ( uint_t y = std::max( 1 + boundarySize, N_edge - 1 - boundarySize - z ); y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }
      }

      // z bottom:
      for ( uint_t z = std::max( 1 + boundarySize, N_edge - 1 - boundarySize ); z <= N_edge - 2; z += 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         opStencilProvider.setZ( h * real_c( z ) );
         for ( uint_t y = 1; y <= N_edge - 2 - z; y += 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( 1 - 1 ), h, l_stencil );
            opStencilProvider.setY( h * real_c( y ) );
            opStencilProvider.setStartX( h * real_c( 1 - 1 ), h, a_stencil );
            for ( uint_t x = 1; x <= N_edge - 2 - z - y; x += 1 )
            {
               // residual:
               opStencilProvider.incrementEval( a_stencil );
               calc_residual( x, y, z, a_stencil, u, b, w );
               // substitution:
               poly_stencil_lower.incrementEval( l_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections( x, y, z, N_edge, l_stencil );
               else
                  get_l_stencil( x, y, z, l_stencil );
               apply_forward_substitution( x, y, z, l_stencil, w );
            }
         }
      }
   }

   // ---------------------------------
   // diagonal & backward substitution:
   // ---------------------------------
   {
      PolyStencil< 7 > poly_stencil_lower( polynomials_l.getDegrees(), lowerDirections );
      poly_stencil_lower.setPolynomial( polynomials_l );
      poly_stencil_lower.setOffset( SD::VERTEX_W, h, 0, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_S, 0, h, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_SE, -h, h, 0 );
      poly_stencil_lower.setOffset( SD::VERTEX_BNW, +h, -h, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BN, 0, -h, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BC, 0, 0, +h );
      poly_stencil_lower.setOffset( SD::VERTEX_BE, -h, 0, +h );

      std::map< SD, real_t > l_stencil;

      PolyStencil< 1 > poly_stencil_diagonal( polynomials_l.getDegrees(), { SD::VERTEX_C } );
      poly_stencil_diagonal.setPolynomial( polynomials_l );

      std::map< SD, real_t > d_stencil;

      // z top
      for ( uint_t z = N_edge - 2; z > N_edge - 2 - boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         poly_stencil_diagonal.setZ( h * real_c( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }
      }

      // z inner
      for ( uint_t z = N_edge - 2 - boundarySize; z >= 1 + boundarySize; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         poly_stencil_diagonal.setZ( h * real_c( z ) );
         // y north:
         for ( uint_t y = N_edge - 2 - z; y > N_edge - 2 - boundarySize - z; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }

         // y inner:
         for ( uint_t y = N_edge - 2 - boundarySize - z; y >= 1 + boundarySize; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            // x west:
            for ( uint_t x = N_edge - 2 - z - y; x > N_edge - 2 - boundarySize - z - y; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }

            LIKWID_MARKER_START( "backward:inner" );
            // x inner:
            for ( uint_t x = N_edge - 2 - boundarySize - z - y; x >= 1 + boundarySize; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useMatrixBoundaryValuesInInnerRegion )
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }

               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
            LIKWID_MARKER_STOP( "backward:inner" );

            // x east:
            for ( uint_t x = std::min( boundarySize, N_edge - 2 - boundarySize - z - y ); x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }

         // y south:
         for ( uint_t y = std::min( boundarySize, N_edge - 2 - boundarySize - z ); y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }
      }

      // z bottom
      for ( uint_t z = std::min( boundarySize, N_edge - 3 - boundarySize ); z >= 1; z -= 1 )
      {
         poly_stencil_lower.setZ( h * real_c( z ) );
         for ( uint_t y = N_edge - 2 - z; y >= 1; y -= 1 )
         {
            poly_stencil_lower.setY( h * real_c( y ) );
            poly_stencil_lower.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            poly_stencil_diagonal.setY( h * real_c( y ) );
            poly_stencil_diagonal.setStartX( h * real_c( N_edge - 2 - z - y + 1 ), -h, l_stencil );
            for ( uint_t x = N_edge - 2 - z - y; x >= 1; x -= 1 )
            {
               poly_stencil_lower.incrementEval( l_stencil );
               poly_stencil_diagonal.incrementEval( d_stencil );
               if ( useBoundaryCorrection )
                  apply_boundary_corrections_on_backward_stencil( x, y, z, N_edge, l_stencil );
               else
               {
                  get_lt_stencil( x, y, z, l_stencil );
                  get_d_stencil( x, y, z, d_stencil );
               }
               apply_diagonal_scaling( x, y, z, d_stencil, w );
               apply_backward_substitution( x, y, z, l_stencil, w );
               add_correction( x, y, z, w, u );
            }
         }
      }
   }
}

} // namespace dim3
} // namespace p1
} // namespace ldlt

template < typename FormType >
class P1QSurrogateCellOperator : public Operator< P1Function< real_t >, P1Function< real_t > >
{
   using funcType = P1Function< real_t >;

 public:
   P1QSurrogateCellOperator( const std::shared_ptr< PrimitiveStorage >& storage,
                             size_t                                     minLevel,
                             size_t                                     maxLevel,
                             const FormType&                            form,
                             bool                                       useSymmetry,
                             const std::array< uint_t, 3 >&             degrees,
                             uint_t                                     assemblyLevel )
   : Operator< P1Function< real_t >, P1Function< real_t > >( storage, minLevel, maxLevel )
   , form_( form )
   , useSymmetry_( useSymmetry )
   , degrees_( degrees )
   {
      // storage for surrogate operator
      std::shared_ptr< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling > polyDataHandlingOp;
      if ( useSymmetry )
      {
         polyDataHandlingOp = std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >(
             minLevel_, maxLevel_, degrees, ldlt::p1::dim3::lowerDirectionsAndCenter );
      }
      else
      {
         polyDataHandlingOp = std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >(
             minLevel_, maxLevel_, degrees, ldlt::p1::dim3::allDirections );
      }
      storage_->addCellData( opPolynomialsID_, polyDataHandlingOp, "P1QSurrogateCellOperator" );

      WALBERLA_CHECK_LESS( storage_->getNumberOfGlobalCells(), 2 );
      for ( uint_t level = minLevel; level <= maxLevel; level += 1 )
      {
         for ( auto& cit : storage_->getCells() )
         {
            factorize_op_matrix_inplace( level, *cit.second, form_, assemblyLevel );
         }
      }
   }

   void apply( const funcType& src,
               const funcType& dst,
               size_t          level,
               DoFType         flag,
               UpdateType      updateType = Replace ) const override final
   {
      WALBERLA_CHECK_EQUAL( updateType, Replace );

      src.template communicate< Vertex, Edge >( level );
      src.template communicate< Edge, Face >( level );
      src.template communicate< Face, Cell >( level );

      for ( auto cit : storage_->getCells() )
      {
         Cell& cell        = *cit.second;
         auto& polynomials = cell.getData( opPolynomialsID_ )->getLevel( level );
         ldlt::p1::dim3::apply_surrogate_operator( polynomials, level, useSymmetry_, cell, src, dst );
      }
   }

 private:
   void factorize_op_matrix_inplace( uint_t level, Cell& cell, FormType& form, uint_t coarseLevel )
   {
      std::unique_ptr< ldlt::p1::dim3::Interpolators > interpolators( nullptr );
      if ( useSymmetry_ )
      {
         interpolators = std::make_unique< ldlt::p1::dim3::Interpolators >( degrees_, ldlt::p1::dim3::lowerDirectionsAndCenter );
      }
      else
      {
         interpolators = std::make_unique< ldlt::p1::dim3::Interpolators >( degrees_, ldlt::p1::dim3::allDirections );
      }

      ldlt::p1::dim3::assemble_surrogate_operator( form, cell, coarseLevel, level, *interpolators );

      auto& polynomials = cell.getData( opPolynomialsID_ )->getLevel( level );

      interpolators->interpolate( polynomials );
   }

 private:
   FormType form_;

   bool                    useSymmetry_;
   std::array< uint_t, 3 > degrees_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierachicalPolynomials, Cell > opPolynomialsID_;
};

enum class MainOperatorStencilType
{
   Constant,
   Varying,
   Polynomial
};

template < class OperatorType, class FormType, bool useBoundaryCorrection = false >
class P1LDLTSurrogateCellSmoother : public CellSmoother< OperatorType >
{
 public:
   using FunctionType = typename OperatorType::srcType;

   P1LDLTSurrogateCellSmoother( std::shared_ptr< PrimitiveStorage > storage,
                                uint_t                              minLevel,
                                uint_t                              maxLevel,
                                std::array< uint_t, 3 >             opDegree,
                                std::array< uint_t, 3 >             iluDegree,
                                bool                                useSymmetry,
                                FormType                            form )
   : storage_( std::move( storage ) )
   , tmp1_( "tmp", storage_, minLevel, maxLevel )
   , tmp2_( "tmp", storage_, minLevel, maxLevel )
   , form_( form )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , iluPolyDegree_( iluDegree )
   , opPolyDegree_( opDegree )
   , useSymmetryOfOperator_( useSymmetry )
   , mainOperatorStencilType_( MainOperatorStencilType::Varying )
   {
      // storage for surrogate ldlt
      auto polyDataHandling = std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >(
          minLevel_, maxLevel_, iluPolyDegree_, ldlt::p1::dim3::lowerDirectionsAndCenter );
      storage_->addCellData( ldltPolynomialsID_, polyDataHandling, "P1LDLTSurrogateCellSmootherPolynomials" );

      // storage for surrogate operator
      std::shared_ptr< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling > polyDataHandlingOp;
      if ( useSymmetryOfOperator_ )
      {
         polyDataHandlingOp = std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >(
             minLevel_, maxLevel_, opPolyDegree_, ldlt::p1::dim3::lowerDirectionsAndCenter );
      }
      else
      {
         polyDataHandlingOp = std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >(
             minLevel_, maxLevel_, opPolyDegree_, ldlt::p1::dim3::allDirections );
      }
      storage_->addCellData( opPolynomialsID_, polyDataHandlingOp, "P1SurrogateCellSmootherPolynomials" );

      // storage for the boundary stencils
      auto boundaryDataHandling =
          std::make_shared< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencilsMemoryDataHandling >( minLevel_, maxLevel_ );
      storage_->addCellData( boundaryID_, boundaryDataHandling, "P1LDLTSurrogateCellSmootherBoundary" );
   }

   void setMainOperatorStencilType( MainOperatorStencilType type ) { mainOperatorStencilType_ = type; }

   using SD = stencilDirection;

   void init( uint_t assemblyLevel, uint_t skipLevel )
   {
      for ( auto& it : storage_->getCells() )
      {
         Cell& cell = *it.second;
         for ( uint_t level = minLevel_; level <= maxLevel_; ++level )
         {
            if ( mainOperatorStencilType_ == MainOperatorStencilType::Varying )
            {
               factorize_op_matrix_inplace( level, cell, form_, assemblyLevel );
            }
            factorize_ldlt_matrix_inplace( level, cell, form_, skipLevel );
         }
      }
   }

   void preSmooth( const OperatorType&                   A,
                   uint_t                                level,
                   const typename OperatorType::srcType& u,
                   const typename OperatorType::dstType& b ) override
   {
      //      tmp1_.assign( { 1. }, { u }, level, DirichletBoundary );
      //      A.apply( u, tmp1_, level, flag_ );
      //      tmp1_.assign( { 1., -1. }, { b, tmp1_ }, level, flag_ );
      //
      //      tmp1_.template communicate< Vertex, Edge >( level );
      //      tmp1_.template communicate< Edge, Face >( level );
      //      tmp1_.template communicate< Face, Cell >( level );

      /*
      tmp1_.assign( { 1. }, { u }, level, DirichletBoundary );
      for ( auto cit : storage_->getCells() )
      {
         Cell& cell        = *cit.second;
         auto& polynomials = cell.getData( opPolynomialsID_ )->getLevel( level );
         ldlt::p1::dim3::apply_surrogate_operator( polynomials, level, useSymmetryOfOperator_, cell, u, tmp1_ );
      }

      tmp1_.assign( { 1., -1. }, { b, tmp1_ }, level, flag_ );
       */

      tmp2_.assign( { 1. }, { u }, level, DirichletBoundary );
      tmp2_.template communicate< Vertex, Edge >( level );
      tmp2_.template communicate< Edge, Face >( level );
      tmp2_.template communicate< Face, Cell >( level );
   }

   void postSmooth( const OperatorType&,
                    uint_t                                level,
                    const typename OperatorType::srcType& u,
                    const typename OperatorType::dstType& ) override
   {
      // u.assign( { 1., 1. }, { tmp2_, u }, level, flag_ );
   }

   void smooth( const OperatorType&                   A,
                uint_t                                level,
                Cell&                                 cell,
                const typename OperatorType::srcType& u,
                const typename OperatorType::dstType& b ) override
   {
      smooth_apply( A, level, cell, u, b );
   }

   void factorize_op_matrix_inplace( uint_t level, Cell& cell, FormType& form, uint_t coarseLevel )
   {
      std::unique_ptr< ldlt::p1::dim3::Interpolators > interpolators( nullptr );
      if ( useSymmetryOfOperator_ )
      {
         interpolators =
             std::make_unique< ldlt::p1::dim3::Interpolators >( opPolyDegree_, ldlt::p1::dim3::lowerDirectionsAndCenter );
      }
      else
      {
         interpolators = std::make_unique< ldlt::p1::dim3::Interpolators >( opPolyDegree_, ldlt::p1::dim3::allDirections );
      }

      ldlt::p1::dim3::assemble_surrogate_operator( form, cell, coarseLevel, level, *interpolators );

      auto& polynomials = cell.getData( opPolynomialsID_ )->getLevel( level );

      interpolators->interpolate( polynomials );
   }

   void factorize_ldlt_matrix_inplace( uint_t level, Cell& cell, FormType& form, uint_t skipLevel )
   {
      auto& polynomials  = cell.getData( ldltPolynomialsID_ )->getLevel( level );
      auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );

      real_t h          = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );
      real_t H          = 1. / real_c( levelinfo::num_microedges_per_edge( skipLevel ) );
      auto   skipLength = uint_c( std::max( 1., std::round( H / h ) ) );

      auto is_interpolation_point = [skipLength, this]( uint_t x, uint_t y, uint_t z, uint_t offx, uint_t offy, uint_t offz ) {
         auto x_b = x - 1 - offx;
         auto y_b = y - 1 - offy;
         auto z_b = z - 1 - offz;
         return ( x_b % skipLength == 0 ) && ( y_b % skipLength == 0 ) && ( z_b % skipLength == 0 );
      };

      ldlt::p1::dim3::Interpolators interpolators( iluPolyDegree_ );

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
         Point3D p( { h * real_c( x ), h * real_c( y ), h * real_c( z ) } );

         // we save the inverse diagonal at the stencil
         stencil[SD::VERTEX_C] = 1. / stencil[SD::VERTEX_C];

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

      polynomials.print();
   }

   void smooth_apply( const OperatorType&,
                      uint_t                                level,
                      Cell&                                 cell,
                      const typename OperatorType::srcType& u,
                      const typename OperatorType::dstType& b )
   {
      // auto& polynomialData = cell.getData( ldltPolynomialsID_ )->getLevel( level );
      // auto& boundaryData   = cell.getData( boundaryID_ )->getLevel( level );
      //
      // ldlt::p1::dim3::apply_surrogate_substitutions< typename OperatorType::srcType, useBoundaryCorrection >(
      //     boundaryData, polynomialData, level, cell, u, b );

      auto& polynomial_a = cell.getData( opPolynomialsID_ )->getLevel( level );
      auto& polynomial_l = cell.getData( ldltPolynomialsID_ )->getLevel( level );
      auto& boundary     = cell.getData( boundaryID_ )->getLevel( level );

      const real_t h = 1. / real_c( levelinfo::num_microedges_per_edge( level ) );

      if ( mainOperatorStencilType_ == MainOperatorStencilType::Varying )
      {
         using StencilProviderType = ldlt::p1::dim3::AssembledStencil< FormType >;
         StencilProviderType opStencilProvider( level, cell, form_ );

         ldlt::p1::dim3::apply_full_surrogate_ilu_smoothing_step< typename OperatorType::srcType,
                                                                  StencilProviderType,
                                                                  useBoundaryCorrection >(
             opStencilProvider, polynomial_l, boundary, level, cell, u, tmp2_, b );
      }
      else if ( mainOperatorStencilType_ == MainOperatorStencilType::Constant )
      {
         using StencilProviderType = ldlt::p1::dim3::ConstantStencil< FormType >;
         StencilProviderType opStencilProvider( level, cell, form_ );

         ldlt::p1::dim3::apply_full_surrogate_ilu_smoothing_step< typename OperatorType::srcType,
                                                                  StencilProviderType,
                                                                  useBoundaryCorrection >(
             opStencilProvider, polynomial_l, boundary, level, cell, u, tmp2_, b );
      }
      else if ( mainOperatorStencilType_ == MainOperatorStencilType::Polynomial )
      {
         using StencilProviderType = ldlt::p1::dim3::PolyStencil< 15 >;
         StencilProviderType poly_stencil_a( polynomial_a.getDegrees(), ldlt::p1::dim3::allDirections );
         if ( useSymmetryOfOperator_ )
            poly_stencil_a.setPolynomialSymmetrical( polynomial_a, h );
         else
            poly_stencil_a.setPolynomial( polynomial_a );
      }
      else
      {
         WALBERLA_ABORT( "Given operator stencil type was not implemented" );
      }
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

      const real_t h = 1. / real_c( N_edge );

      constexpr auto cidx = vertexdof::macrocell::indexFromVertex;

      std::map< SD, real_t > l_stencil;

      for ( auto it : storage_->getCells() )
      {
         Cell& cell           = *it.second;
         auto& boundaryData   = cell.getData( boundaryID_ )->getLevel( level );
         auto& polynomialData = cell.getData( ldltPolynomialsID_ )->getLevel( level );
         auto& polynomial     = polynomialData.getPolynomial( direction );
         auto  u_data         = cell.getData( u.getCellDataID() )->getPointer( level );

         for ( uint_t z = 0; z <= N_edge - 1; z += 1 )
         {
            for ( uint_t y = 0; y <= N_edge - 1 - z; y += 1 )
            {
               for ( uint_t x = 0; x <= N_edge - 1 - z - y; x += 1 )
               {
                  real_t polyValue = polynomial.eval( ldlt::p1::dim3::to_point( x, y, z, h ) );
                  polyValue = ldlt::p1::dim3::apply_boundary_corrections_to_scalar( x, y, z, N_edge, direction, polyValue );
                  if ( !useBoundaryCorrection && ldlt::p1::dim3::on_cell_boundary( x, y, z, 2, N_edge ) )
                     polyValue = boundaryData.get( x, y, z )[direction];
                  if ( direction == SD::VERTEX_C )
                     polyValue = 1. / polyValue;
                  u_data[cidx( level, x, y, z, SD::VERTEX_C )] = polyValue;
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

   std::array< uint_t, 3 > iluPolyDegree_;
   std::array< uint_t, 3 > opPolyDegree_;

   bool useSymmetryOfOperator_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencils, Cell > boundaryID_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierachicalPolynomials, Cell > ldltPolynomialsID_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierachicalPolynomials, Cell > opPolynomialsID_;

   MainOperatorStencilType mainOperatorStencilType_;
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