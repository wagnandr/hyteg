#pragma once

#include "hyteg/p2functionspace/polynomial/P2PolynomialStencil.hpp"
#include "hyteg/polynomial/LSQPInterpolator.hpp"

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

void apply_boundary_corrections( uint_t x, uint_t y, uint_t z, uint_t N, std::map< SD, real_t >& stencil )
{
   if ( x == 1 )
   {
      stencil[stencilDirection::VERTEX_NW]  = 0;
      stencil[stencilDirection::VERTEX_W]   = 0;
      stencil[stencilDirection::VERTEX_TW] = 0;
      stencil[stencilDirection::VERTEX_BNW]  = 0;
   }
   if ( x + y + z == N - 2 )
   {
      stencil[stencilDirection::VERTEX_E]  = 0;
      stencil[stencilDirection::VERTEX_N]  = 0;
      stencil[stencilDirection::VERTEX_TC] = 0;
      stencil[stencilDirection::VERTEX_TSE] = 0;
   }
   if ( y == 1 )
   {
      stencil[stencilDirection::VERTEX_S]  = 0;
      stencil[stencilDirection::VERTEX_SE] = 0;
      stencil[stencilDirection::VERTEX_TS] = 0;
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
            beta[SD::VERTEX_BC][fidx(x, y)] = beta_bc;
            beta[SD::VERTEX_S][fidx(x, y)] = beta_s;
            beta[SD::VERTEX_BNW][fidx(x, y)] = beta_bnw;
            beta[SD::VERTEX_BE][fidx(x, y)] = beta_be;
            beta[SD::VERTEX_W][fidx(x, y)] = beta_w;
            beta[SD::VERTEX_BN][fidx(x, y)] = beta_bn;
            beta[SD::VERTEX_SE][fidx(x, y)] = beta_se;
            beta[SD::VERTEX_C][fidx(x, y)] = beta_c;

            // copy into stencil:
            l_stencil[SD::VERTEX_BC] = beta_bc;
            l_stencil[SD::VERTEX_S] = beta_s;
            l_stencil[SD::VERTEX_BNW] = beta_bnw;
            l_stencil[SD::VERTEX_BE] = beta_be;
            l_stencil[SD::VERTEX_W] = beta_w;
            l_stencil[SD::VERTEX_BN] = beta_bn;
            l_stencil[SD::VERTEX_SE] = beta_se;
            l_stencil[SD::VERTEX_C] = beta_c;

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
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_E )] * u[cidx( x, y, z, SD::VERTEX_E )];

            // N
            get_l_stencil( x, y + 1, z, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_N )] * u[cidx( x, y, z, SD::VERTEX_N )];

            // NW
            get_l_stencil( x - 1, y + 1, z, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_NW )] * u[cidx( x, y, z, SD::VERTEX_NW )];

            // TSE
            get_l_stencil( x + 1, y - 1, z + 1, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TSE )] * u[cidx( x, y, z, SD::VERTEX_TSE )];

            // TS
            get_l_stencil( x, y - 1, z + 1, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TS )] * u[cidx( x, y, z, SD::VERTEX_TS )];

            // TC
            get_l_stencil( x, y, z + 1, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TC )] * u[cidx( x, y, z, SD::VERTEX_TC )];

            // TW
            get_l_stencil( x-1, y, z + 1, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TW )] * u[cidx( x, y, z, SD::VERTEX_TW )];
         }
      }
   }
}

} // namespace dim3
} // namespace p1
} // namespace ldlt

template < class OperatorType, class FormType >
class P1LDLTInplaceCellSmoother : public CellSmoother< OperatorType >
{
 public:
   using FunctionType = typename OperatorType::srcType;

   P1LDLTInplaceCellSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
   : storage_( std::move( storage ) )
   , tmp1_( "tmp", storage_, minLevel, maxLevel )
   , tmp2_( "tmp", storage_, minLevel, maxLevel )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   {}

   using SD = stencilDirection;

   static constexpr auto cindex = vertexdof::macrocell::indexFromVertex;

   void init( const OperatorType& ) {}

   void smooth( const OperatorType&                   A,
                uint_t                                level,
                Cell&                                 cell,
                const typename OperatorType::srcType& u,
                const typename OperatorType::dstType& b ) override
   {
      tmp1_.assign( { 1. }, { u }, level, DirichletBoundary );
      A.apply( u, tmp1_, level, flag_ );
      tmp1_.assign( { 1., -1. }, { b, tmp1_ }, level, flag_ );
      smooth_apply( A, level, cell, tmp2_, tmp1_ );
      u.assign( { 1., 1. }, { tmp2_, u }, level, flag_ );
   }

   using InplaceMatrix = std::map< SD, std::vector< real_t > >;

   InplaceMatrix factorize_matrix_inplace( const OperatorType&, uint_t level, Cell& cell )
   {
      // we assume that the logical coordinates of the _inner_ layer ranges from 1 to N:
      const auto N_cell = levelinfo::num_microvertices_per_cell( level );

      InplaceMatrix l_matrix;
      for ( auto d : ldlt::p1::dim3::lowerDirections )
      {
         l_matrix[d] = std::vector< real_t >( N_cell, 0 );
      }
      l_matrix[SD::VERTEX_C] = std::vector< real_t >( N_cell, 1 );

      ldlt::p1::dim3::factorize_matrix(
          form_, level, cell, [&l_matrix, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
             for ( auto d : ldlt::p1::dim3::lowerDirectionsAndCenter )
                l_matrix[d][cindex( level, x, y, z, SD::VERTEX_C )] = stencil[d];
          } );

      return l_matrix;
   }

   void smooth_apply( const OperatorType&                   A,
                      uint_t                                level,
                      Cell&                                 cell,
                      const typename OperatorType::srcType& u,
                      const typename OperatorType::dstType& b )
   {
      auto L = factorize_matrix_inplace( A, level, cell );

      auto stencil_provider = [&L, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
         for ( auto d : ldlt::p1::dim3::lowerDirectionsAndCenter )
            stencil[d] = L[d][cindex( level, x, y, z, SD::VERTEX_C )];
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

 private:
   std::shared_ptr< PrimitiveStorage > storage_;

   FunctionType tmp1_;
   FunctionType tmp2_;

   FormType form_;

   DoFType flag_;

   uint_t minLevel_;
   uint_t maxLevel_;
};

} // namespace hyteg