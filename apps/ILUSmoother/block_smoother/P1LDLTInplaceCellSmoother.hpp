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
            get_l_stencil( x - 1, y, z + 1, l_stencil );
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
      if ( stencils_.count( x ) == 0 || stencils_[x].count( y ) == 0 )
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
   using Basis      = MonomialBasis3D;
   using Polynomial = Polynomial3D< MonomialBasis3D >;

   explicit LDLTPolynomials( uint_t degree )
   : degree_( degree )
   {
      for ( auto d : lowerDirectionsAndCenter )
         polynomials_.emplace(d, degree );
   }

   inline Polynomial& getPolynomial( stencilDirection direction ) { return polynomials_[direction]; }

   [[nodiscard]] inline uint_t getDegree() const { return degree_; }

 private:
   uint_t degree_;

   std::map< SD, Polynomial3D< MonomialBasis3D > > polynomials_;
};

class LDLTHierachicalPolynomials
{
 public:
   LDLTHierachicalPolynomials( uint_t minLevel, uint_t maxLevel, uint_t degree )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , collections( maxLevel - minLevel + 1, LDLTPolynomials( degree ) )
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
   explicit LDLTHierachicalPolynomialsDataHandling( const uint_t& minLevel, const uint_t& maxLevel, const uint_t& degree )
   : minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , degree_( degree )
   {}

   std::shared_ptr< LDLTHierachicalPolynomials > initialize( const Cell* const ) const override
   {
      auto collection = std::make_shared< LDLTHierachicalPolynomials >( minLevel_, maxLevel_, degree_ );
      return collection;
   }

 private:
   uint_t minLevel_;
   uint_t maxLevel_;
   uint_t degree_;
};

class Interpolators
{
 public:
   using Interpolator3D = VariableLSQPInterpolator3D< MonomialBasis3D >;

   Interpolators()
   {
      for ( auto d : lowerDirectionsAndCenter )
         interpolators[d] = Interpolator3D();
   }

   Interpolator3D& operator()( SD direction ) { return interpolators[direction]; }

   void addStencil( const Point3D& p, const std::map< SD, real_t >& stencil )
   {
      for ( auto d : lowerDirectionsAndCenter )
         interpolators[d].addInterpolationPoint( p, stencil.at( d ) );
   }

   void interpolate( LDLTPolynomials& poly )
   {
      for ( auto d : lowerDirectionsAndCenter )
         interpolators[d].interpolate( poly.getPolynomial( d ) );
   }

 private:
   std::map< SD, Interpolator3D > interpolators;
};

template < uint_t degree, typename FunctionType >
void apply_surrogate_substitutions_impl(
    LDLTBoundaryStencils&   boundaryStencils,
    LDLTPolynomials&    polynomials,
    uint_t              level,
    Cell&               cell,
    const FunctionType& u_function,
    const FunctionType& b_function )
{
   const auto cidx = [level]( uint_t x, uint_t y, uint_t z, SD dir ) {
     return vertexdof::macrocell::indexFromVertex( level, x, y, z, dir );
   };

   auto get_l_stencil = [&boundaryStencils, level]( uint_t x, uint_t y, uint_t z, std::map< SD, real_t >& stencil ) {
     stencil = boundaryStencils.get( x, y, z );
   };

   const auto N_edge = levelinfo::num_microvertices_per_edge( level );

   real_t h          = 1. / static_cast< real_t >( levelinfo::num_microedges_per_edge( level ) );

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
            // Point3D p( { h * static_cast< real_t >( x ), h * static_cast< real_t >( y ), h * static_cast< real_t >( z ) } );
            // WALBERLA_LOG_INFO( x << " " << y << " " << z << " " << std::abs(polynomials.getPolynomial(SD::VERTEX_C).eval(p) - l_stencil[SD::VERTEX_C]) / l_stencil[SD::VERTEX_C]);
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
            get_l_stencil( x - 1, y, z + 1, l_stencil );
            u[cidx( x, y, z, SD::VERTEX_C )] -= l_stencil[opposite( SD::VERTEX_TW )] * u[cidx( x, y, z, SD::VERTEX_TW )];
         }
      }
   }
}

template < uint_t minDegree, uint_t maxDegree, typename FunctionType >
void apply_surrogate_substitutions(
    LDLTBoundaryStencils&   stencils,
    LDLTPolynomials&    polynomials,
    uint_t              level,
    uint_t              degree,
    Cell&               cell,
    const FunctionType& u_function,
    const FunctionType& b_function )
{
   if ( minDegree == degree )
   {
      apply_surrogate_substitutions_impl< minDegree >( stencils , polynomials, level, cell, u_function, b_function );
   }
   else
   {
      if constexpr ( minDegree < maxDegree )
      {
         apply_surrogate_substitutions< minDegree + 1, maxDegree >( stencils, polynomials, level, degree, cell, u_function, b_function );
      }
      else
      {
         WALBERLA_ABORT( "degree " << degree << " is larger than the maximum degree " << maxDegree );
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

   P1LDLTSurrogateCellSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel, uint_t degree )
   : storage_( std::move( storage ) )
   , tmp1_( "tmp", storage_, minLevel, maxLevel )
   , tmp2_( "tmp", storage_, minLevel, maxLevel )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   , polyDegree_( degree )
   {
      // storage for surrogate operator
      auto polyDataHandling =
          std::make_shared< ldlt::p1::dim3::LDLTHierachicalPolynomialsDataHandling >( minLevel_, maxLevel_, polyDegree_ );
      storage_->addCellData( polynomialsID_, polyDataHandling, "P1LDLTSurrogateCellSmootherPolynomials" );

      // storage for the boundary stencils
      auto boundaryDataHandling =
          std::make_shared< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencilsMemoryDataHandling >( minLevel_, maxLevel_ );
      storage_->addCellData( boundaryID_, boundaryDataHandling, "P1LDLTSurrogateCellSmootherBoundary" );
   }

   using SD = stencilDirection;

   static constexpr auto cindex = vertexdof::macrocell::indexFromVertex;

   void init( FormType& form, uint_t skipLevel )
   {
      for ( auto& it : storage_->getCells() )
      {
         Cell& cell = *it.second;
         for ( uint_t level = minLevel_; level <= maxLevel_; ++level )
         {
            factorize_matrix_inplace( level, cell, form, skipLevel );
         }
      }
   }

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

   void factorize_matrix_inplace( uint_t level, Cell& cell, FormType& form, uint_t skipLevel )
   {
      auto& polynomials  = cell.getData( polynomialsID_ )->getLevel( level );
      auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );

      real_t h          = 1. / static_cast< real_t >( levelinfo::num_microedges_per_edge( level ) );
      real_t H          = 1. / static_cast< real_t >( levelinfo::num_microedges_per_edge( skipLevel ) );
      auto   skipLength = static_cast< uint_t >( std::max( 1., std::round( H / h ) ) );

      auto is_interpolation_point = [skipLength, this]( uint_t x, uint_t y, uint_t z ) {
         auto x_b = x - 1;
         auto y_b = y - 1;
         auto z_b = z - 1;
         return ( x_b % skipLength == 0 ) && ( y_b % skipLength == 0 ) && ( z_b % skipLength == 0 );
      };

      ldlt::p1::dim3::Interpolators interpolators;

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
         boundaryData.add( x, y, z, stencil );
         if ( is_interpolation_point( x, y, z ) )
         {
            Point3D p( { h * static_cast< real_t >( x ), h * static_cast< real_t >( y ), h * static_cast< real_t >( z ) } );
            interpolators.addStencil( p, stencil );
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
      auto& polynomialData  = cell.getData( polynomialsID_ )->getLevel( level );
      auto& boundaryData = cell.getData( boundaryID_ )->getLevel( level );

      ldlt::p1::dim3::apply_surrogate_substitutions<0, 12>( boundaryData, polynomialData, level, polyDegree_, cell, u, b );
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

   uint_t polyDegree_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencils, Cell > boundaryID_;

   PrimitiveDataID< ldlt::p1::dim3::LDLTHierachicalPolynomials, Cell > polynomialsID_;
};

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
   {
      auto boundaryDataHandling =
          std::make_shared< ldlt::p1::dim3::LDLTHierarchicalBoundaryStencilsMemoryDataHandling >( minLevel_, maxLevel_ );
      storage_->addCellData( boundaryID_, boundaryDataHandling, "P1LDLTInplaceCellSmootherBoundary" );
   }

   using SD = stencilDirection;

   static constexpr auto cindex = vertexdof::macrocell::indexFromVertex;

   void init( FormType& form )
   {
      for ( auto& it : storage_->getCells() )
      {
         Cell& cell = *it.second;
         for ( uint_t level = minLevel_; level <= maxLevel_; ++level )
         {
            factorize_matrix_inplace( level, cell, form );
         }
      }
   }

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