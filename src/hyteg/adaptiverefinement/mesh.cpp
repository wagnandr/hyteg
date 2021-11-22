
#include "mesh.hpp"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <map>

#include "refine_cell.hpp"

namespace hyteg {

namespace adaptiveRefinement {

template < class K_Simplex >
void Mesh< K_Simplex >::refineRG( const std::set< std::shared_ptr< K_Simplex > >& elements_to_refine )
{
   // remove green edges
   auto R = elements_to_refine;
   remove_green_edges( R );

   /* recursively apply red refinement for elements
        that otherwise would be subject to multiple
        green refinement steps
   */
   std::set< std::shared_ptr< K_Simplex > > U = _T;
   std::set< std::shared_ptr< K_Simplex > > refined;
   while ( not R.empty() )
   {
      refined.merge( refine_red( R, U ) );

      R = find_elements_for_red_refinement( U );
   }

   // apply green refinement
   refined.merge( refine_green( U ) );

   // update current configuration
   _T = U;
   _T.merge( refined );
}

template < class K_Simplex >
std::set< std::shared_ptr< K_Simplex > > Mesh< K_Simplex >::refine_red( const std::set< std::shared_ptr< K_Simplex > >& R,
                                                                        std::set< std::shared_ptr< K_Simplex > >&       U )
{
   std::set< std::shared_ptr< K_Simplex > > refined;

   for ( auto& el : R )
   {
      // remove green edges
      bool check_subelements = el->kill_children();

      auto subelements = refine_element_red( el );

      // mark el as processed
      U.erase( el );

      // mark subelements as unprocessed if necessary
      if ( check_subelements )
      {
         U.merge( subelements );
      }

      refined.merge( subelements );
   }

   return refined;
}

template < class K_Simplex >
void Mesh< K_Simplex >::remove_green_edges( std::set< std::shared_ptr< K_Simplex > >& R )
{
   auto T_cpy = _T;

   for ( auto& el : T_cpy )
   {
      if ( el->has_green_edge() )
      {
         _T.erase( el );
         _T.insert( el->get_parent() );

         if ( R.erase( el ) )
         {
            R.insert( el->get_parent() );
         }
      }
   }
}

template <>
std::set< std::shared_ptr< Simplex2 > >
    Mesh< Simplex2 >::find_elements_for_red_refinement( const std::set< std::shared_ptr< Simplex2 > >& U )
{
   std::set< std::shared_ptr< Simplex2 > > R;

   for ( auto& el : U )
   {
      if ( el->vertices_on_edges() > 1 )
      {
         R.insert( el );
      }
   }

   return R;
}

template <>
std::set< std::shared_ptr< Simplex3 > >
    Mesh< Simplex3 >::find_elements_for_red_refinement( const std::set< std::shared_ptr< Simplex3 > >& U )
{
   std::set< std::shared_ptr< Simplex3 > > R;

   for ( auto& el : U )
   {
      int n_red = 0;

      for ( auto& face : el->get_faces() )
      {
         if ( face->vertices_on_edges() > 1 )
         {
            if ( face->get_children().size() == 2 )
            {
               // remove green edge from face
               face->kill_children();
            }

            if ( not face->has_children() )
            {
               // apply red refinement to face
               refine_face_red( _vertices, face );
            }

            ++n_red;
         }
      }

      // if more than one face has been red-refined, mark cell for red refinement
      if ( n_red > 1 )
      {
         R.insert( el );
      }
   }

   return R;
}

template <>
std::set< std::shared_ptr< Simplex2 > > Mesh< Simplex2 >::refine_green( std::set< std::shared_ptr< Simplex2 > >& U )
{
   std::set< std::shared_ptr< Simplex2 > > refined;
   std::set< std::shared_ptr< Simplex2 > > U_cpy = U;

   for ( auto& el : U_cpy )
   {
      // count number of new vertices on the edges of el
      int new_vertices = el->vertices_on_edges();

      if ( new_vertices > 0 )
      {
         assert( not el->has_green_edge() );
         assert( new_vertices == 1 );

         /* if green refinement had been applied to the same element before,
            nothing has to be done
         */
         if ( el->has_children() )
         {
            for ( auto& child : el->get_children() )
            {
               refined.insert( child );
            }
         }
         else
         {
            refined.merge( refine_face_green( el ) );
         }

         // mark el as processed
         U.erase( el );
      }
   }

   return refined;
}

template <>
std::set< std::shared_ptr< Simplex3 > > Mesh< Simplex3 >::refine_green( std::set< std::shared_ptr< Simplex3 > >& U )
{
   std::set< std::shared_ptr< Simplex3 > > refined;
   std::set< std::shared_ptr< Simplex3 > > U_cpy = U;

   auto keepChildren = [&]( std::shared_ptr< Simplex3 > el ) {
      for ( auto& child : el->get_children() )
      {
         refined.insert( child );
      }
   };

   for ( auto& el : U_cpy )
   {
      int new_vertices = el->vertices_on_edges();
      assert( new_vertices <= 3 );

      switch ( new_vertices )
      {
      case 0:
         continue;
         break;

      case 1:
         if ( el->has_children() )
         {
            assert( el->get_children().size() == 2 );
            keepChildren( el );
         }
         else
         {
            refined.merge( refine_cell_green_1( el ) );
         }
         break;

      case 2:
         if ( el->has_children() and el->get_children().size() == 4 )
         {
            keepChildren( el );
         }
         else
         {
            assert( el->get_children().size() == 0 or el->get_children().size() == 2 );
            el->kill_children();
            refined.merge( refine_cell_green_2( el ) );
         }
         break;

      case 3:
         if ( el->has_children() and el->get_children().size() == 4 )
         {
            keepChildren( el );
         }
         else
         {
            assert( el->get_children().size() == 0 or el->get_children().size() == 2 );
            el->kill_children();
            refined.merge( refine_cell_green_3( el ) );
         }
         break;

      default:
         assert( false );
         break;
      }

      // mark el as processed
      U.erase( el );
   }

   return refined;
}

template <>
std::set< std::shared_ptr< Simplex3 > > Mesh< Simplex3 >::refine_element_red( std::shared_ptr< Simplex3 > element )
{
   return refine_cell_red( _vertices, element );
}

template <>
std::set< std::shared_ptr< Simplex2 > > Mesh< Simplex2 >::refine_element_red( std::shared_ptr< Simplex2 > element )
{
   return refine_face_red( _vertices, element );
}

template < class K_Simplex >
std::pair< real_t, real_t > Mesh< K_Simplex >::min_max_angle() const
{
   std::pair< real_t, real_t > mm{ 10, 0 };

   for ( auto& el : _T )
   {
      auto mm_el = el->min_max_angle( _vertices );

      mm.first  = std::min( mm.first, mm_el.first );
      mm.second = std::max( mm.second, mm_el.second );
   }

   return mm;
}

template < class K_Simplex >
real_t Mesh< K_Simplex >::volume() const
{
   real_t v_tot = 0;

   for ( auto& el : _T )
   {
      v_tot += el->volume( _vertices );
   }

   return v_tot;
}

template class Mesh< Simplex2 >;
template class Mesh< Simplex3 >;

} // namespace adaptiveRefinement
} // namespace hyteg
