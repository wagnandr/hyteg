#pragma once

#include "refine_face.hpp"
#include "simplexFactory.hpp"

namespace hyteg {
namespace adaptiveRefinement {

/* apply red refinement to cell and add required vertices to vertices
      @return sub-elements
   */
inline std::set< std::shared_ptr< Simplex3 > > refine_cell_red( std::vector< Point3D >&     vertices,
                                                                std::shared_ptr< Simplex3 > cell )
{
   // === split faces ===
   for ( auto& face : cell->get_faces() )
   {
      if ( face->get_children().size() == 2 )
      {
         // remove green edge from face
         face->kill_children();
      }

      if ( not face->has_children() )
      {
         // apply red refinement to face
         refine_face_red( vertices, face );
      }
   }

   // === collect vertex indices ===
   std::vector< int64_t > ref_vertices( 10 );

   // old vertices
   for ( int i = 0; i < 4; ++i )
   {
      ref_vertices[i] = cell->get_vertices()[i];
   }
   // new vertices
   for ( int i = 0; i < 6; ++i )
   {
      ref_vertices[i + 4] = cell->get_edges()[i]->get_midpoint_idx();
   }

   SimplexFactory fac( cell, ref_vertices );

   // === collect edges ===
   // ref_edges[{i,j}] is the edge connecting ref_vertices[i] and ref_vertices[j]

   // edge-children
   for ( int i = 0; i < 3; ++i )
   {
      auto child_edges_bottom = cell->get_edges()[i]->get_children_sorted( { ref_vertices[i], ref_vertices[( i + 1 ) % 3] } );
      auto child_edges_top    = cell->get_edges()[i + 3]->get_children_sorted( { ref_vertices[i], ref_vertices[3] } );
      fac.add_edge( i, i + 4, child_edges_bottom[0] );
      fac.add_edge( ( i + 1 ) % 3, i + 4, child_edges_bottom[1] );
      fac.add_edge( i, i + 7, child_edges_top[0] );
      fac.add_edge( 3, i + 7, child_edges_top[1] );
   }

   // edges on face interior
   auto innerface   = cell->get_faces()[0]->get_children()[3];
   auto child_edges = innerface->get_edges_sorted( { ref_vertices[4], ref_vertices[5], ref_vertices[6] } );
   fac.add_edge( 4, 5, child_edges[0] );
   fac.add_edge( 5, 6, child_edges[1] );
   fac.add_edge( 4, 6, child_edges[2] );

   innerface   = cell->get_faces()[1]->get_children()[3];
   child_edges = innerface->get_edges_sorted( { ref_vertices[4], ref_vertices[7], ref_vertices[8] } );
   fac.add_edge( 4, 7, child_edges[0] );
   fac.add_edge( 7, 8, child_edges[1] );
   fac.add_edge( 4, 8, child_edges[2] );

   innerface   = cell->get_faces()[2]->get_children()[3];
   child_edges = innerface->get_edges_sorted( { ref_vertices[5], ref_vertices[8], ref_vertices[9] } );
   fac.add_edge( 5, 8, child_edges[0] );
   fac.add_edge( 8, 9, child_edges[1] );
   fac.add_edge( 5, 9, child_edges[2] );

   innerface   = cell->get_faces()[3]->get_children()[3];
   child_edges = innerface->get_edges_sorted( { ref_vertices[6], ref_vertices[7], ref_vertices[9] } );
   fac.add_edge( 6, 7, child_edges[0] );
   fac.add_edge( 7, 9, child_edges[1] );
   fac.add_edge( 6, 9, child_edges[2] );

   // === new edge in cell interior ===
   // find shortest possible edge (either {4,9}, {5,7} or {6,8})
   auto distPt  = [&]( const Point3D& x, const Point3D& y ) { return ( x - y ).norm(); };
   auto edgeLen = [&]( const std::array< int, 2 >& edge ) {
      return distPt( vertices[ref_vertices[edge[0]]], vertices[ref_vertices[edge[1]]] );
   };
   std::set< std::array< int, 2 > > possible_edges{ { 4, 9 }, { 5, 7 }, { 6, 8 } };
   std::array< int, 2 >             new_edge{ 4, 9 };
   real_t                           new_edge_len = edgeLen( new_edge );

   for ( auto& edge : possible_edges )
   {
      auto len = edgeLen( edge );
      if ( len < new_edge_len )
      {
         new_edge     = edge;
         new_edge_len = len;
      }
   }

   auto vtx_a = new_edge[0];
   auto vtx_b = new_edge[1];

   fac.make_edge( vtx_a, vtx_b );

   // remaining vertices, i.e. {4,5,6,7,8,9}\{vtx_a,vtx_b}
   std::set< int > vertices_wo_ab{ 4, 5, 6, 7, 8, 9 };
   vertices_wo_ab.erase( vtx_a );
   vertices_wo_ab.erase( vtx_b );

   // === collect faces ===

   // face[0]:
   auto child_faces = cell->get_faces()[0]->get_children_sorted( { ref_vertices[0], ref_vertices[1], ref_vertices[2] } );
   fac.add_face( 0, 4, 6, child_faces[0] );
   fac.add_face( 1, 4, 5, child_faces[1] );
   fac.add_face( 2, 5, 6, child_faces[2] );
   fac.add_face( 4, 5, 6, child_faces[3] );

   // face[1]:
   child_faces = cell->get_faces()[1]->get_children_sorted( { ref_vertices[0], ref_vertices[1], ref_vertices[3] } );
   fac.add_face( 0, 4, 7, child_faces[0] );
   fac.add_face( 1, 4, 8, child_faces[1] );
   fac.add_face( 3, 7, 8, child_faces[2] );
   fac.add_face( 4, 7, 8, child_faces[3] );

   // face[2]:
   child_faces = cell->get_faces()[2]->get_children_sorted( { ref_vertices[1], ref_vertices[2], ref_vertices[3] } );
   fac.add_face( 1, 5, 8, child_faces[0] );
   fac.add_face( 2, 5, 9, child_faces[1] );
   fac.add_face( 3, 8, 9, child_faces[2] );
   fac.add_face( 5, 8, 9, child_faces[3] );

   // face[3]:
   child_faces = cell->get_faces()[3]->get_children_sorted( { ref_vertices[2], ref_vertices[0], ref_vertices[3] } );
   fac.add_face( 2, 6, 9, child_faces[0] );
   fac.add_face( 0, 6, 7, child_faces[1] );
   fac.add_face( 3, 7, 9, child_faces[2] );
   fac.add_face( 6, 7, 9, child_faces[3] );

   // new inner faces
   fac.make_face( 4, 5, 8 );
   fac.make_face( 5, 6, 9 );
   fac.make_face( 4, 6, 7 );
   fac.make_face( 7, 8, 9 );

   for ( auto& vtx : vertices_wo_ab )
   {
      fac.make_face( vtx, vtx_a, vtx_b );
   }

   // === split cell ===

   // new outer cells
   cell->add_child( fac.make_cell( 0, 4, 6, 7 ) );
   cell->add_child( fac.make_cell( 4, 1, 5, 8 ) );
   cell->add_child( fac.make_cell( 6, 5, 2, 9 ) );
   cell->add_child( fac.make_cell( 7, 8, 9, 3 ) );

   /* find correct vertices for new inner cells, i.e. vtx_a,vtx_b and
      two other vertices from {4,5,6,7,8,9}\{vtx_a,vtx_b} that do form
      an existing edge. This is ensured by excluding vertices forming
      one of the edges in {{4, 9}, {5, 7}, {6, 8}}.
   */
   std::vector< int > test_vertices;
   std::copy( vertices_wo_ab.begin(), vertices_wo_ab.end(), std::back_inserter( test_vertices ) );

   for ( int c = 0; c < 4; ++c )
   {
      int vtx_c = test_vertices[c];
      for ( int d = c + 1; d < 4; ++d )
      {
         int vtx_d = test_vertices[d];

         Idx< 2 > edge_cd{ vtx_c, vtx_d };

         if ( possible_edges.count( edge_cd ) )
         {
            continue;
         }

         cell->add_child( fac.make_cell( vtx_a, vtx_b, vtx_c, vtx_d ) );
      }
   }

   return std::set< std::shared_ptr< Simplex3 > >( cell->get_children().begin(), cell->get_children().end() );
}

/* apply green refinement to cell with 1 new vertex
      @return sub-elements
   */
inline std::set< std::shared_ptr< Simplex3 > > refine_cell_green_1( std::shared_ptr< Simplex3 > cell )
{
   auto faces = cell->refined_faces();
   WALBERLA_ASSERT( faces[GREEN].size() == 2 and faces[RED].size() == 0 );

   // === split faces ===

   for ( auto& face : faces[GREEN] )
   {
      if ( not face->has_children() )
      {
         refine_face_green( face );
      }
   }

   // === collect vertex indices ===
   std::vector< int64_t > ref_vertices( 5 );

   // vertices at split edge
   std::shared_ptr< Simplex1 > split_edge;

   for ( auto& edge : cell->get_edges() )
   {
      ref_vertices[4] = edge->get_midpoint_idx();

      if ( ref_vertices[4] >= 0 )
      {
         split_edge = edge;
         break;
      }
   }
   WALBERLA_ASSERT( split_edge );

   ref_vertices[1] = split_edge->get_vertices()[0];
   ref_vertices[2] = split_edge->get_vertices()[1];

   auto is_unbranded = [&]( int64_t vtx ) {
      return ( vtx != ref_vertices[1] and vtx != ref_vertices[2] and vtx != ref_vertices[4] );
   };
   std::vector< int64_t > unbranded;
   std::copy_if( cell->get_vertices().begin(), cell->get_vertices().end(), std::back_inserter( unbranded ), is_unbranded );

   ref_vertices[0] = unbranded[0];
   ref_vertices[3] = unbranded[1];

   SimplexFactory fac( cell, ref_vertices );

   // === collect faces ===

   auto face0        = cell->get_Face( ref_vertices[0], ref_vertices[1], ref_vertices[2] );
   auto child_faces0 = face0->get_children_sorted( { ref_vertices[1], ref_vertices[2], ref_vertices[0] } );

   auto face2        = cell->get_Face( ref_vertices[3], ref_vertices[1], ref_vertices[2] );
   auto child_faces2 = face2->get_children_sorted( { ref_vertices[1], ref_vertices[2], ref_vertices[3] } );

   fac.add_face( 0, 1, 3, cell->get_Face( ref_vertices[0], ref_vertices[1], ref_vertices[3] ) );
   fac.add_face( 0, 2, 3, cell->get_Face( ref_vertices[0], ref_vertices[2], ref_vertices[3] ) );
   fac.add_face( 0, 1, 4, child_faces0[0] );
   fac.add_face( 0, 2, 4, child_faces0[1] );
   fac.add_face( 1, 3, 4, child_faces2[0] );
   fac.add_face( 2, 3, 4, child_faces2[1] );

   // === collect edges ===

   fac.add_edge( 0, 1, cell->get_Edge( ref_vertices[0], ref_vertices[1] ) );
   fac.add_edge( 0, 2, cell->get_Edge( ref_vertices[0], ref_vertices[2] ) );
   fac.add_edge( 0, 3, cell->get_Edge( ref_vertices[0], ref_vertices[3] ) );
   fac.add_edge( 1, 3, cell->get_Edge( ref_vertices[1], ref_vertices[3] ) );
   fac.add_edge( 2, 3, cell->get_Edge( ref_vertices[2], ref_vertices[3] ) );
   fac.add_edge( 0, 4, child_faces0[0]->get_Edge( ref_vertices[0], ref_vertices[4] ) );
   fac.add_edge( 1, 4, child_faces0[0]->get_Edge( ref_vertices[1], ref_vertices[4] ) );
   fac.add_edge( 2, 4, child_faces0[1]->get_Edge( ref_vertices[2], ref_vertices[4] ) );
   fac.add_edge( 3, 4, child_faces2[0]->get_Edge( ref_vertices[3], ref_vertices[4] ) );

   // === new interior face ===

   fac.make_face( 0, 3, 4 );

   // === split cell ===

   cell->add_child( fac.make_cell( 0, 1, 4, 3 ) );
   cell->add_child( fac.make_cell( 0, 2, 4, 3 ) );

   return std::set< std::shared_ptr< Simplex3 > >( cell->get_children().begin(), cell->get_children().end() );
}

/* apply green refinement to cell with 2 new vertices
      @return sub-elements
   */
inline std::set< std::shared_ptr< Simplex3 > > refine_cell_green_2( std::shared_ptr< Simplex3 > cell )
{
   auto faces = cell->refined_faces();
   WALBERLA_ASSERT( faces[GREEN].size() == 4 and faces[RED].size() == 0 );

   // === split faces ===

   for ( auto& face : faces[GREEN] )
   {
      if ( not face->has_children() )
      {
         refine_face_green( face );
      }
   }

   // === collect vertex indices ===
   std::vector< int64_t > ref_vertices( 6 );

   // vertices at split edges
   std::array< std::shared_ptr< Simplex1 >, 2 > split_edges;

   int split_idx = 0;
   for ( auto& edge : cell->get_edges() )
   {
      if ( split_idx >= 2 )
         break;

      ref_vertices[4 + split_idx] = edge->get_midpoint_idx();

      if ( ref_vertices[4 + split_idx] >= 0 )
      {
         split_edges[split_idx] = edge;
         ++split_idx;
      }
   }
   WALBERLA_ASSERT( split_edges[0] and split_edges[1] );

   ref_vertices[1] = split_edges[0]->get_vertices()[0];
   ref_vertices[2] = split_edges[0]->get_vertices()[1];

   ref_vertices[0] = split_edges[1]->get_vertices()[0];
   ref_vertices[3] = split_edges[1]->get_vertices()[1];

   SimplexFactory fac( cell, ref_vertices );

   // === collect faces ===

   auto face0        = cell->get_Face( ref_vertices[0], ref_vertices[1], ref_vertices[2] );
   auto child_faces0 = face0->get_children_sorted( { ref_vertices[1], ref_vertices[2], ref_vertices[0] } );

   auto face1        = cell->get_Face( ref_vertices[0], ref_vertices[1], ref_vertices[3] );
   auto child_faces1 = face1->get_children_sorted( { ref_vertices[0], ref_vertices[3], ref_vertices[1] } );

   auto face2        = cell->get_Face( ref_vertices[3], ref_vertices[1], ref_vertices[2] );
   auto child_faces2 = face2->get_children_sorted( { ref_vertices[1], ref_vertices[2], ref_vertices[3] } );

   auto face3        = cell->get_Face( ref_vertices[0], ref_vertices[2], ref_vertices[3] );
   auto child_faces3 = face3->get_children_sorted( { ref_vertices[0], ref_vertices[3], ref_vertices[2] } );

   fac.add_face( 0, 1, 4, child_faces0[0] );
   fac.add_face( 0, 2, 4, child_faces0[1] );
   fac.add_face( 0, 1, 5, child_faces1[0] );
   fac.add_face( 1, 3, 5, child_faces1[1] );
   fac.add_face( 1, 3, 4, child_faces2[0] );
   fac.add_face( 2, 3, 4, child_faces2[1] );
   fac.add_face( 0, 2, 5, child_faces3[0] );
   fac.add_face( 2, 3, 5, child_faces3[1] );

   // === collect edges ===

   fac.add_edge( 0, 1, cell->get_Edge( ref_vertices[0], ref_vertices[1] ) );
   fac.add_edge( 0, 2, cell->get_Edge( ref_vertices[0], ref_vertices[2] ) );
   fac.add_edge( 1, 3, cell->get_Edge( ref_vertices[1], ref_vertices[3] ) );
   fac.add_edge( 2, 3, cell->get_Edge( ref_vertices[2], ref_vertices[3] ) );
   fac.add_edge( 0, 4, child_faces0[0]->get_Edge( ref_vertices[0], ref_vertices[4] ) );
   fac.add_edge( 1, 4, child_faces0[0]->get_Edge( ref_vertices[1], ref_vertices[4] ) );
   fac.add_edge( 2, 4, child_faces0[1]->get_Edge( ref_vertices[2], ref_vertices[4] ) );
   fac.add_edge( 3, 4, child_faces2[0]->get_Edge( ref_vertices[3], ref_vertices[4] ) );
   fac.add_edge( 0, 5, child_faces1[0]->get_Edge( ref_vertices[0], ref_vertices[5] ) );
   fac.add_edge( 1, 5, child_faces1[0]->get_Edge( ref_vertices[1], ref_vertices[5] ) );
   fac.add_edge( 2, 5, child_faces3[0]->get_Edge( ref_vertices[2], ref_vertices[5] ) );
   fac.add_edge( 3, 5, child_faces3[1]->get_Edge( ref_vertices[3], ref_vertices[5] ) );

   // new edge in cell interior
   fac.make_edge( 4, 5, GREEN );

   // === new faces in cell interior

   for ( int k = 0; k < 4; ++k )
   {
      fac.make_face( k, 4, 5 );
   }

   // === split cell ===

   cell->add_child( fac.make_cell( 0, 1, 4, 5 ) );
   cell->add_child( fac.make_cell( 0, 2, 4, 5 ) );
   cell->add_child( fac.make_cell( 1, 3, 4, 5 ) );
   cell->add_child( fac.make_cell( 2, 3, 4, 5 ) );

   return std::set< std::shared_ptr< Simplex3 > >( cell->get_children().begin(), cell->get_children().end() );
}

/* apply green refinement to cell with 3 new vertices
      @return sub-elements
   */
inline std::set< std::shared_ptr< Simplex3 > > refine_cell_green_3( std::shared_ptr< Simplex3 > cell )
{
   auto faces = cell->refined_faces();
   WALBERLA_ASSERT( faces[GREEN].size() == 3 and faces[RED].size() == 1 );

   // === split faces ===

   for ( auto& face : faces[GREEN] )
   {
      if ( not face->has_children() )
      {
         refine_face_green( face );
      }
   }

   // === collect vertex indices ===
   std::vector< int64_t > ref_vertices( 7 );

   // vertices at red face
   auto red_vtxs  = faces[RED][0]->get_vertices();
   auto red_edges = faces[RED][0]->get_edges_sorted( red_vtxs );

   for ( int k = 0; k < 3; ++k )
   {
      ref_vertices[k]     = red_vtxs[k];
      ref_vertices[k + 4] = red_edges[k]->get_midpoint_idx();
   }

   // top vertex
   auto old_vtxs = cell->get_vertices();

   for ( int k = 0; k < 4; ++k )
   {
      bool found = true;

      for ( int bottom = 0; bottom < 3; ++bottom )
      {
         if ( old_vtxs[k] == ref_vertices[bottom] )
         {
            found = false;
         }
      }

      if ( found )
      {
         ref_vertices[3] = old_vtxs[k];
         break;
      }
   }

   SimplexFactory fac( cell, ref_vertices );

   // === collect faces ===

   auto face0        = cell->get_Face( ref_vertices[0], ref_vertices[1], ref_vertices[2] );
   auto child_faces0 = face0->get_children_sorted( { ref_vertices[0], ref_vertices[1], ref_vertices[2] } );

   auto face1        = cell->get_Face( ref_vertices[0], ref_vertices[1], ref_vertices[3] );
   auto child_faces1 = face1->get_children_sorted( { ref_vertices[0], ref_vertices[1], ref_vertices[3] } );

   auto face2        = cell->get_Face( ref_vertices[1], ref_vertices[2], ref_vertices[3] );
   auto child_faces2 = face2->get_children_sorted( { ref_vertices[1], ref_vertices[2], ref_vertices[3] } );

   auto face3        = cell->get_Face( ref_vertices[2], ref_vertices[0], ref_vertices[3] );
   auto child_faces3 = face3->get_children_sorted( { ref_vertices[2], ref_vertices[0], ref_vertices[3] } );

   fac.add_face( 0, 4, 6, child_faces0[0] );
   fac.add_face( 1, 4, 5, child_faces0[1] );
   fac.add_face( 2, 5, 6, child_faces0[2] );
   fac.add_face( 4, 5, 6, child_faces0[3] );

   fac.add_face( 0, 3, 4, child_faces1[0] );
   fac.add_face( 1, 3, 4, child_faces1[1] );

   fac.add_face( 1, 3, 5, child_faces2[0] );
   fac.add_face( 2, 3, 5, child_faces2[1] );

   fac.add_face( 2, 3, 6, child_faces3[0] );
   fac.add_face( 0, 3, 6, child_faces3[1] );

   // === collect edges ===

   // old edges
   fac.add_edge( 0, 3, cell->get_Edge( ref_vertices[0], ref_vertices[3] ) );
   fac.add_edge( 1, 3, cell->get_Edge( ref_vertices[1], ref_vertices[3] ) );
   fac.add_edge( 2, 3, cell->get_Edge( ref_vertices[2], ref_vertices[3] ) );
   // red edges
   fac.add_edge( 0, 4, child_faces0[0]->get_Edge( ref_vertices[0], ref_vertices[4] ) );
   fac.add_edge( 0, 6, child_faces0[0]->get_Edge( ref_vertices[0], ref_vertices[6] ) );
   fac.add_edge( 1, 4, child_faces0[1]->get_Edge( ref_vertices[1], ref_vertices[4] ) );
   fac.add_edge( 1, 5, child_faces0[1]->get_Edge( ref_vertices[1], ref_vertices[5] ) );
   fac.add_edge( 2, 5, child_faces0[2]->get_Edge( ref_vertices[2], ref_vertices[5] ) );
   fac.add_edge( 2, 6, child_faces0[2]->get_Edge( ref_vertices[2], ref_vertices[6] ) );
   fac.add_edge( 4, 5, child_faces0[3]->get_Edge( ref_vertices[4], ref_vertices[5] ) );
   fac.add_edge( 5, 6, child_faces0[3]->get_Edge( ref_vertices[5], ref_vertices[6] ) );
   fac.add_edge( 4, 6, child_faces0[3]->get_Edge( ref_vertices[6], ref_vertices[4] ) );
   // green edges
   fac.add_edge( 3, 4, child_faces1[0]->get_Edge( ref_vertices[3], ref_vertices[4] ) );
   fac.add_edge( 3, 5, child_faces2[0]->get_Edge( ref_vertices[3], ref_vertices[5] ) );
   fac.add_edge( 3, 6, child_faces3[0]->get_Edge( ref_vertices[3], ref_vertices[6] ) );

   // === new faces in cell interior

   fac.make_face( 3, 4, 5 );
   fac.make_face( 3, 5, 6 );
   fac.make_face( 3, 4, 6 );

   // === split cell ===

   cell->add_child( fac.make_cell( 0, 4, 6, 3 ) );
   cell->add_child( fac.make_cell( 1, 4, 5, 3 ) );
   cell->add_child( fac.make_cell( 2, 5, 6, 3 ) );
   cell->add_child( fac.make_cell( 4, 5, 6, 3 ) );

   return std::set< std::shared_ptr< Simplex3 > >( cell->get_children().begin(), cell->get_children().end() );
}

} // namespace adaptiveRefinement
} // namespace hyteg
