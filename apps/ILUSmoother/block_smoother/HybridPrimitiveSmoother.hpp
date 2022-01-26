/*
 * Copyright (c) 2020 Andreas Wagner, Daniel Drzisga.
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

#include <utility>

#include "PrimitiveSmoothers.hpp"

namespace hyteg {

/// A block smoother for a multigrid hierarchy.
template < typename OperatorType >
class HybridPrimitiveSmoother : public Solver< OperatorType >
{
 public:
   using FunctionType = typename OperatorType::srcType;

   explicit HybridPrimitiveSmoother( std::shared_ptr< PrimitiveStorage > storage, uint_t minLevel, uint_t maxLevel )
   : storage_( std::move( storage ) )
   , flag_( hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary )
   , consecutiveSmoothingStepsOnVertices_( 1 )
   , consecutiveSmoothingStepsOnEdges_( 1 )
   , consecutiveSmoothingStepsOnFaces_( 1 )
   , consecutiveSmoothingStepsOnCells_( 1 )
   , consecutiveBackwardsSmoothingStepsOnVertices_( 0 )
   , consecutiveBackwardsSmoothingStepsOnEdges_( 0 )
   , consecutiveBackwardsSmoothingStepsOnFaces_( 0 )
   , minLevel_( minLevel )
   , maxLevel_( maxLevel )
   // TODO: Add sensible defaults here
   , vertex_smoother( nullptr )
   , edge_smoother( nullptr )
   , face_smoother( nullptr )
   , cell_smoother( nullptr )
   {}

   /// Sets the vertex smoother
   void setVertexSmoother( std::shared_ptr< VertexSmoother< OperatorType > > vs ) { vertex_smoother = std::move( vs ); }

   /// Sets the edge smoother
   void setEdgeSmoother( std::shared_ptr< EdgeSmoother< OperatorType > > es ) { edge_smoother = std::move( es ); }

   /// Sets the face smoother
   void setFaceSmoother( std::shared_ptr< FaceSmoother< OperatorType > > fs ) { face_smoother = std::move( fs ); }

   /// Sets the cell smoother
   void setCellSmoother( std::shared_ptr< CellSmoother< OperatorType > > cs ) { cell_smoother = std::move( cs ); }

   /// Sets the number of consecutive smoothing steps applied on the vertex primitive without communication.
   void setConsecutiveSmoothingStepsOnVertices( uint_t steps ) { consecutiveSmoothingStepsOnVertices_ = steps; }

   /// Sets the number of consecutive smoothing steps applied on the edge primitive without communication.
   void setConsecutiveSmoothingStepsOnEdges( uint_t steps ) { consecutiveSmoothingStepsOnEdges_ = steps; }

   /// Sets the number of consecutive smoothing steps applied on the face primitive without communication.
   void setConsecutiveSmoothingStepsOnFaces( uint_t steps ) { consecutiveSmoothingStepsOnFaces_ = steps; }

   /// Sets the number of consecutive smoothing steps applied on the cell primitive without communication.
   void setConsecutiveSmoothingStepsOnCells( uint_t steps ) { consecutiveSmoothingStepsOnCells_ = steps; }

   /// Sets the number of consecutive smoothing steps applied on the vertex primitive without communication.
   void setConsecutiveBackwardsSmoothingStepsOnVertices( uint_t steps ) { consecutiveBackwardsSmoothingStepsOnVertices_ = steps; }

   /// Sets the number of consecutive smoothing steps applied on the edge primitive without communication.
   void setConsecutiveBackwardsSmoothingStepsOnEdges( uint_t steps ) { consecutiveBackwardsSmoothingStepsOnEdges_ = steps; }

   /// Sets the number of consecutive smoothing steps applied on the face primitive without communication.
   void setConsecutiveBackwardsSmoothingStepsOnFaces( uint_t steps ) { consecutiveBackwardsSmoothingStepsOnFaces_ = steps; }

   /// Applies GS on the vertices.
   void smooth_gs_on_vertices( const OperatorType& op, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      for ( auto& it : storage_->getVertices() )
      {
         Vertex& vertex = *it.second;

         const DoFType vertexBC = x.getBoundaryCondition().getBoundaryType( vertex.getMeshBoundaryFlag() );
         if ( testFlag( vertexBC, hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary ) )
         {
            vertex_smoother->smooth( op, level, vertex, x, b );
         }
      }
   }

   /// Applies GS on the edges.
   void smooth_gs_on_edges( const OperatorType& op, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      for ( auto& it : storage_->getEdges() )
      {
         Edge& edge = *it.second;

         const DoFType edgeBC = x.getBoundaryCondition().getBoundaryType( edge.getMeshBoundaryFlag() );
         if ( testFlag( edgeBC, hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary ) )
         {
            edge_smoother->smooth( op, level, edge, x, b );
         }
      }
   }

   /// Applies GS on the edges.
   void smooth_gs_backwards_on_edges( const OperatorType& op, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      for ( auto& it : storage_->getEdges() )
      {
         Edge& edge = *it.second;

         const DoFType edgeBC = x.getBoundaryCondition().getBoundaryType( edge.getMeshBoundaryFlag() );
         if ( testFlag( edgeBC, hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary ) )
         {
            edge_smoother->smooth_backwards( op, level, edge, x, b );
         }
      }
   }

   /// Applies GS on the faces.
   void smooth_gs_on_faces( const OperatorType& op, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      for ( auto& it : storage_->getFaces() )
      {
         Face& face = *it.second;

         const DoFType faceBC = x.getBoundaryCondition().getBoundaryType( face.getMeshBoundaryFlag() );
         if ( testFlag( faceBC, hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary ) )
         {
            face_smoother->smooth( op, level, face, x, b );
         }
      }
   }

   /// Applies GS on the faces.
   void smooth_gs_backwards_on_faces( const OperatorType& op, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      for ( auto& it : storage_->getFaces() )
      {
         Face& face = *it.second;

         const DoFType faceBC = x.getBoundaryCondition().getBoundaryType( face.getMeshBoundaryFlag() );
         if ( testFlag( faceBC, hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary ) )
         {
            face_smoother->smooth_backwards( op, level, face, x, b );
         }
      }
   }

   void smooth_gs_on_cells( const OperatorType& op, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      for ( auto& it : storage_->getCells() )
      {
         Cell& cell = *it.second;

         const DoFType cellBC = x.getBoundaryCondition().getBoundaryType( cell.getMeshBoundaryFlag() );
         if ( testFlag( cellBC, hyteg::Inner | hyteg::NeumannBoundary | hyteg::FreeslipBoundary ) )
         {
            cell_smoother->smooth_backwards( op, level, cell, x, b );
         }
      }
   }

   void solve( const OperatorType& op, const FunctionType& x, const FunctionType& b, const walberla::uint_t level ) override
   {
      x.template communicate< Vertex, Edge >( level );
      x.template communicate< Edge, Face >( level );
      x.template communicate< Face, Cell >( level );

      b.template communicate< Vertex, Edge >( level );
      b.template communicate< Edge, Face >( level );
      b.template communicate< Face, Cell >( level );

      x.template communicate< Cell, Face >( level );
      x.template communicate< Face, Edge >( level );
      x.template communicate< Edge, Vertex >( level );

      for ( uint_t it = 0; it < consecutiveSmoothingStepsOnVertices_; ++it )
         smooth_gs_on_vertices( op, x, b, level );

      x.template communicate< Vertex, Edge >( level );

      // these are necessary for our block-vertex smoother:
      x.template communicate< Edge, Face >( level );
      x.template communicate< Face, Edge >( level );

      for ( uint_t it = 0; it < consecutiveSmoothingStepsOnEdges_; ++it )
         smooth_gs_on_edges( op, x, b, level );

      x.template communicate< Edge, Face >( level );

      for ( uint_t pit = 0; pit < consecutiveSmoothingStepsOnFaces_; ++pit )
         smooth_gs_on_faces( op, x, b, level );

      x.template communicate< Face, Cell >( level );

      for ( uint_t pit = 0; pit < consecutiveSmoothingStepsOnCells_; ++pit )
         smooth_gs_on_cells( op, x, b, level );

      x.template communicate< Cell, Face >( level );

      for ( uint_t pit = 0; pit < consecutiveBackwardsSmoothingStepsOnFaces_; ++pit )
         smooth_gs_backwards_on_faces( op, x, b, level );

      x.template communicate< Face, Edge >( level );

      for ( uint_t it = 0; it < consecutiveBackwardsSmoothingStepsOnEdges_; ++it )
         smooth_gs_backwards_on_edges( op, x, b, level );

      x.template communicate< Edge, Vertex >( level );

      for ( uint_t it = 0; it < consecutiveBackwardsSmoothingStepsOnVertices_; ++it )
         smooth_gs_on_vertices( op, x, b, level );
   }

 private:
   std::shared_ptr< PrimitiveStorage > storage_;
   DoFType                             flag_;
   uint_t                              consecutiveSmoothingStepsOnVertices_;
   uint_t                              consecutiveSmoothingStepsOnEdges_;
   uint_t                              consecutiveSmoothingStepsOnFaces_;
   uint_t                              consecutiveSmoothingStepsOnCells_;
   uint_t                              consecutiveBackwardsSmoothingStepsOnVertices_;
   uint_t                              consecutiveBackwardsSmoothingStepsOnEdges_;
   uint_t                              consecutiveBackwardsSmoothingStepsOnFaces_;
   uint_t                              minLevel_;
   uint_t                              maxLevel_;

   /// Smoother for the macro vertices.
   std::shared_ptr< VertexSmoother< OperatorType > > vertex_smoother;

   /// Smoother for the macro edges.
   std::shared_ptr< EdgeSmoother< OperatorType > > edge_smoother;

   /// Smoother for the macro faces.
   std::shared_ptr< FaceSmoother< OperatorType > > face_smoother;

   /// Smoother for the macro cells.
   std::shared_ptr< CellSmoother< OperatorType > > cell_smoother;
};

} // namespace hyteg
