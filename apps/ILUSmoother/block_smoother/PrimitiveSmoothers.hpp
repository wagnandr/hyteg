/*
* Copyright (c) 2021 Andreas Wagner.
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

namespace hyteg {

using walberla::uint_t;

// forward declarations:
class Vertex;
class Edge;
class Face;
class Cell;

template < typename OperatorType >
class VertexSmoother
{
 public:
   using VSFunctionType = typename OperatorType::srcType;

   virtual ~VertexSmoother() = default;

   virtual void smooth( const OperatorType&, uint_t level, Vertex&, const VSFunctionType& x, const VSFunctionType& b ) = 0;
};

template < typename OperatorType >
class EdgeSmoother
{
 public:
   using ESFunctionType = typename OperatorType::srcType;

   virtual ~EdgeSmoother() = default;

   virtual void smooth( const OperatorType&, uint_t level, Edge&, const ESFunctionType& x, const ESFunctionType& b ) = 0;

   virtual void
       smooth_backwards( const OperatorType&, uint_t level, Edge&, const ESFunctionType& x, const ESFunctionType& b ) = 0;
};

template < typename OperatorType >
class FaceSmoother
{
 public:
   using FSFunctionType = typename OperatorType::srcType;

   virtual ~FaceSmoother() = default;

   virtual void smooth( const OperatorType&, uint_t level, Face&, const FSFunctionType& x, const FSFunctionType& b ) = 0;

   virtual void
       smooth_backwards( const OperatorType&, uint_t level, Face&, const FSFunctionType& x, const FSFunctionType& b ) = 0;
};

template < typename OperatorType >
class CellSmoother
{
 public:
   using FSFunctionType = typename OperatorType::srcType;

   virtual ~CellSmoother() = default;

   virtual void preSmooth( const OperatorType&, uint_t, const FSFunctionType&, const FSFunctionType& ) {};

   virtual void postSmooth( const OperatorType&, uint_t, const FSFunctionType&, const FSFunctionType& ) {};

   virtual void smooth( const OperatorType&, uint_t level, Cell&, const FSFunctionType& x, const FSFunctionType& b ) = 0;

   virtual void
       smooth_backwards( const OperatorType&, uint_t level, Cell&, const FSFunctionType& x, const FSFunctionType& b ) = 0;
};

} // namespace hyteg
