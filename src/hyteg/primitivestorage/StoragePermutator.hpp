/*
 * Copyright (c) 2020 Andreas Wagner.
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

#include <functional>
#include <vector>
#include <array>

#include "core/DataTypes.h"

#include "waLBerlaDefinitions.h"

namespace hyteg {

using walberla::real_t;
using walberla::uint_t;

class SetupPrimitiveStorage;
class Face;
class Cell;

template < typename T, uint_t N, uint_t M >
class Matrix;

class StoragePermutator
{
 public:
   void permutate_randomly( SetupPrimitiveStorage& setup, uint_t seed );
   void permutate_ilu( SetupPrimitiveStorage& setup );
   void permutate( SetupPrimitiveStorage& setup, const std::function< std::array< uint_t, 4 > (const Cell&)> & permutator );

 private:
   void permutate( SetupPrimitiveStorage& setup, Cell& cell, std::array< uint_t, 4 > permutation );

   std::array< real_t, 3> get_triangle_angles(const Cell& cell, const std::vector< uint_t >& vertexIds) const;
   real_t get_triangle_area(const Cell& cell, const std::vector< uint_t >& vertexIds) const;
   real_t getHeight(const Cell& cell, uint_t id) const;

   uint_t getMaxAreaTriangle(const Cell& cell) const;
   uint_t getMaxAngleTriangle(const Cell& cell) const;
   uint_t getMinAngleTriangle(const Cell& cell) const;
   uint_t getMinHeightBaseTriangle(const Cell& cell) const;
   uint_t getMaxHeightBaseTriangle(const Cell& cell) const;
};

} // namespace hyteg
