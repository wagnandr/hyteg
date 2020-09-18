/*
 * Copyright (c) 2017-2019 Dominik Thoennes, Marcus Mohr.
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

#include "hyteg/geometry/GeometryMap.hpp"

namespace hyteg {

/// Base class for all forms
class Form
{
 public:
   virtual ~Form() {}

   virtual bool assemble2D() const = 0;

   virtual bool assemble3D() const = 0;

   virtual bool assembly2DDefined() const = 0;

   virtual bool assembly3DDefined() const = 0;

   /// Set the geometry/blending map for the form
   ///
   /// \note
   /// - This method is used e.g. by the ElementwiseOperators.
   /// - In the case of the FEniCS forms the map is ignored.
   void setGeometryMap( const std::shared_ptr< GeometryMap > & geometryMap ) {
     WALBERLA_ASSERT_NOT_NULLPTR( geometryMap );
     geometryMap_ = geometryMap;
   }

 protected:

   std::shared_ptr< GeometryMap > geometryMap_;
};

} // namespace hyteg
