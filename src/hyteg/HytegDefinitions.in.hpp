/*
 * Copyright (c) 2017-2019 Dominik Thoennes, Marcus Mohr, Nils Kohl.
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

#cmakedefine HYTEG_BUILD_WITH_PETSC
#cmakedefine HYTEG_BUILD_WITH_EIGEN
#cmakedefine HYTEG_USE_GENERATED_KERNELS
#cmakedefine HYTEG_P1_COLORING

#ifdef HYTEG_USE_GENERATED_KERNELS
namespace hyteg {
namespace globalDefines {
constexpr bool useGeneratedKernels = true;
} // namespace globalDefines
} // namespace hyteg
#else
namespace hyteg {
namespace globalDefines {
constexpr bool useGeneratedKernels = false;
} // namesapce globalDefines
} // namespace hyteg
#endif

#ifdef HYTEG_P1_COLORING
namespace hyteg {
namespace globalDefines {
constexpr bool useP1Coloring = true;
} // namespace globalDefines
} // namespace hyteg
#else
namespace hyteg {
namespace globalDefines {
constexpr bool useP1Coloring = false;
} // namesapce globalDefines
} // namespace hyteg
#endif