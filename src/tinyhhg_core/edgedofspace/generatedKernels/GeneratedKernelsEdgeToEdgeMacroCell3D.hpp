
//////////////////////////////////////////////////////////////////////////////
// This file is generated! To fix issues, please fix them in the generator. //
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "core/DataTypes.h"

#include "core/Macros.h"

#include "tinyhhg_core/edgedofspace/EdgeDoFIndexing.hpp"

#include <map>

#define RESTRICT WALBERLA_RESTRICT

namespace hhg {
namespace edgedof {
namespace macrocell {
namespace generated {

void assign_3D_macrocell_edgedof_1_rhs_function(double * RESTRICT _data_edgeCellDst_X, double * RESTRICT _data_edgeCellDst_XY, double * RESTRICT _data_edgeCellDst_XYZ, double * RESTRICT _data_edgeCellDst_XZ, double * RESTRICT _data_edgeCellDst_Y, double * RESTRICT _data_edgeCellDst_YZ, double * RESTRICT _data_edgeCellDst_Z, double * RESTRICT _data_edgeCellSrc_X, double * RESTRICT _data_edgeCellSrc_XY, double * RESTRICT _data_edgeCellSrc_XYZ, double * RESTRICT _data_edgeCellSrc_XZ, double * RESTRICT _data_edgeCellSrc_Y, double * RESTRICT _data_edgeCellSrc_YZ, double * RESTRICT _data_edgeCellSrc_Z, double c, int64_t level);

void apply_3D_macrocell_edgedof_to_edgedof_replace(double * RESTRICT _data_edgeCellDst_X, double * RESTRICT _data_edgeCellDst_XY, double * RESTRICT _data_edgeCellDst_XYZ, double * RESTRICT _data_edgeCellDst_XZ, double * RESTRICT _data_edgeCellDst_Y, double * RESTRICT _data_edgeCellDst_YZ, double * RESTRICT _data_edgeCellDst_Z, double const * RESTRICT const _data_edgeCellSrc_X, double const * RESTRICT const _data_edgeCellSrc_XY, double const * RESTRICT const _data_edgeCellSrc_XYZ, double const * RESTRICT const _data_edgeCellSrc_XZ, double const * RESTRICT const _data_edgeCellSrc_Y, double const * RESTRICT const _data_edgeCellSrc_YZ, double const * RESTRICT const _data_edgeCellSrc_Z, std::map< hhg::edgedof::EdgeDoFOrientation, std::map< hhg::edgedof::EdgeDoFOrientation, std::map< hhg::indexing::IndexIncrement, double > > > e2eStencilMap, int64_t level);

} // namespace generated
} // namespace macrocell
} // namespace edgedof
} // namespace hhg