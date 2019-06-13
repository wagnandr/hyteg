
//////////////////////////////////////////////////////////////////////////////
// This file is generated! To fix issues, please fix them in the generator. //
//////////////////////////////////////////////////////////////////////////////

#pragma once
#include "all.hpp"
#include "core/DataTypes.h"
#include "core/Macros.h"
#include "tinyhhg_core/edgedofspace/EdgeDoFIndexing.hpp"
#include <map>
#define RESTRICT WALBERLA_RESTRICT

namespace hhg {
namespace vertexdof {
namespace macrocell {
namespace generated {

void apply_3D_macrocell_vertexdof_to_vertexdof_replace_colored_impl_group_5(double * RESTRICT _data_p1CellDst_group_5, double const * RESTRICT const _data_p1CellSrc_group_0_const, double const * RESTRICT const _data_p1CellSrc_group_1_const, double const * RESTRICT const _data_p1CellSrc_group_2_const, double const * RESTRICT const _data_p1CellSrc_group_3_const, double const * RESTRICT const _data_p1CellSrc_group_4_const, double const * RESTRICT const _data_p1CellSrc_group_5_const, double const * RESTRICT const _data_p1CellSrc_group_6_const, double const * RESTRICT const _data_p1CellSrc_group_7_const, int32_t level, std::map< hhg::indexing::IndexIncrement, double > p1CellStencil);

} // namespace generated
} // namespace macrocell
} // namespace vertexdof
} // namespace hhg