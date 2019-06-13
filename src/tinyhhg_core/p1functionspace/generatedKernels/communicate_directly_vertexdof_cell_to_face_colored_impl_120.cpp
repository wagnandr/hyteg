
//////////////////////////////////////////////////////////////////////////////
// This file is generated! To fix issues, please fix them in the generator. //
//////////////////////////////////////////////////////////////////////////////

#include "communicate_directly_vertexdof_cell_to_face_colored_impl_120.hpp"

namespace hhg {
namespace vertexdof {
namespace comm {
namespace generated {

static void communicate_directly_vertexdof_cell_to_face_colored_impl_120_level_any(double const * RESTRICT const _data_p1_cell_src_group_0_const, double * RESTRICT _data_p1_face_dst_gl0, int32_t level)
{
   {
      for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
      {
         // bottom left vertex
         for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
         // bottom edge
         for (int ctr_1 = 1; ctr_1 < (1 << (level)) - 1; ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
         // bottom right vertex
         for (int ctr_1 = (1 << (level)) - 1; ctr_1 < (1 << (level)); ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
      }
      for (int ctr_2 = 1; ctr_2 < (1 << (level)) - 1; ctr_2 += 1)
      {
         // left edge
         for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
         // inner triangle
         for (int ctr_1 = 1; ctr_1 < -ctr_2 + (1 << (level)) - 1; ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
         // diagonal edge
         for (int ctr_1 = -ctr_2 + (1 << (level)) - 1; ctr_1 < -ctr_2 + (1 << (level)); ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
      }
      for (int ctr_2 = (1 << (level)) - 1; ctr_2 < (1 << (level)); ctr_2 += 1)
      {
         // top vertex
         for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
         {
            _data_p1_face_dst_gl0[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = _data_p1_cell_src_group_0_const[(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*((ctr_1 - ((ctr_1) % (2))) / (2)) + ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (0): ((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) - 1)*((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*(1 << (level - 1))) / (6)) + ((((1 << (level - 1)) + 1)*((1 << (level - 1)) + 2)*((1 << (level - 1)) + 3)) / (6)))) - (((((ctr_1 - ((ctr_1) % (2))) / (2)) + 1)*((ctr_1 - ((ctr_1) % (2))) / (2))) / (2)) + (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) + (1 << (level - 1)) + 1)) / (6)) - (((((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)))*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) - 1)*(((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (0))) ? (2): ((((4*((1) % (2)) + 2*((ctr_1) % (2)) + ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2))) == (7))) ? (0): (1))) - ((1 - ((1) % (2))) / (2)) + (1 << (level - 1)) + 1)) / (6)) + ((-ctr_1 - ctr_2 - ((-ctr_1 - ctr_2 + (1 << (level)) - 1) % (2)) + (1 << (level)) - 1) / (2))];
         }
      }
   }
}


void communicate_directly_vertexdof_cell_to_face_colored_impl_120(double const * RESTRICT const _data_p1_cell_src_group_0_const, double * RESTRICT _data_p1_face_dst_gl0, int32_t level)
{
    switch( level )
    {

    default:
        communicate_directly_vertexdof_cell_to_face_colored_impl_120_level_any(_data_p1_cell_src_group_0_const, _data_p1_face_dst_gl0, level);
        break;
    }
}
    

} // namespace generated
} // namespace comm
} // namespace vertexdof
} // namespace hhg