
//////////////////////////////////////////////////////////////////////////////
// This file is generated! To fix issues, please fix them in the generator. //
//////////////////////////////////////////////////////////////////////////////

#include "gaussseidel_2D_macroface_vertexdof_to_vertexdof_backwards.hpp"

namespace hhg {
namespace vertexdof {
namespace macroface {
namespace generated {

static void gaussseidel_2D_macroface_vertexdof_to_vertexdof_backwards_level_any(double * RESTRICT _data_p1FaceDst, double * RESTRICT _data_p1FaceRhs, double const * RESTRICT const _data_p1FaceStencil, int32_t level)
{
   const double xi_0 = _data_p1FaceStencil[3];
   const double xi_9 = 1 / (xi_0);
   const double xi_1 = _data_p1FaceStencil[2];
   const double xi_2 = _data_p1FaceStencil[5];
   const double xi_3 = _data_p1FaceStencil[0];
   const double xi_4 = _data_p1FaceStencil[6];
   const double xi_5 = _data_p1FaceStencil[1];
   const double xi_6 = _data_p1FaceStencil[4];
   for (int ctr_2 = (1 << (level)) - 1; ctr_2 >= 1; ctr_2 += -1)
   {
      // inner triangle
      for (int ctr_1 = -ctr_2 + (1 << (level)) - 1; ctr_1 >= 1; ctr_1 += -1)
      {
         const double xi_17 = _data_p1FaceRhs[ctr_1 + ctr_2*((1 << (level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_11 = -xi_1*_data_p1FaceDst[ctr_1 + ctr_2*((1 << (level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
         const double xi_12 = -xi_2*_data_p1FaceDst[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 2) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2)) - 1];
         const double xi_13 = -xi_3*_data_p1FaceDst[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 2) - ((ctr_2*(ctr_2 - 1)) / (2))];
         const double xi_14 = -xi_4*_data_p1FaceDst[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 2) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2))];
         const double xi_15 = -xi_5*_data_p1FaceDst[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 2) - ((ctr_2*(ctr_2 - 1)) / (2)) + 1];
         const double xi_16 = -xi_6*_data_p1FaceDst[ctr_1 + ctr_2*((1 << (level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2)) + 1];
         _data_p1FaceDst[ctr_1 + ctr_2*((1 << (level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_9*(xi_11 + xi_12 + xi_13 + xi_14 + xi_15 + xi_16 + xi_17);
      }
   }
}


void gaussseidel_2D_macroface_vertexdof_to_vertexdof_backwards(double * RESTRICT _data_p1FaceDst, double * RESTRICT _data_p1FaceRhs, double const * RESTRICT const _data_p1FaceStencil, int32_t level)
{
    switch( level )
    {

    default:
        gaussseidel_2D_macroface_vertexdof_to_vertexdof_backwards_level_any(_data_p1FaceDst, _data_p1FaceRhs, _data_p1FaceStencil, level);
        break;
    }
}
    

} // namespace generated
} // namespace macroface
} // namespace vertexdof
} // namespace hhg