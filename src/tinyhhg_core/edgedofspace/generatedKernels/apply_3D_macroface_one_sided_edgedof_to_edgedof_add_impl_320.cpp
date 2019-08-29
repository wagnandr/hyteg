
//////////////////////////////////////////////////////////////////////////////
// This file is generated! To fix issues, please fix them in the generator. //
//////////////////////////////////////////////////////////////////////////////

#include "apply_3D_macroface_one_sided_edgedof_to_edgedof_add_impl_320.hpp"

namespace hhg {
namespace edgedof {
namespace macroface {
namespace generated {

static void apply_3D_macroface_one_sided_edgedof_to_edgedof_add_impl_320_level_any(double * RESTRICT _data_edgeFaceDst_X, double * RESTRICT _data_edgeFaceDst_XY, double * RESTRICT _data_edgeFaceDst_Y, double const * RESTRICT const _data_edgeFaceSrc_X, double const * RESTRICT const _data_edgeFaceSrc_XY, double const * RESTRICT const _data_edgeFaceSrc_Y, double const * RESTRICT const _data_edgeFaceSrc_gl0_X, double const * RESTRICT const _data_edgeFaceSrc_gl0_XYZ, double const * RESTRICT const _data_edgeFaceSrc_gl0_XZ, double const * RESTRICT const _data_edgeFaceSrc_gl0_Y, double const * RESTRICT const _data_edgeFaceSrc_gl0_YZ, double const * RESTRICT const _data_edgeFaceSrc_gl0_Z, std::map< hhg::edgedof::EdgeDoFOrientation, std::map< hhg::edgedof::EdgeDoFOrientation, std::map< hhg::indexing::IndexIncrement, double > > > e2e_cell_stencil, int32_t level)
{
   const double xi_74 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::XYZ][{ 0, 0, -1 }];
   const double xi_75 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::XY][{ 0, 0, 0 }];
   const double xi_76 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::XZ][{ 0, 0, 0 }];
   const double xi_77 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::X][{ 0, 0, 0 }];
   const double xi_78 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::YZ][{ 0, 0, -1 }];
   const double xi_79 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::YZ][{ 0, 0, 0 }];
   const double xi_80 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::Y][{ 0, 0, 0 }];
   const double xi_81 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::Z][{ 0, 0, 0 }];
   const double xi_82 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Y][hhg::edgedof::EdgeDoFOrientation::Z][{ 0, 1, -1 }];
   const double xi_168 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::XYZ][{ 0, -1, 0 }];
   const double xi_169 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::XY][{ 0, -1, 1 }];
   const double xi_170 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::XY][{ 0, 0, 0 }];
   const double xi_171 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::XZ][{ 0, 0, 0 }];
   const double xi_172 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::X][{ 0, -1, 1 }];
   const double xi_173 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::X][{ 0, 0, 0 }];
   const double xi_174 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::YZ][{ 0, -1, 0 }];
   const double xi_175 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::YZ][{ 0, 0, 0 }];
   const double xi_176 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::YZ][{ 1, -1, 0 }];
   const double xi_177 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::Y][{ 0, -1, 1 }];
   const double xi_178 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::Y][{ 0, 0, 0 }];
   const double xi_179 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::Z][hhg::edgedof::EdgeDoFOrientation::Z][{ 0, 0, 0 }];
   const double xi_207 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::XYZ][{ 0, 0, 0 }];
   const double xi_208 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::XY][{ 0, 0, 0 }];
   const double xi_209 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::XY][{ 0, 0, 1 }];
   const double xi_210 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::XZ][{ 0, 0, 0 }];
   const double xi_211 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::X][{ 0, 0, 0 }];
   const double xi_212 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::X][{ 0, 0, 1 }];
   const double xi_213 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::YZ][{ 0, 0, 0 }];
   const double xi_214 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::Y][{ 0, 0, 0 }];
   const double xi_215 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::Y][{ 0, 0, 1 }];
   const double xi_216 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::Z][{ 0, 0, 0 }];
   const double xi_217 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::Z][{ 0, 1, 0 }];
   const double xi_218 = e2e_cell_stencil[hhg::edgedof::EdgeDoFOrientation::YZ][hhg::edgedof::EdgeDoFOrientation::Z][{ 1, 0, 0 }];
   {
      for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
      {
         // bottom left vertex
         for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
         {
            const double xi_94 = _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_85 = xi_74*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_86 = xi_75*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_87 = xi_76*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_88 = xi_77*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_89 = xi_78*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2))];
            const double xi_90 = xi_79*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_91 = xi_80*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_92 = xi_81*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_93 = xi_82*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) + 1];
            _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_85 + xi_86 + xi_87 + xi_88 + xi_89 + xi_90 + xi_91 + xi_92 + xi_93 + xi_94;
         }
         // bottom edge
         for (int ctr_1 = 1; ctr_1 < (1 << (level)) - 1; ctr_1 += 1)
         {
            const double xi_191 = _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_204 = _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_182 = xi_74*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_183 = xi_75*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_184 = xi_76*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_185 = xi_77*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_186 = xi_78*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2))];
            const double xi_187 = xi_79*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_188 = xi_80*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_189 = xi_81*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_190 = xi_82*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) + 1];
            const double xi_192 = xi_168*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_193 = xi_169*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_194 = xi_170*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_195 = xi_171*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_196 = xi_172*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_197 = xi_173*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_198 = xi_174*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2)) - 1];
            const double xi_199 = xi_175*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_200 = xi_176*_data_edgeFaceSrc_gl0_X[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_201 = xi_177*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_202 = xi_178*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_203 = xi_179*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_182 + xi_183 + xi_184 + xi_185 + xi_186 + xi_187 + xi_188 + xi_189 + xi_190 + xi_191;
            _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_192 + xi_193 + xi_194 + xi_195 + xi_196 + xi_197 + xi_198 + xi_199 + xi_200 + xi_201 + xi_202 + xi_203 + xi_204;
         }
         // bottom right vertex
         for (int ctr_1 = (1 << (level)) - 1; ctr_1 < (1 << (level)); ctr_1 += 1)
         {
            const double xi_123 = _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_120 = xi_168*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_121 = xi_169*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_122 = xi_170*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_111 = xi_171*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_112 = xi_172*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_113 = xi_173*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_114 = xi_174*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2)) - 1];
            const double xi_115 = xi_175*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_116 = xi_176*_data_edgeFaceSrc_gl0_X[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_117 = xi_177*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_118 = xi_178*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_119 = xi_179*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_111 + xi_112 + xi_113 + xi_114 + xi_115 + xi_116 + xi_117 + xi_118 + xi_119 + xi_120 + xi_121 + xi_122 + xi_123;
         }
      }
      for (int ctr_2 = 1; ctr_2 < (1 << (level)) - 1; ctr_2 += 1)
      {
         // left edge
         for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
         {
            const double xi_242 = _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_252 = _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_230 = xi_207*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_231 = xi_208*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_232 = xi_209*_data_edgeFaceSrc_gl0_XZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_233 = xi_210*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_234 = xi_211*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_235 = xi_212*_data_edgeFaceSrc_gl0_YZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_236 = xi_213*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_237 = xi_214*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_238 = xi_215*_data_edgeFaceSrc_XY[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_239 = xi_216*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_240 = xi_217*_data_edgeFaceSrc_Y[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2)) + 1];
            const double xi_241 = xi_218*_data_edgeFaceSrc_gl0_Y[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_243 = xi_74*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_244 = xi_75*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_245 = xi_76*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_246 = xi_77*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_247 = xi_78*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2))];
            const double xi_248 = xi_79*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_249 = xi_80*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_250 = xi_81*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_251 = xi_82*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) + 1];
            _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_230 + xi_231 + xi_232 + xi_233 + xi_234 + xi_235 + xi_236 + xi_237 + xi_238 + xi_239 + xi_240 + xi_241 + xi_242;
            _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_243 + xi_244 + xi_245 + xi_246 + xi_247 + xi_248 + xi_249 + xi_250 + xi_251 + xi_252;
         }
         // inner triangle
         for (int ctr_1 = 1; ctr_1 < -ctr_2 + (1 << (level)) - 1; ctr_1 += 1)
         {
            const double xi_48 = _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_58 = _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_71 = _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_36 = xi_207*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_40 = xi_208*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_41 = xi_209*_data_edgeFaceSrc_gl0_XZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_42 = xi_210*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_43 = xi_211*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_44 = xi_212*_data_edgeFaceSrc_gl0_YZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_45 = xi_213*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_46 = xi_214*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_47 = xi_215*_data_edgeFaceSrc_XY[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_37 = xi_216*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_38 = xi_217*_data_edgeFaceSrc_Y[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2)) + 1];
            const double xi_39 = xi_218*_data_edgeFaceSrc_gl0_Y[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_49 = xi_74*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_50 = xi_75*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_51 = xi_76*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_52 = xi_77*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_53 = xi_78*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2))];
            const double xi_54 = xi_79*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_55 = xi_80*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_56 = xi_81*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_57 = xi_82*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) + 1];
            const double xi_59 = xi_168*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_60 = xi_169*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_61 = xi_170*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_62 = xi_171*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_63 = xi_172*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_64 = xi_173*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_65 = xi_174*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2)) - 1];
            const double xi_66 = xi_175*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_67 = xi_176*_data_edgeFaceSrc_gl0_X[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_68 = xi_177*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_69 = xi_178*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_70 = xi_179*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_36 + xi_37 + xi_38 + xi_39 + xi_40 + xi_41 + xi_42 + xi_43 + xi_44 + xi_45 + xi_46 + xi_47 + xi_48;
            _data_edgeFaceDst_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_49 + xi_50 + xi_51 + xi_52 + xi_53 + xi_54 + xi_55 + xi_56 + xi_57 + xi_58;
            _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_59 + xi_60 + xi_61 + xi_62 + xi_63 + xi_64 + xi_65 + xi_66 + xi_67 + xi_68 + xi_69 + xi_70 + xi_71;
         }
         // diagonal edge
         for (int ctr_1 = -ctr_2 + (1 << (level)) - 1; ctr_1 < -ctr_2 + (1 << (level)); ctr_1 += 1)
         {
            const double xi_293 = _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_306 = _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_281 = xi_207*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_282 = xi_208*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_283 = xi_209*_data_edgeFaceSrc_gl0_XZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_284 = xi_210*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_285 = xi_211*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_286 = xi_212*_data_edgeFaceSrc_gl0_YZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_287 = xi_213*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_288 = xi_214*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_289 = xi_215*_data_edgeFaceSrc_XY[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_290 = xi_216*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_291 = xi_217*_data_edgeFaceSrc_Y[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2)) + 1];
            const double xi_292 = xi_218*_data_edgeFaceSrc_gl0_Y[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_294 = xi_168*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_295 = xi_169*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_296 = xi_170*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_297 = xi_171*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_298 = xi_172*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_299 = xi_173*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_300 = xi_174*_data_edgeFaceSrc_X[ctr_1 + (ctr_2 + 1)*((1 << (level)) + 1) - (((ctr_2 + 1)*(ctr_2 + 2)) / (2)) - 1];
            const double xi_301 = xi_175*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_302 = xi_176*_data_edgeFaceSrc_gl0_X[ctr_1 + ctr_2*(1 << (level)) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_303 = xi_177*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2)) - 1];
            const double xi_304 = xi_178*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_305 = xi_179*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_281 + xi_282 + xi_283 + xi_284 + xi_285 + xi_286 + xi_287 + xi_288 + xi_289 + xi_290 + xi_291 + xi_292 + xi_293;
            _data_edgeFaceDst_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_294 + xi_295 + xi_296 + xi_297 + xi_298 + xi_299 + xi_300 + xi_301 + xi_302 + xi_303 + xi_304 + xi_305 + xi_306;
         }
      }
      for (int ctr_2 = (1 << (level)) - 1; ctr_2 < (1 << (level)); ctr_2 += 1)
      {
         // top vertex
         for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
         {
            const double xi_152 = _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_140 = xi_207*_data_edgeFaceSrc_gl0_XYZ[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_141 = xi_208*_data_edgeFaceSrc_gl0_XZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_142 = xi_209*_data_edgeFaceSrc_gl0_XZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_143 = xi_210*_data_edgeFaceSrc_gl0_Z[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_144 = xi_211*_data_edgeFaceSrc_gl0_YZ[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_145 = xi_212*_data_edgeFaceSrc_gl0_YZ[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_146 = xi_213*_data_edgeFaceSrc_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_147 = xi_214*_data_edgeFaceSrc_XY[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_148 = xi_215*_data_edgeFaceSrc_XY[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2))];
            const double xi_149 = xi_216*_data_edgeFaceSrc_Y[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))];
            const double xi_150 = xi_217*_data_edgeFaceSrc_Y[ctr_1 + (ctr_2 - 1)*((1 << (level)) + 1) - ((ctr_2*(ctr_2 - 1)) / (2)) + 1];
            const double xi_151 = xi_218*_data_edgeFaceSrc_gl0_Y[ctr_1 + (ctr_2 - 1)*(1 << (level)) - ((ctr_2*(ctr_2 - 1)) / (2))];
            _data_edgeFaceDst_X[ctr_1 + ctr_2*((1 << (level)) + 1) - ((ctr_2*(ctr_2 + 1)) / (2))] = xi_140 + xi_141 + xi_142 + xi_143 + xi_144 + xi_145 + xi_146 + xi_147 + xi_148 + xi_149 + xi_150 + xi_151 + xi_152;
         }
      }
   }
}


void apply_3D_macroface_one_sided_edgedof_to_edgedof_add_impl_320(double * RESTRICT _data_edgeFaceDst_X, double * RESTRICT _data_edgeFaceDst_XY, double * RESTRICT _data_edgeFaceDst_Y, double const * RESTRICT const _data_edgeFaceSrc_X, double const * RESTRICT const _data_edgeFaceSrc_XY, double const * RESTRICT const _data_edgeFaceSrc_Y, double const * RESTRICT const _data_edgeFaceSrc_gl0_X, double const * RESTRICT const _data_edgeFaceSrc_gl0_XYZ, double const * RESTRICT const _data_edgeFaceSrc_gl0_XZ, double const * RESTRICT const _data_edgeFaceSrc_gl0_Y, double const * RESTRICT const _data_edgeFaceSrc_gl0_YZ, double const * RESTRICT const _data_edgeFaceSrc_gl0_Z, std::map< hhg::edgedof::EdgeDoFOrientation, std::map< hhg::edgedof::EdgeDoFOrientation, std::map< hhg::indexing::IndexIncrement, double > > > e2e_cell_stencil, int32_t level)
{
    switch( level )
    {

    default:
        apply_3D_macroface_one_sided_edgedof_to_edgedof_add_impl_320_level_any(_data_edgeFaceDst_X, _data_edgeFaceDst_XY, _data_edgeFaceDst_Y, _data_edgeFaceSrc_X, _data_edgeFaceSrc_XY, _data_edgeFaceSrc_Y, _data_edgeFaceSrc_gl0_X, _data_edgeFaceSrc_gl0_XYZ, _data_edgeFaceSrc_gl0_XZ, _data_edgeFaceSrc_gl0_Y, _data_edgeFaceSrc_gl0_YZ, _data_edgeFaceSrc_gl0_Z, e2e_cell_stencil, level);
        break;
    }
}
    

} // namespace generated
} // namespace macroface
} // namespace edgedof
} // namespace hhg