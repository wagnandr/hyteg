
//////////////////////////////////////////////////////////////////////////////
// This file is generated! To fix issues, please fix them in the generator. //
//////////////////////////////////////////////////////////////////////////////

#include "GeneratedKernelsP2MacroFace2D.hpp"

namespace hhg {
namespace P2 {
namespace macroface {
namespace generated {

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_2(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 4; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 4; ctr_1 < 5; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 4; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 4; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 9] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 4; ctr_1 < -ctr_2 + 5; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 7] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 10] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 4; ctr_2 < 5; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 9] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + ((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 18] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 18*ctr_2 + 2*((72) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 17] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 20*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 6*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_3(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 8; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 8; ctr_1 < 9; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 8; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 8; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 17] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 8; ctr_1 < -ctr_2 + 9; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 15] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 18] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 8; ctr_2 < 9; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 17] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + ((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 34] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 34*ctr_2 + 2*((272) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 33] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 36*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 10*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_4(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 16; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 16; ctr_1 < 17; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 16; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 16; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 33] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 16; ctr_1 < -ctr_2 + 17; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 31] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 34] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 16; ctr_2 < 17; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 33] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + ((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 66] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 66*ctr_2 + 2*((1056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 68*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 18*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_5(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 32; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 32; ctr_1 < 33; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 32; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 32; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 64] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 65] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 64] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 32; ctr_1 < -ctr_2 + 33; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 63] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 66] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 32; ctr_2 < 33; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 65] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + ((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 130] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 130*ctr_2 + 2*((4160) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 129] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 132*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 34*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_6(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 64; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 64; ctr_1 < 65; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 64; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 64; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 128] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 129] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 128] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 64; ctr_1 < -ctr_2 + 65; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 127] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 130] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 64; ctr_2 < 65; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 129] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + ((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 258] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 258*ctr_2 + 2*((16512) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 257] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 260*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 66*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_7(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 128; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 128; ctr_1 < 129; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 128; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 128; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 256] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 257] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 256] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 128; ctr_1 < -ctr_2 + 129; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 255] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 258] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 128; ctr_2 < 129; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 257] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + ((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 514] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 514*ctr_2 + 2*((65792) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 513] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 516*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 130*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_8(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 256; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 256; ctr_1 < 257; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 256; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 256; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 512] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 513] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 512] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 256; ctr_1 < -ctr_2 + 257; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 511] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 514] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 256; ctr_2 < 257; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 513] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + ((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1026] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 1026*ctr_2 + 2*((262656) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 1025] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 1028*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 258*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_9(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 512; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 512; ctr_1 < 513; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 512; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 512; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1024] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1025] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1024] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 512; ctr_1 < -ctr_2 + 513; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 1023] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1026] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 512; ctr_2 < 513; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1025] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + ((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2050] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 2050*ctr_2 + 2*((1049600) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 2049] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 2052*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 514*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_10(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 1024; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1024; ctr_1 < 1025; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 1024; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 1024; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2048] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2049] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2048] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 1024; ctr_1 < -ctr_2 + 1025; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2047] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2050] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1024; ctr_2 < 1025; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 2049] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + ((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4098] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 4098*ctr_2 + 2*((4196352) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 4097] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 4100*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 1026*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_11(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 2048; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 2048; ctr_1 < 2049; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 2048; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 2048; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4096] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4097] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4096] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 2048; ctr_1 < -ctr_2 + 2049; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 4095] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4098] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 2048; ctr_2 < 2049; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 4097] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + ((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8194] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 8194*ctr_2 + 2*((16781312) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 8193] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 8196*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 2050*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_12(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 4096; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 4096; ctr_1 < 4097; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 4096; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 4096; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8192] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8193] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8192] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 4096; ctr_1 < -ctr_2 + 4097; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 8191] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8194] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 4096; ctr_2 < 4097; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 8193] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + ((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16386] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 16386*ctr_2 + 2*((67117056) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 16385] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 16388*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 4098*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_13(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 8192; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 8192; ctr_1 < 8193; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 8192; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 8192; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16384] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16385] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16384] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 8192; ctr_1 < -ctr_2 + 8193; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 16383] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16386] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 8192; ctr_2 < 8193; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 16385] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + ((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32770] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 32770*ctr_2 + 2*((268451840) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 32769] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 32772*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 8194*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_14(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < 16384; ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 16384; ctr_1 < 16385; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < 16384; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + 16384; ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32768] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32769] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32768] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + 16384; ctr_1 < -ctr_2 + 16385; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 32767] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32770] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 16384; ctr_2 < 16385; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 32769] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + ((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65538] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + 65538*ctr_2 + 2*((1073774592) / (2)) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) - 65537] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 65540*ctr_2 - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + 16386*ctr_2 - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}

static void prolongate_2D_macroface_P2_push_from_vertexdofs_level_any(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, int64_t coarse_level, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
   const double xi_53 = 1 / (num_neighbor_faces_edge0);
   const double xi_55 = 1 / (num_neighbor_faces_edge2);
   const double xi_56 = 1 / (num_neighbor_faces_vertex0);
   const double xi_116 = 1 / (num_neighbor_faces_edge0);
   const double xi_73 = 1 / (num_neighbor_faces_edge0);
   const double xi_75 = 1 / (num_neighbor_faces_edge1);
   const double xi_76 = 1 / (num_neighbor_faces_vertex1);
   const double xi_188 = 1 / (num_neighbor_faces_edge2);
   const double xi_152 = 1 / (num_neighbor_faces_edge1);
   const double xi_93 = 1 / (num_neighbor_faces_edge1);
   const double xi_95 = 1 / (num_neighbor_faces_edge2);
   const double xi_96 = 1 / (num_neighbor_faces_vertex2);
   for (int ctr_2 = 0; ctr_2 < 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_59 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_61 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_63 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_65 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_68 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_67 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))];
         const double xi_69 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_58 = xi_53*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_60 = xi_53*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_62 = xi_55*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_64 = xi_55*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_58 + xi_59;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_60 + xi_61;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_62 + xi_63;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_64 + xi_65;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))] = xi_67 + xi_68;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_68 + xi_69;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_56*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < (1 << (coarse_level)); ctr_1 += 1)
      {
         const double xi_119 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_121 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_123 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_125 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_144 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_127 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2];
         const double xi_129 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_131 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_133 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_135 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_137 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))];
         const double xi_139 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_142 = 0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_141 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_143 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_145 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_124 = xi_116*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_122 = xi_116*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_119 + xi_124;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_121 + xi_122;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_122 + xi_123;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_124 + xi_125;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2] = xi_127 + xi_144;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_129 + xi_144;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_131 + xi_144;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_133 + xi_144;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_135 + xi_144;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))] = xi_137 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_139 + xi_144;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_141 + xi_142;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_142 + xi_143;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_144 + xi_145;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_116*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = (1 << (coarse_level)); ctr_1 < (1 << (coarse_level)) + 1; ctr_1 += 1)
      {
         const double xi_79 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_81 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_83 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_85 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_88 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_87 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2];
         const double xi_89 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_78 = xi_73*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_80 = xi_73*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_82 = xi_75*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_84 = xi_75*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_78 + xi_79;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_80 + xi_81;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_82 + xi_83;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_84 + xi_85;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2] = xi_87 + xi_88;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_88 + xi_89;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_76*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = 1; ctr_2 < (1 << (coarse_level)); ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_191 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_193 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_195 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_197 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_216 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_199 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))];
         const double xi_206 = 0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_201 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_203 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_205 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_207 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_209 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 1];
         const double xi_211 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_213 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_215 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_217 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_196 = xi_188*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_194 = xi_188*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_191 + xi_196;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_193 + xi_194;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_194 + xi_195;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_196 + xi_197;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))] = xi_199 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_201 + xi_206;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_203 + xi_216;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_205 + xi_216;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_206 + xi_207;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 1] = xi_209 + xi_216;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_211 + xi_216;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_213 + xi_216;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_215 + xi_216;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_216 + xi_217;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_188*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = 1; ctr_1 < -ctr_2 + (1 << (coarse_level)); ctr_1 += 1)
      {
         const double xi_48 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_3 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2];
         const double xi_5 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_7 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_9 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_11 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_13 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))];
         const double xi_15 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_17 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_36 = 0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_19 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_21 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_23 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_25 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))];
         const double xi_27 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_29 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1];
         const double xi_31 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_33 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1];
         const double xi_35 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_37 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_39 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 1];
         const double xi_41 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_43 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_45 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_47 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_49 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2] = xi_3 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_48 + xi_5;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_48 + xi_7;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_48 + xi_9;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_11 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2))] = xi_13 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_15 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_17 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_19 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_21 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_23 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = xi_25 + xi_36;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_27 + xi_48;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + 1] = xi_29 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_31 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1] = xi_33 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_35 + xi_36;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_36 + xi_37;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 1] = xi_39 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_41 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_43 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_45 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_47 + xi_48;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_48 + xi_49;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = _data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
      for (int ctr_1 = -ctr_2 + (1 << (coarse_level)); ctr_1 < -ctr_2 + (1 << (coarse_level)) + 1; ctr_1 += 1)
      {
         const double xi_155 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_157 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_159 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_161 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_180 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_163 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2];
         const double xi_165 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2];
         const double xi_167 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2];
         const double xi_174 = 0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_169 = _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1];
         const double xi_171 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1];
         const double xi_173 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1];
         const double xi_175 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_177 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_179 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_181 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_160 = xi_152*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_158 = xi_152*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_155 + xi_160;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_157 + xi_158;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_158 + xi_159;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_160 + xi_161;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 + 1)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 + 1)*(2*ctr_2 + 2)) / (2)) - 2] = xi_163 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 2] = xi_165 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 2] = xi_167 + xi_180;
         _data_edgeFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 + 1)) / (2)) - 1] = xi_169 + xi_174;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) - 1] = xi_171 + xi_180;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) - 1] = xi_173 + xi_180;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_174 + xi_175;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_177 + xi_180;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_179 + xi_180;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_180 + xi_181;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_152*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   for (int ctr_2 = (1 << (coarse_level)); ctr_2 < (1 << (coarse_level)) + 1; ctr_2 += 1)
   {
      for (int ctr_1 = 0; ctr_1 < 1; ctr_1 += 1)
      {
         const double xi_99 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_101 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_103 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_105 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_108 = -0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_107 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))];
         const double xi_109 = _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1];
         const double xi_98 = xi_93*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_100 = xi_93*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_102 = xi_95*0.375*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         const double xi_104 = xi_95*-0.125*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_98 + xi_99;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_100 + xi_101;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 1)*((1 << (coarse_level + 1)) + 1) - ((2*ctr_2*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_102 + xi_103;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_104 + xi_105;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + ((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2))] = xi_107 + xi_108;
         _data_edgeFineDst[2*ctr_1 + (2*ctr_2 - 2)*((1 << (coarse_level + 1)) + 1) - (((2*ctr_2 - 2)*(2*ctr_2 - 1)) / (2)) + 2*((((1 << (coarse_level + 1)) + 1)*(1 << (coarse_level + 1))) / (2)) + 1] = xi_108 + xi_109;
         _data_vertexFineDst[2*ctr_1 + 2*ctr_2*((1 << (coarse_level + 1)) + 2) - ((2*ctr_2*(2*ctr_2 + 1)) / (2))] = 1.0*xi_96*_data_vertexCoarseSrc[ctr_1 + ctr_2*((1 << (coarse_level)) + 2) - ((ctr_2*(ctr_2 + 1)) / (2))];
      }
   }
   {
      
   }
}


void prolongate_2D_macroface_P2_push_from_vertexdofs(double * _data_edgeFineDst, double * _data_vertexCoarseSrc, double * _data_vertexFineDst, int64_t coarse_level, double num_neighbor_faces_edge0, double num_neighbor_faces_edge1, double num_neighbor_faces_edge2, double num_neighbor_faces_vertex0, double num_neighbor_faces_vertex1, double num_neighbor_faces_vertex2)
{
    switch( coarse_level )
    {
    case 2:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_2(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 3:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_3(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 4:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_4(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 5:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_5(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 6:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_6(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 7:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_7(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 8:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_8(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 9:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_9(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 10:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_10(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 11:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_11(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 12:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_12(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 13:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_13(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    case 14:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_14(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    default:
        prolongate_2D_macroface_P2_push_from_vertexdofs_level_any(_data_edgeFineDst, _data_vertexCoarseSrc, _data_vertexFineDst, coarse_level, num_neighbor_faces_edge0, num_neighbor_faces_edge1, num_neighbor_faces_edge2, num_neighbor_faces_vertex0, num_neighbor_faces_vertex1, num_neighbor_faces_vertex2);
        break;
    }
}
    

} // namespace generated
} // namespace macroface
} // namespace P2
} // namespace hhg