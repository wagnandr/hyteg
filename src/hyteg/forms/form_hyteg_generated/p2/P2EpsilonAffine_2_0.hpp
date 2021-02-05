/*
 * Copyright (c) 2017-2021 Nils Kohl.
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

/*
 * The entire file was generated with the HyTeG form generator.
 * 
 * Software:
 *
 * - quadpy version: 0.16.5
 *
 * Avoid modifying this file. If buggy, consider fixing the generator itself.
 */

#pragma once

#include "hyteg/geometry/GeometryMap.hpp"
#include "hyteg/forms/form_hyteg_base/P1FormHyTeG.hpp"
#include "hyteg/forms/form_hyteg_base/P2FormHyTeG.hpp"

namespace hyteg {

/// Implementation of the integration of a weak form over an element.
///
/// - name:        P2EpsilonAffine_2_0
/// - description: 
/// - trial space: Lagrange, degree: 2
/// - test space:  Lagrange, degree: 2
///
class P2EpsilonAffine_2_0 : public P2FormHyTeG
{



 public:

   /// \brief Integrates the weak form over the passed element (vertices in computational space).
   ///
   /// - element geometry:                       triangle, dim: 2, vertices: 3
   /// - element matrix dimensions (rows, cols): (6, 6)
   /// - quadrature rule:                        Dunavant 2 | points: 3, degree: 2, test tolerance: 2.22e-16
   /// - floating point operations:
   ///                                             adds    muls    divs    abs    assignments    function_calls
   ///                                           ------  ------  ------  -----  -------------  ----------------
   ///                                                0       0       0      0             87                 0
   ///
   void integrateAll( const std::array< Point3D, 3 >& coords, Matrix< real_t, 6, 6 >& elMat ) const override
   {
      real_t p_affine_0_0 = coords[0][0];
      real_t p_affine_0_1 = coords[0][1];
      real_t p_affine_1_0 = coords[1][0];
      real_t p_affine_1_1 = coords[1][1];
      real_t p_affine_2_0 = coords[2][0];
      real_t p_affine_2_1 = coords[2][1];
      real_t q_p_0_0 = 0.16666666666666666;
      real_t q_p_0_1 = 0.66666666666666663;
      real_t q_p_1_0 = 0.66666666666666663;
      real_t q_p_1_1 = 0.16666666666666666;
      real_t q_p_2_0 = 0.16666666666666666;
      real_t q_p_2_1 = 0.16666666666666666;
      real_t w_p_0 = 0.16666666666666666;
      real_t w_p_1 = 0.16666666666666666;
      real_t w_p_2 = 0.16666666666666666;
      real_t a_0_0 = 0;
      real_t a_0_1 = 0;
      real_t a_0_2 = 0;
      real_t a_0_3 = 0;
      real_t a_0_4 = 0;
      real_t a_0_5 = 0;
      real_t a_1_0 = 0;
      real_t a_1_1 = 0;
      real_t a_1_2 = 0;
      real_t a_1_3 = 0;
      real_t a_1_4 = 0;
      real_t a_1_5 = 0;
      real_t a_2_0 = 0;
      real_t a_2_1 = 0;
      real_t a_2_2 = 0;
      real_t a_2_3 = 0;
      real_t a_2_4 = 0;
      real_t a_2_5 = 0;
      real_t a_3_0 = 0;
      real_t a_3_1 = 0;
      real_t a_3_2 = 0;
      real_t a_3_3 = 0;
      real_t a_3_4 = 0;
      real_t a_3_5 = 0;
      real_t a_4_0 = 0;
      real_t a_4_1 = 0;
      real_t a_4_2 = 0;
      real_t a_4_3 = 0;
      real_t a_4_4 = 0;
      real_t a_4_5 = 0;
      real_t a_5_0 = 0;
      real_t a_5_1 = 0;
      real_t a_5_2 = 0;
      real_t a_5_3 = 0;
      real_t a_5_4 = 0;
      real_t a_5_5 = 0;
      (elMat(0, 0)) = a_0_0;
      (elMat(0, 1)) = a_0_1;
      (elMat(0, 2)) = a_0_2;
      (elMat(0, 3)) = a_0_3;
      (elMat(0, 4)) = a_0_4;
      (elMat(0, 5)) = a_0_5;
      (elMat(1, 0)) = a_1_0;
      (elMat(1, 1)) = a_1_1;
      (elMat(1, 2)) = a_1_2;
      (elMat(1, 3)) = a_1_3;
      (elMat(1, 4)) = a_1_4;
      (elMat(1, 5)) = a_1_5;
      (elMat(2, 0)) = a_2_0;
      (elMat(2, 1)) = a_2_1;
      (elMat(2, 2)) = a_2_2;
      (elMat(2, 3)) = a_2_3;
      (elMat(2, 4)) = a_2_4;
      (elMat(2, 5)) = a_2_5;
      (elMat(3, 0)) = a_3_0;
      (elMat(3, 1)) = a_3_1;
      (elMat(3, 2)) = a_3_2;
      (elMat(3, 3)) = a_3_3;
      (elMat(3, 4)) = a_3_4;
      (elMat(3, 5)) = a_3_5;
      (elMat(4, 0)) = a_4_0;
      (elMat(4, 1)) = a_4_1;
      (elMat(4, 2)) = a_4_2;
      (elMat(4, 3)) = a_4_3;
      (elMat(4, 4)) = a_4_4;
      (elMat(4, 5)) = a_4_5;
      (elMat(5, 0)) = a_5_0;
      (elMat(5, 1)) = a_5_1;
      (elMat(5, 2)) = a_5_2;
      (elMat(5, 3)) = a_5_3;
      (elMat(5, 4)) = a_5_4;
      (elMat(5, 5)) = a_5_5;
   }

   /// \brief Integrates the weak form over the passed element (vertices in computational space).
   ///
   /// - element geometry:                       tetrahedron, dim: 3, vertices: 4
   /// - element matrix dimensions (rows, cols): (10, 10)
   /// - quadrature rule:                        Vioreanu-Rokhlin 1 | points: 4, degree: 2, test tolerance: 2.379e-17
   /// - floating point operations:
   ///                                             adds    muls    divs    abs    assignments    function_calls
   ///                                           ------  ------  ------  -----  -------------  ----------------
   ///                                              499     756       2      1            575                 0
   ///
   void integrateAll( const std::array< Point3D, 4 >& coords, Matrix< real_t, 10, 10 >& elMat ) const override
   {
      real_t p_affine_0_0 = coords[0][0];
      real_t p_affine_0_1 = coords[0][1];
      real_t p_affine_0_2 = coords[0][2];
      real_t p_affine_1_0 = coords[1][0];
      real_t p_affine_1_1 = coords[1][1];
      real_t p_affine_1_2 = coords[1][2];
      real_t p_affine_2_0 = coords[2][0];
      real_t p_affine_2_1 = coords[2][1];
      real_t p_affine_2_2 = coords[2][2];
      real_t p_affine_3_0 = coords[3][0];
      real_t p_affine_3_1 = coords[3][1];
      real_t p_affine_3_2 = coords[3][2];
      real_t q_p_0_0 = 0.13819660112501059;
      real_t q_p_0_1 = 0.13819660112501059;
      real_t q_p_0_2 = 0.58541019662496829;
      real_t q_p_1_0 = 0.13819660112501059;
      real_t q_p_1_1 = 0.58541019662496829;
      real_t q_p_1_2 = 0.13819660112501059;
      real_t q_p_2_0 = 0.58541019662496829;
      real_t q_p_2_1 = 0.13819660112501059;
      real_t q_p_2_2 = 0.13819660112501059;
      real_t q_p_3_0 = 0.13819660112501059;
      real_t q_p_3_1 = 0.13819660112501059;
      real_t q_p_3_2 = 0.13819660112501059;
      real_t w_p_0 = 0.041666666666666657;
      real_t w_p_1 = 0.041666666666666657;
      real_t w_p_2 = 0.041666666666666657;
      real_t w_p_3 = 0.041666666666666657;
      real_t tmp_0 = -p_affine_0_0;
      real_t tmp_1 = p_affine_1_0 + tmp_0;
      real_t tmp_2 = -p_affine_0_1;
      real_t tmp_3 = p_affine_2_1 + tmp_2;
      real_t tmp_4 = tmp_1*tmp_3;
      real_t tmp_5 = p_affine_2_0 + tmp_0;
      real_t tmp_6 = p_affine_1_1 + tmp_2;
      real_t tmp_7 = tmp_5*tmp_6;
      real_t tmp_8 = tmp_4 - tmp_7;
      real_t tmp_9 = -p_affine_0_2;
      real_t tmp_10 = p_affine_3_2 + tmp_9;
      real_t tmp_11 = p_affine_1_2 + tmp_9;
      real_t tmp_12 = p_affine_3_1 + tmp_2;
      real_t tmp_13 = tmp_12*tmp_5;
      real_t tmp_14 = p_affine_2_2 + tmp_9;
      real_t tmp_15 = p_affine_3_0 + tmp_0;
      real_t tmp_16 = tmp_15*tmp_6;
      real_t tmp_17 = tmp_1*tmp_12;
      real_t tmp_18 = tmp_15*tmp_3;
      real_t tmp_19 = tmp_10*tmp_4 - tmp_10*tmp_7 + tmp_11*tmp_13 - tmp_11*tmp_18 + tmp_14*tmp_16 - tmp_14*tmp_17;
      real_t tmp_20 = 1.0 / (tmp_19);
      real_t tmp_21 = 4.0*q_p_0_0;
      real_t tmp_22 = 4.0*q_p_0_1;
      real_t tmp_23 = 4.0*q_p_0_2;
      real_t tmp_24 = tmp_20*(tmp_21 + tmp_22 + tmp_23 - 3.0);
      real_t tmp_25 = tmp_16 - tmp_17;
      real_t tmp_26 = tmp_13 - tmp_18;
      real_t tmp_27 = tmp_24*tmp_25 + tmp_24*tmp_26 + tmp_24*tmp_8;
      real_t tmp_28 = -tmp_11*tmp_3 + tmp_14*tmp_6;
      real_t tmp_29 = 0.5*tmp_24;
      real_t tmp_30 = -tmp_10*tmp_6 + tmp_11*tmp_12;
      real_t tmp_31 = tmp_10*tmp_3 - tmp_12*tmp_14;
      real_t tmp_32 = tmp_28*tmp_29 + tmp_29*tmp_30 + tmp_29*tmp_31;
      real_t tmp_33 = p_affine_0_0*p_affine_1_1;
      real_t tmp_34 = p_affine_0_0*p_affine_1_2;
      real_t tmp_35 = p_affine_2_1*p_affine_3_2;
      real_t tmp_36 = p_affine_0_1*p_affine_1_0;
      real_t tmp_37 = p_affine_0_1*p_affine_1_2;
      real_t tmp_38 = p_affine_2_2*p_affine_3_0;
      real_t tmp_39 = p_affine_0_2*p_affine_1_0;
      real_t tmp_40 = p_affine_0_2*p_affine_1_1;
      real_t tmp_41 = p_affine_2_0*p_affine_3_1;
      real_t tmp_42 = p_affine_2_2*p_affine_3_1;
      real_t tmp_43 = p_affine_2_0*p_affine_3_2;
      real_t tmp_44 = p_affine_2_1*p_affine_3_0;
      real_t tmp_45 = std::abs(p_affine_0_0*tmp_35 - p_affine_0_0*tmp_42 + p_affine_0_1*tmp_38 - p_affine_0_1*tmp_43 + p_affine_0_2*tmp_41 - p_affine_0_2*tmp_44 - p_affine_1_0*tmp_35 + p_affine_1_0*tmp_42 - p_affine_1_1*tmp_38 + p_affine_1_1*tmp_43 - p_affine_1_2*tmp_41 + p_affine_1_2*tmp_44 + p_affine_2_0*tmp_37 - p_affine_2_0*tmp_40 - p_affine_2_1*tmp_34 + p_affine_2_1*tmp_39 + p_affine_2_2*tmp_33 - p_affine_2_2*tmp_36 - p_affine_3_0*tmp_37 + p_affine_3_0*tmp_40 + p_affine_3_1*tmp_34 - p_affine_3_1*tmp_39 - p_affine_3_2*tmp_33 + p_affine_3_2*tmp_36);
      real_t tmp_46 = 2*tmp_45;
      real_t tmp_47 = tmp_46*w_p_0;
      real_t tmp_48 = tmp_32*tmp_47;
      real_t tmp_49 = 4.0*q_p_1_0;
      real_t tmp_50 = 4.0*q_p_1_1;
      real_t tmp_51 = 4.0*q_p_1_2;
      real_t tmp_52 = tmp_20*(tmp_49 + tmp_50 + tmp_51 - 3.0);
      real_t tmp_53 = tmp_25*tmp_52 + tmp_26*tmp_52 + tmp_52*tmp_8;
      real_t tmp_54 = 0.5*tmp_52;
      real_t tmp_55 = tmp_28*tmp_54 + tmp_30*tmp_54 + tmp_31*tmp_54;
      real_t tmp_56 = tmp_46*w_p_1;
      real_t tmp_57 = tmp_55*tmp_56;
      real_t tmp_58 = 4.0*q_p_2_0;
      real_t tmp_59 = 4.0*q_p_2_1;
      real_t tmp_60 = 4.0*q_p_2_2;
      real_t tmp_61 = tmp_20*(tmp_58 + tmp_59 + tmp_60 - 3.0);
      real_t tmp_62 = tmp_25*tmp_61 + tmp_26*tmp_61 + tmp_61*tmp_8;
      real_t tmp_63 = 0.5*tmp_61;
      real_t tmp_64 = tmp_28*tmp_63 + tmp_30*tmp_63 + tmp_31*tmp_63;
      real_t tmp_65 = tmp_46*w_p_2;
      real_t tmp_66 = tmp_64*tmp_65;
      real_t tmp_67 = 4.0*q_p_3_0;
      real_t tmp_68 = 4.0*q_p_3_1;
      real_t tmp_69 = 4.0*q_p_3_2;
      real_t tmp_70 = tmp_20*(tmp_67 + tmp_68 + tmp_69 - 3.0);
      real_t tmp_71 = tmp_25*tmp_70 + tmp_26*tmp_70 + tmp_70*tmp_8;
      real_t tmp_72 = 0.5*tmp_70;
      real_t tmp_73 = tmp_28*tmp_72 + tmp_30*tmp_72 + tmp_31*tmp_72;
      real_t tmp_74 = tmp_46*w_p_3;
      real_t tmp_75 = tmp_73*tmp_74;
      real_t tmp_76 = tmp_20*tmp_26;
      real_t tmp_77 = 2.0*tmp_45;
      real_t tmp_78 = tmp_76*tmp_77;
      real_t tmp_79 = w_p_0*(tmp_21 - 1.0);
      real_t tmp_80 = tmp_78*tmp_79;
      real_t tmp_81 = tmp_55*w_p_1;
      real_t tmp_82 = tmp_49 - 1.0;
      real_t tmp_83 = tmp_78*tmp_82;
      real_t tmp_84 = tmp_64*w_p_2;
      real_t tmp_85 = tmp_58 - 1.0;
      real_t tmp_86 = tmp_78*tmp_85;
      real_t tmp_87 = tmp_73*w_p_3;
      real_t tmp_88 = tmp_67 - 1.0;
      real_t tmp_89 = tmp_78*tmp_88;
      real_t tmp_90 = tmp_20*tmp_25;
      real_t tmp_91 = tmp_77*tmp_90;
      real_t tmp_92 = tmp_22 - 1.0;
      real_t tmp_93 = tmp_92*w_p_0;
      real_t tmp_94 = tmp_91*tmp_93;
      real_t tmp_95 = tmp_50 - 1.0;
      real_t tmp_96 = tmp_91*tmp_95;
      real_t tmp_97 = tmp_59 - 1.0;
      real_t tmp_98 = tmp_91*tmp_97;
      real_t tmp_99 = tmp_68 - 1.0;
      real_t tmp_100 = tmp_91*tmp_99;
      real_t tmp_101 = tmp_20*tmp_8;
      real_t tmp_102 = tmp_101*tmp_77;
      real_t tmp_103 = tmp_23 - 1.0;
      real_t tmp_104 = tmp_103*w_p_0;
      real_t tmp_105 = tmp_102*tmp_104;
      real_t tmp_106 = tmp_51 - 1.0;
      real_t tmp_107 = tmp_102*tmp_106;
      real_t tmp_108 = tmp_60 - 1.0;
      real_t tmp_109 = tmp_102*tmp_108;
      real_t tmp_110 = tmp_69 - 1.0;
      real_t tmp_111 = tmp_102*tmp_110;
      real_t tmp_112 = tmp_101*tmp_22;
      real_t tmp_113 = tmp_23*tmp_90;
      real_t tmp_114 = tmp_112 + tmp_113;
      real_t tmp_115 = tmp_101*tmp_50;
      real_t tmp_116 = tmp_51*tmp_90;
      real_t tmp_117 = tmp_115 + tmp_116;
      real_t tmp_118 = tmp_101*tmp_59;
      real_t tmp_119 = tmp_60*tmp_90;
      real_t tmp_120 = tmp_118 + tmp_119;
      real_t tmp_121 = tmp_101*tmp_68;
      real_t tmp_122 = tmp_69*tmp_90;
      real_t tmp_123 = tmp_121 + tmp_122;
      real_t tmp_124 = tmp_101*tmp_21;
      real_t tmp_125 = tmp_23*tmp_76;
      real_t tmp_126 = tmp_124 + tmp_125;
      real_t tmp_127 = tmp_101*tmp_49;
      real_t tmp_128 = tmp_51*tmp_76;
      real_t tmp_129 = tmp_127 + tmp_128;
      real_t tmp_130 = tmp_101*tmp_58;
      real_t tmp_131 = tmp_60*tmp_76;
      real_t tmp_132 = tmp_130 + tmp_131;
      real_t tmp_133 = tmp_101*tmp_67;
      real_t tmp_134 = tmp_69*tmp_76;
      real_t tmp_135 = tmp_133 + tmp_134;
      real_t tmp_136 = tmp_21*tmp_90;
      real_t tmp_137 = tmp_22*tmp_76;
      real_t tmp_138 = tmp_136 + tmp_137;
      real_t tmp_139 = tmp_49*tmp_90;
      real_t tmp_140 = tmp_50*tmp_76;
      real_t tmp_141 = tmp_139 + tmp_140;
      real_t tmp_142 = tmp_58*tmp_90;
      real_t tmp_143 = tmp_59*tmp_76;
      real_t tmp_144 = tmp_142 + tmp_143;
      real_t tmp_145 = tmp_67*tmp_90;
      real_t tmp_146 = tmp_68*tmp_76;
      real_t tmp_147 = tmp_145 + tmp_146;
      real_t tmp_148 = -tmp_22;
      real_t tmp_149 = 4.0 - tmp_21;
      real_t tmp_150 = -8.0*q_p_0_2 + tmp_148 + tmp_149;
      real_t tmp_151 = tmp_101*tmp_150 - tmp_113 - tmp_125;
      real_t tmp_152 = -tmp_50;
      real_t tmp_153 = 4.0 - tmp_49;
      real_t tmp_154 = -8.0*q_p_1_2 + tmp_152 + tmp_153;
      real_t tmp_155 = tmp_101*tmp_154 - tmp_116 - tmp_128;
      real_t tmp_156 = -tmp_59;
      real_t tmp_157 = 4.0 - tmp_58;
      real_t tmp_158 = -8.0*q_p_2_2 + tmp_156 + tmp_157;
      real_t tmp_159 = tmp_101*tmp_158 - tmp_119 - tmp_131;
      real_t tmp_160 = -tmp_68;
      real_t tmp_161 = 4.0 - tmp_67;
      real_t tmp_162 = -8.0*q_p_3_2 + tmp_160 + tmp_161;
      real_t tmp_163 = tmp_101*tmp_162 - tmp_122 - tmp_134;
      real_t tmp_164 = -tmp_23;
      real_t tmp_165 = -8.0*q_p_0_1 + tmp_149 + tmp_164;
      real_t tmp_166 = -tmp_112 - tmp_137 + tmp_165*tmp_90;
      real_t tmp_167 = -tmp_51;
      real_t tmp_168 = -8.0*q_p_1_1 + tmp_153 + tmp_167;
      real_t tmp_169 = -tmp_115 - tmp_140 + tmp_168*tmp_90;
      real_t tmp_170 = -tmp_60;
      real_t tmp_171 = -8.0*q_p_2_1 + tmp_157 + tmp_170;
      real_t tmp_172 = -tmp_118 - tmp_143 + tmp_171*tmp_90;
      real_t tmp_173 = -tmp_69;
      real_t tmp_174 = -8.0*q_p_3_1 + tmp_161 + tmp_173;
      real_t tmp_175 = -tmp_121 - tmp_146 + tmp_174*tmp_90;
      real_t tmp_176 = -8.0*q_p_0_0 + tmp_148 + tmp_164 + 4.0;
      real_t tmp_177 = -tmp_124 - tmp_136 + tmp_176*tmp_76;
      real_t tmp_178 = -8.0*q_p_1_0 + tmp_152 + tmp_167 + 4.0;
      real_t tmp_179 = -tmp_127 - tmp_139 + tmp_178*tmp_76;
      real_t tmp_180 = -8.0*q_p_2_0 + tmp_156 + tmp_170 + 4.0;
      real_t tmp_181 = -tmp_130 - tmp_142 + tmp_180*tmp_76;
      real_t tmp_182 = -8.0*q_p_3_0 + tmp_160 + tmp_173 + 4.0;
      real_t tmp_183 = -tmp_133 - tmp_145 + tmp_182*tmp_76;
      real_t tmp_184 = tmp_31*tmp_45;
      real_t tmp_185 = 1.0*tmp_20;
      real_t tmp_186 = tmp_184*tmp_185;
      real_t tmp_187 = tmp_186*tmp_79;
      real_t tmp_188 = tmp_53*w_p_1;
      real_t tmp_189 = tmp_186*tmp_82;
      real_t tmp_190 = tmp_62*w_p_2;
      real_t tmp_191 = tmp_186*tmp_85;
      real_t tmp_192 = tmp_71*w_p_3;
      real_t tmp_193 = tmp_186*tmp_88;
      real_t tmp_194 = 1.0 / (tmp_19*tmp_19);
      real_t tmp_195 = 16.0*tmp_194;
      real_t tmp_196 = tmp_195*w_p_0;
      real_t tmp_197 = tmp_184*tmp_26;
      real_t tmp_198 = tmp_195*tmp_197;
      real_t tmp_199 = 1.0*tmp_194;
      real_t tmp_200 = tmp_184*tmp_199;
      real_t tmp_201 = tmp_200*tmp_25;
      real_t tmp_202 = tmp_79*tmp_92;
      real_t tmp_203 = tmp_82*w_p_1;
      real_t tmp_204 = tmp_203*tmp_95;
      real_t tmp_205 = tmp_85*w_p_2;
      real_t tmp_206 = tmp_205*tmp_97;
      real_t tmp_207 = tmp_88*w_p_3;
      real_t tmp_208 = tmp_207*tmp_99;
      real_t tmp_209 = tmp_200*tmp_8;
      real_t tmp_210 = tmp_103*tmp_79;
      real_t tmp_211 = tmp_106*tmp_203;
      real_t tmp_212 = tmp_108*tmp_205;
      real_t tmp_213 = tmp_110*tmp_207;
      real_t tmp_214 = tmp_189*w_p_1;
      real_t tmp_215 = tmp_191*w_p_2;
      real_t tmp_216 = tmp_193*w_p_3;
      real_t tmp_217 = tmp_30*tmp_45;
      real_t tmp_218 = tmp_185*tmp_217;
      real_t tmp_219 = tmp_218*tmp_93;
      real_t tmp_220 = tmp_218*tmp_95;
      real_t tmp_221 = tmp_218*tmp_97;
      real_t tmp_222 = tmp_218*tmp_99;
      real_t tmp_223 = tmp_199*tmp_217;
      real_t tmp_224 = tmp_223*tmp_26;
      real_t tmp_225 = tmp_217*tmp_25;
      real_t tmp_226 = tmp_195*tmp_225;
      real_t tmp_227 = tmp_223*tmp_8;
      real_t tmp_228 = tmp_103*tmp_93;
      real_t tmp_229 = tmp_106*tmp_95*w_p_1;
      real_t tmp_230 = tmp_108*tmp_97*w_p_2;
      real_t tmp_231 = tmp_110*tmp_99*w_p_3;
      real_t tmp_232 = tmp_220*w_p_1;
      real_t tmp_233 = tmp_221*w_p_2;
      real_t tmp_234 = tmp_222*w_p_3;
      real_t tmp_235 = tmp_28*tmp_45;
      real_t tmp_236 = tmp_185*tmp_235;
      real_t tmp_237 = tmp_104*tmp_236;
      real_t tmp_238 = tmp_106*tmp_236;
      real_t tmp_239 = tmp_108*tmp_236;
      real_t tmp_240 = tmp_110*tmp_236;
      real_t tmp_241 = tmp_199*tmp_235;
      real_t tmp_242 = tmp_241*tmp_26;
      real_t tmp_243 = tmp_241*tmp_25;
      real_t tmp_244 = tmp_235*tmp_8;
      real_t tmp_245 = tmp_195*tmp_244;
      real_t tmp_246 = tmp_238*w_p_1;
      real_t tmp_247 = tmp_239*w_p_2;
      real_t tmp_248 = tmp_240*w_p_3;
      real_t tmp_249 = 2.0*tmp_20;
      real_t tmp_250 = tmp_249*tmp_28;
      real_t tmp_251 = q_p_0_1*tmp_250;
      real_t tmp_252 = tmp_249*tmp_30;
      real_t tmp_253 = q_p_0_2*tmp_252;
      real_t tmp_254 = tmp_251 + tmp_253;
      real_t tmp_255 = tmp_254*tmp_47;
      real_t tmp_256 = q_p_1_1*tmp_250;
      real_t tmp_257 = q_p_1_2*tmp_252;
      real_t tmp_258 = tmp_256 + tmp_257;
      real_t tmp_259 = tmp_258*tmp_56;
      real_t tmp_260 = q_p_2_1*tmp_250;
      real_t tmp_261 = q_p_2_2*tmp_252;
      real_t tmp_262 = tmp_260 + tmp_261;
      real_t tmp_263 = tmp_262*tmp_65;
      real_t tmp_264 = q_p_3_1*tmp_250;
      real_t tmp_265 = q_p_3_2*tmp_252;
      real_t tmp_266 = tmp_264 + tmp_265;
      real_t tmp_267 = tmp_266*tmp_74;
      real_t tmp_268 = tmp_258*w_p_1;
      real_t tmp_269 = tmp_262*w_p_2;
      real_t tmp_270 = tmp_266*w_p_3;
      real_t tmp_271 = q_p_0_0*tmp_250;
      real_t tmp_272 = tmp_249*tmp_31;
      real_t tmp_273 = q_p_0_2*tmp_272;
      real_t tmp_274 = tmp_271 + tmp_273;
      real_t tmp_275 = tmp_274*tmp_47;
      real_t tmp_276 = q_p_1_0*tmp_250;
      real_t tmp_277 = q_p_1_2*tmp_272;
      real_t tmp_278 = tmp_276 + tmp_277;
      real_t tmp_279 = tmp_278*tmp_56;
      real_t tmp_280 = q_p_2_0*tmp_250;
      real_t tmp_281 = q_p_2_2*tmp_272;
      real_t tmp_282 = tmp_280 + tmp_281;
      real_t tmp_283 = tmp_282*tmp_65;
      real_t tmp_284 = q_p_3_0*tmp_250;
      real_t tmp_285 = q_p_3_2*tmp_272;
      real_t tmp_286 = tmp_284 + tmp_285;
      real_t tmp_287 = tmp_286*tmp_74;
      real_t tmp_288 = tmp_278*w_p_1;
      real_t tmp_289 = tmp_282*w_p_2;
      real_t tmp_290 = tmp_286*w_p_3;
      real_t tmp_291 = q_p_0_0*tmp_252;
      real_t tmp_292 = q_p_0_1*tmp_272;
      real_t tmp_293 = tmp_291 + tmp_292;
      real_t tmp_294 = tmp_293*tmp_47;
      real_t tmp_295 = q_p_1_0*tmp_252;
      real_t tmp_296 = q_p_1_1*tmp_272;
      real_t tmp_297 = tmp_295 + tmp_296;
      real_t tmp_298 = tmp_297*tmp_56;
      real_t tmp_299 = q_p_2_0*tmp_252;
      real_t tmp_300 = q_p_2_1*tmp_272;
      real_t tmp_301 = tmp_299 + tmp_300;
      real_t tmp_302 = tmp_301*tmp_65;
      real_t tmp_303 = q_p_3_0*tmp_252;
      real_t tmp_304 = q_p_3_1*tmp_272;
      real_t tmp_305 = tmp_303 + tmp_304;
      real_t tmp_306 = tmp_305*tmp_74;
      real_t tmp_307 = tmp_297*w_p_1;
      real_t tmp_308 = tmp_301*w_p_2;
      real_t tmp_309 = tmp_305*w_p_3;
      real_t tmp_310 = 0.5*tmp_20;
      real_t tmp_311 = tmp_28*tmp_310;
      real_t tmp_312 = tmp_150*tmp_311 - tmp_253 - tmp_273;
      real_t tmp_313 = tmp_312*tmp_47;
      real_t tmp_314 = tmp_154*tmp_311 - tmp_257 - tmp_277;
      real_t tmp_315 = tmp_314*tmp_56;
      real_t tmp_316 = tmp_158*tmp_311 - tmp_261 - tmp_281;
      real_t tmp_317 = tmp_316*tmp_65;
      real_t tmp_318 = tmp_162*tmp_311 - tmp_265 - tmp_285;
      real_t tmp_319 = tmp_318*tmp_74;
      real_t tmp_320 = tmp_314*w_p_1;
      real_t tmp_321 = tmp_316*w_p_2;
      real_t tmp_322 = tmp_318*w_p_3;
      real_t tmp_323 = tmp_30*tmp_310;
      real_t tmp_324 = tmp_165*tmp_323 - tmp_251 - tmp_292;
      real_t tmp_325 = tmp_324*tmp_47;
      real_t tmp_326 = tmp_168*tmp_323 - tmp_256 - tmp_296;
      real_t tmp_327 = tmp_326*tmp_56;
      real_t tmp_328 = tmp_171*tmp_323 - tmp_260 - tmp_300;
      real_t tmp_329 = tmp_328*tmp_65;
      real_t tmp_330 = tmp_174*tmp_323 - tmp_264 - tmp_304;
      real_t tmp_331 = tmp_330*tmp_74;
      real_t tmp_332 = tmp_326*w_p_1;
      real_t tmp_333 = tmp_328*w_p_2;
      real_t tmp_334 = tmp_330*w_p_3;
      real_t tmp_335 = tmp_31*tmp_310;
      real_t tmp_336 = tmp_176*tmp_335 - tmp_271 - tmp_291;
      real_t tmp_337 = tmp_336*tmp_47;
      real_t tmp_338 = tmp_178*tmp_335 - tmp_276 - tmp_295;
      real_t tmp_339 = tmp_338*tmp_56;
      real_t tmp_340 = tmp_180*tmp_335 - tmp_280 - tmp_299;
      real_t tmp_341 = tmp_340*tmp_65;
      real_t tmp_342 = tmp_182*tmp_335 - tmp_284 - tmp_303;
      real_t tmp_343 = tmp_342*tmp_74;
      real_t tmp_344 = tmp_338*w_p_1;
      real_t tmp_345 = tmp_340*w_p_2;
      real_t tmp_346 = tmp_342*w_p_3;
      real_t a_0_0 = tmp_27*tmp_48 + tmp_53*tmp_57 + tmp_62*tmp_66 + tmp_71*tmp_75;
      real_t a_0_1 = tmp_32*tmp_80 + tmp_81*tmp_83 + tmp_84*tmp_86 + tmp_87*tmp_89;
      real_t a_0_2 = tmp_100*tmp_87 + tmp_32*tmp_94 + tmp_81*tmp_96 + tmp_84*tmp_98;
      real_t a_0_3 = tmp_105*tmp_32 + tmp_107*tmp_81 + tmp_109*tmp_84 + tmp_111*tmp_87;
      real_t a_0_4 = tmp_114*tmp_48 + tmp_117*tmp_57 + tmp_120*tmp_66 + tmp_123*tmp_75;
      real_t a_0_5 = tmp_126*tmp_48 + tmp_129*tmp_57 + tmp_132*tmp_66 + tmp_135*tmp_75;
      real_t a_0_6 = tmp_138*tmp_48 + tmp_141*tmp_57 + tmp_144*tmp_66 + tmp_147*tmp_75;
      real_t a_0_7 = tmp_151*tmp_48 + tmp_155*tmp_57 + tmp_159*tmp_66 + tmp_163*tmp_75;
      real_t a_0_8 = tmp_166*tmp_48 + tmp_169*tmp_57 + tmp_172*tmp_66 + tmp_175*tmp_75;
      real_t a_0_9 = tmp_177*tmp_48 + tmp_179*tmp_57 + tmp_181*tmp_66 + tmp_183*tmp_75;
      real_t a_1_0 = tmp_187*tmp_27 + tmp_188*tmp_189 + tmp_190*tmp_191 + tmp_192*tmp_193;
      real_t a_1_1 = tmp_196*tmp_197*((q_p_0_0 - 0.25)*(q_p_0_0 - 0.25)) + tmp_198*w_p_1*((q_p_1_0 - 0.25)*(q_p_1_0 - 0.25)) + tmp_198*w_p_2*((q_p_2_0 - 0.25)*(q_p_2_0 - 0.25)) + tmp_198*w_p_3*((q_p_3_0 - 0.25)*(q_p_3_0 - 0.25));
      real_t a_1_2 = tmp_201*tmp_202 + tmp_201*tmp_204 + tmp_201*tmp_206 + tmp_201*tmp_208;
      real_t a_1_3 = tmp_209*tmp_210 + tmp_209*tmp_211 + tmp_209*tmp_212 + tmp_209*tmp_213;
      real_t a_1_4 = tmp_114*tmp_187 + tmp_117*tmp_214 + tmp_120*tmp_215 + tmp_123*tmp_216;
      real_t a_1_5 = tmp_126*tmp_187 + tmp_129*tmp_214 + tmp_132*tmp_215 + tmp_135*tmp_216;
      real_t a_1_6 = tmp_138*tmp_187 + tmp_141*tmp_214 + tmp_144*tmp_215 + tmp_147*tmp_216;
      real_t a_1_7 = tmp_151*tmp_187 + tmp_155*tmp_214 + tmp_159*tmp_215 + tmp_163*tmp_216;
      real_t a_1_8 = tmp_166*tmp_187 + tmp_169*tmp_214 + tmp_172*tmp_215 + tmp_175*tmp_216;
      real_t a_1_9 = tmp_177*tmp_187 + tmp_179*tmp_214 + tmp_181*tmp_215 + tmp_183*tmp_216;
      real_t a_2_0 = tmp_188*tmp_220 + tmp_190*tmp_221 + tmp_192*tmp_222 + tmp_219*tmp_27;
      real_t a_2_1 = tmp_202*tmp_224 + tmp_204*tmp_224 + tmp_206*tmp_224 + tmp_208*tmp_224;
      real_t a_2_2 = tmp_196*tmp_225*((q_p_0_1 - 0.25)*(q_p_0_1 - 0.25)) + tmp_226*w_p_1*((q_p_1_1 - 0.25)*(q_p_1_1 - 0.25)) + tmp_226*w_p_2*((q_p_2_1 - 0.25)*(q_p_2_1 - 0.25)) + tmp_226*w_p_3*((q_p_3_1 - 0.25)*(q_p_3_1 - 0.25));
      real_t a_2_3 = tmp_227*tmp_228 + tmp_227*tmp_229 + tmp_227*tmp_230 + tmp_227*tmp_231;
      real_t a_2_4 = tmp_114*tmp_219 + tmp_117*tmp_232 + tmp_120*tmp_233 + tmp_123*tmp_234;
      real_t a_2_5 = tmp_126*tmp_219 + tmp_129*tmp_232 + tmp_132*tmp_233 + tmp_135*tmp_234;
      real_t a_2_6 = tmp_138*tmp_219 + tmp_141*tmp_232 + tmp_144*tmp_233 + tmp_147*tmp_234;
      real_t a_2_7 = tmp_151*tmp_219 + tmp_155*tmp_232 + tmp_159*tmp_233 + tmp_163*tmp_234;
      real_t a_2_8 = tmp_166*tmp_219 + tmp_169*tmp_232 + tmp_172*tmp_233 + tmp_175*tmp_234;
      real_t a_2_9 = tmp_177*tmp_219 + tmp_179*tmp_232 + tmp_181*tmp_233 + tmp_183*tmp_234;
      real_t a_3_0 = tmp_188*tmp_238 + tmp_190*tmp_239 + tmp_192*tmp_240 + tmp_237*tmp_27;
      real_t a_3_1 = tmp_210*tmp_242 + tmp_211*tmp_242 + tmp_212*tmp_242 + tmp_213*tmp_242;
      real_t a_3_2 = tmp_228*tmp_243 + tmp_229*tmp_243 + tmp_230*tmp_243 + tmp_231*tmp_243;
      real_t a_3_3 = tmp_196*tmp_244*((q_p_0_2 - 0.25)*(q_p_0_2 - 0.25)) + tmp_245*w_p_1*((q_p_1_2 - 0.25)*(q_p_1_2 - 0.25)) + tmp_245*w_p_2*((q_p_2_2 - 0.25)*(q_p_2_2 - 0.25)) + tmp_245*w_p_3*((q_p_3_2 - 0.25)*(q_p_3_2 - 0.25));
      real_t a_3_4 = tmp_114*tmp_237 + tmp_117*tmp_246 + tmp_120*tmp_247 + tmp_123*tmp_248;
      real_t a_3_5 = tmp_126*tmp_237 + tmp_129*tmp_246 + tmp_132*tmp_247 + tmp_135*tmp_248;
      real_t a_3_6 = tmp_138*tmp_237 + tmp_141*tmp_246 + tmp_144*tmp_247 + tmp_147*tmp_248;
      real_t a_3_7 = tmp_151*tmp_237 + tmp_155*tmp_246 + tmp_159*tmp_247 + tmp_163*tmp_248;
      real_t a_3_8 = tmp_166*tmp_237 + tmp_169*tmp_246 + tmp_172*tmp_247 + tmp_175*tmp_248;
      real_t a_3_9 = tmp_177*tmp_237 + tmp_179*tmp_246 + tmp_181*tmp_247 + tmp_183*tmp_248;
      real_t a_4_0 = tmp_255*tmp_27 + tmp_259*tmp_53 + tmp_263*tmp_62 + tmp_267*tmp_71;
      real_t a_4_1 = tmp_254*tmp_80 + tmp_268*tmp_83 + tmp_269*tmp_86 + tmp_270*tmp_89;
      real_t a_4_2 = tmp_100*tmp_270 + tmp_254*tmp_94 + tmp_268*tmp_96 + tmp_269*tmp_98;
      real_t a_4_3 = tmp_105*tmp_254 + tmp_107*tmp_268 + tmp_109*tmp_269 + tmp_111*tmp_270;
      real_t a_4_4 = tmp_114*tmp_255 + tmp_117*tmp_259 + tmp_120*tmp_263 + tmp_123*tmp_267;
      real_t a_4_5 = tmp_126*tmp_255 + tmp_129*tmp_259 + tmp_132*tmp_263 + tmp_135*tmp_267;
      real_t a_4_6 = tmp_138*tmp_255 + tmp_141*tmp_259 + tmp_144*tmp_263 + tmp_147*tmp_267;
      real_t a_4_7 = tmp_151*tmp_255 + tmp_155*tmp_259 + tmp_159*tmp_263 + tmp_163*tmp_267;
      real_t a_4_8 = tmp_166*tmp_255 + tmp_169*tmp_259 + tmp_172*tmp_263 + tmp_175*tmp_267;
      real_t a_4_9 = tmp_177*tmp_255 + tmp_179*tmp_259 + tmp_181*tmp_263 + tmp_183*tmp_267;
      real_t a_5_0 = tmp_27*tmp_275 + tmp_279*tmp_53 + tmp_283*tmp_62 + tmp_287*tmp_71;
      real_t a_5_1 = tmp_274*tmp_80 + tmp_288*tmp_83 + tmp_289*tmp_86 + tmp_290*tmp_89;
      real_t a_5_2 = tmp_100*tmp_290 + tmp_274*tmp_94 + tmp_288*tmp_96 + tmp_289*tmp_98;
      real_t a_5_3 = tmp_105*tmp_274 + tmp_107*tmp_288 + tmp_109*tmp_289 + tmp_111*tmp_290;
      real_t a_5_4 = tmp_114*tmp_275 + tmp_117*tmp_279 + tmp_120*tmp_283 + tmp_123*tmp_287;
      real_t a_5_5 = tmp_126*tmp_275 + tmp_129*tmp_279 + tmp_132*tmp_283 + tmp_135*tmp_287;
      real_t a_5_6 = tmp_138*tmp_275 + tmp_141*tmp_279 + tmp_144*tmp_283 + tmp_147*tmp_287;
      real_t a_5_7 = tmp_151*tmp_275 + tmp_155*tmp_279 + tmp_159*tmp_283 + tmp_163*tmp_287;
      real_t a_5_8 = tmp_166*tmp_275 + tmp_169*tmp_279 + tmp_172*tmp_283 + tmp_175*tmp_287;
      real_t a_5_9 = tmp_177*tmp_275 + tmp_179*tmp_279 + tmp_181*tmp_283 + tmp_183*tmp_287;
      real_t a_6_0 = tmp_27*tmp_294 + tmp_298*tmp_53 + tmp_302*tmp_62 + tmp_306*tmp_71;
      real_t a_6_1 = tmp_293*tmp_80 + tmp_307*tmp_83 + tmp_308*tmp_86 + tmp_309*tmp_89;
      real_t a_6_2 = tmp_100*tmp_309 + tmp_293*tmp_94 + tmp_307*tmp_96 + tmp_308*tmp_98;
      real_t a_6_3 = tmp_105*tmp_293 + tmp_107*tmp_307 + tmp_109*tmp_308 + tmp_111*tmp_309;
      real_t a_6_4 = tmp_114*tmp_294 + tmp_117*tmp_298 + tmp_120*tmp_302 + tmp_123*tmp_306;
      real_t a_6_5 = tmp_126*tmp_294 + tmp_129*tmp_298 + tmp_132*tmp_302 + tmp_135*tmp_306;
      real_t a_6_6 = tmp_138*tmp_294 + tmp_141*tmp_298 + tmp_144*tmp_302 + tmp_147*tmp_306;
      real_t a_6_7 = tmp_151*tmp_294 + tmp_155*tmp_298 + tmp_159*tmp_302 + tmp_163*tmp_306;
      real_t a_6_8 = tmp_166*tmp_294 + tmp_169*tmp_298 + tmp_172*tmp_302 + tmp_175*tmp_306;
      real_t a_6_9 = tmp_177*tmp_294 + tmp_179*tmp_298 + tmp_181*tmp_302 + tmp_183*tmp_306;
      real_t a_7_0 = tmp_27*tmp_313 + tmp_315*tmp_53 + tmp_317*tmp_62 + tmp_319*tmp_71;
      real_t a_7_1 = tmp_312*tmp_80 + tmp_320*tmp_83 + tmp_321*tmp_86 + tmp_322*tmp_89;
      real_t a_7_2 = tmp_100*tmp_322 + tmp_312*tmp_94 + tmp_320*tmp_96 + tmp_321*tmp_98;
      real_t a_7_3 = tmp_105*tmp_312 + tmp_107*tmp_320 + tmp_109*tmp_321 + tmp_111*tmp_322;
      real_t a_7_4 = tmp_114*tmp_313 + tmp_117*tmp_315 + tmp_120*tmp_317 + tmp_123*tmp_319;
      real_t a_7_5 = tmp_126*tmp_313 + tmp_129*tmp_315 + tmp_132*tmp_317 + tmp_135*tmp_319;
      real_t a_7_6 = tmp_138*tmp_313 + tmp_141*tmp_315 + tmp_144*tmp_317 + tmp_147*tmp_319;
      real_t a_7_7 = tmp_151*tmp_313 + tmp_155*tmp_315 + tmp_159*tmp_317 + tmp_163*tmp_319;
      real_t a_7_8 = tmp_166*tmp_313 + tmp_169*tmp_315 + tmp_172*tmp_317 + tmp_175*tmp_319;
      real_t a_7_9 = tmp_177*tmp_313 + tmp_179*tmp_315 + tmp_181*tmp_317 + tmp_183*tmp_319;
      real_t a_8_0 = tmp_27*tmp_325 + tmp_327*tmp_53 + tmp_329*tmp_62 + tmp_331*tmp_71;
      real_t a_8_1 = tmp_324*tmp_80 + tmp_332*tmp_83 + tmp_333*tmp_86 + tmp_334*tmp_89;
      real_t a_8_2 = tmp_100*tmp_334 + tmp_324*tmp_94 + tmp_332*tmp_96 + tmp_333*tmp_98;
      real_t a_8_3 = tmp_105*tmp_324 + tmp_107*tmp_332 + tmp_109*tmp_333 + tmp_111*tmp_334;
      real_t a_8_4 = tmp_114*tmp_325 + tmp_117*tmp_327 + tmp_120*tmp_329 + tmp_123*tmp_331;
      real_t a_8_5 = tmp_126*tmp_325 + tmp_129*tmp_327 + tmp_132*tmp_329 + tmp_135*tmp_331;
      real_t a_8_6 = tmp_138*tmp_325 + tmp_141*tmp_327 + tmp_144*tmp_329 + tmp_147*tmp_331;
      real_t a_8_7 = tmp_151*tmp_325 + tmp_155*tmp_327 + tmp_159*tmp_329 + tmp_163*tmp_331;
      real_t a_8_8 = tmp_166*tmp_325 + tmp_169*tmp_327 + tmp_172*tmp_329 + tmp_175*tmp_331;
      real_t a_8_9 = tmp_177*tmp_325 + tmp_179*tmp_327 + tmp_181*tmp_329 + tmp_183*tmp_331;
      real_t a_9_0 = tmp_27*tmp_337 + tmp_339*tmp_53 + tmp_341*tmp_62 + tmp_343*tmp_71;
      real_t a_9_1 = tmp_336*tmp_80 + tmp_344*tmp_83 + tmp_345*tmp_86 + tmp_346*tmp_89;
      real_t a_9_2 = tmp_100*tmp_346 + tmp_336*tmp_94 + tmp_344*tmp_96 + tmp_345*tmp_98;
      real_t a_9_3 = tmp_105*tmp_336 + tmp_107*tmp_344 + tmp_109*tmp_345 + tmp_111*tmp_346;
      real_t a_9_4 = tmp_114*tmp_337 + tmp_117*tmp_339 + tmp_120*tmp_341 + tmp_123*tmp_343;
      real_t a_9_5 = tmp_126*tmp_337 + tmp_129*tmp_339 + tmp_132*tmp_341 + tmp_135*tmp_343;
      real_t a_9_6 = tmp_138*tmp_337 + tmp_141*tmp_339 + tmp_144*tmp_341 + tmp_147*tmp_343;
      real_t a_9_7 = tmp_151*tmp_337 + tmp_155*tmp_339 + tmp_159*tmp_341 + tmp_163*tmp_343;
      real_t a_9_8 = tmp_166*tmp_337 + tmp_169*tmp_339 + tmp_172*tmp_341 + tmp_175*tmp_343;
      real_t a_9_9 = tmp_177*tmp_337 + tmp_179*tmp_339 + tmp_181*tmp_341 + tmp_183*tmp_343;
      (elMat(0, 0)) = a_0_0;
      (elMat(0, 1)) = a_0_1;
      (elMat(0, 2)) = a_0_2;
      (elMat(0, 3)) = a_0_3;
      (elMat(0, 4)) = a_0_4;
      (elMat(0, 5)) = a_0_5;
      (elMat(0, 6)) = a_0_6;
      (elMat(0, 7)) = a_0_7;
      (elMat(0, 8)) = a_0_8;
      (elMat(0, 9)) = a_0_9;
      (elMat(1, 0)) = a_1_0;
      (elMat(1, 1)) = a_1_1;
      (elMat(1, 2)) = a_1_2;
      (elMat(1, 3)) = a_1_3;
      (elMat(1, 4)) = a_1_4;
      (elMat(1, 5)) = a_1_5;
      (elMat(1, 6)) = a_1_6;
      (elMat(1, 7)) = a_1_7;
      (elMat(1, 8)) = a_1_8;
      (elMat(1, 9)) = a_1_9;
      (elMat(2, 0)) = a_2_0;
      (elMat(2, 1)) = a_2_1;
      (elMat(2, 2)) = a_2_2;
      (elMat(2, 3)) = a_2_3;
      (elMat(2, 4)) = a_2_4;
      (elMat(2, 5)) = a_2_5;
      (elMat(2, 6)) = a_2_6;
      (elMat(2, 7)) = a_2_7;
      (elMat(2, 8)) = a_2_8;
      (elMat(2, 9)) = a_2_9;
      (elMat(3, 0)) = a_3_0;
      (elMat(3, 1)) = a_3_1;
      (elMat(3, 2)) = a_3_2;
      (elMat(3, 3)) = a_3_3;
      (elMat(3, 4)) = a_3_4;
      (elMat(3, 5)) = a_3_5;
      (elMat(3, 6)) = a_3_6;
      (elMat(3, 7)) = a_3_7;
      (elMat(3, 8)) = a_3_8;
      (elMat(3, 9)) = a_3_9;
      (elMat(4, 0)) = a_4_0;
      (elMat(4, 1)) = a_4_1;
      (elMat(4, 2)) = a_4_2;
      (elMat(4, 3)) = a_4_3;
      (elMat(4, 4)) = a_4_4;
      (elMat(4, 5)) = a_4_5;
      (elMat(4, 6)) = a_4_6;
      (elMat(4, 7)) = a_4_7;
      (elMat(4, 8)) = a_4_8;
      (elMat(4, 9)) = a_4_9;
      (elMat(5, 0)) = a_5_0;
      (elMat(5, 1)) = a_5_1;
      (elMat(5, 2)) = a_5_2;
      (elMat(5, 3)) = a_5_3;
      (elMat(5, 4)) = a_5_4;
      (elMat(5, 5)) = a_5_5;
      (elMat(5, 6)) = a_5_6;
      (elMat(5, 7)) = a_5_7;
      (elMat(5, 8)) = a_5_8;
      (elMat(5, 9)) = a_5_9;
      (elMat(6, 0)) = a_6_0;
      (elMat(6, 1)) = a_6_1;
      (elMat(6, 2)) = a_6_2;
      (elMat(6, 3)) = a_6_3;
      (elMat(6, 4)) = a_6_4;
      (elMat(6, 5)) = a_6_5;
      (elMat(6, 6)) = a_6_6;
      (elMat(6, 7)) = a_6_7;
      (elMat(6, 8)) = a_6_8;
      (elMat(6, 9)) = a_6_9;
      (elMat(7, 0)) = a_7_0;
      (elMat(7, 1)) = a_7_1;
      (elMat(7, 2)) = a_7_2;
      (elMat(7, 3)) = a_7_3;
      (elMat(7, 4)) = a_7_4;
      (elMat(7, 5)) = a_7_5;
      (elMat(7, 6)) = a_7_6;
      (elMat(7, 7)) = a_7_7;
      (elMat(7, 8)) = a_7_8;
      (elMat(7, 9)) = a_7_9;
      (elMat(8, 0)) = a_8_0;
      (elMat(8, 1)) = a_8_1;
      (elMat(8, 2)) = a_8_2;
      (elMat(8, 3)) = a_8_3;
      (elMat(8, 4)) = a_8_4;
      (elMat(8, 5)) = a_8_5;
      (elMat(8, 6)) = a_8_6;
      (elMat(8, 7)) = a_8_7;
      (elMat(8, 8)) = a_8_8;
      (elMat(8, 9)) = a_8_9;
      (elMat(9, 0)) = a_9_0;
      (elMat(9, 1)) = a_9_1;
      (elMat(9, 2)) = a_9_2;
      (elMat(9, 3)) = a_9_3;
      (elMat(9, 4)) = a_9_4;
      (elMat(9, 5)) = a_9_5;
      (elMat(9, 6)) = a_9_6;
      (elMat(9, 7)) = a_9_7;
      (elMat(9, 8)) = a_9_8;
      (elMat(9, 9)) = a_9_9;
   }

};

} // namespace hyteg
