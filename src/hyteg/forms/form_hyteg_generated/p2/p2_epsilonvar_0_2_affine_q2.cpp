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

#include "p2_epsilonvar_0_2_affine_q2.hpp"

namespace hyteg {
namespace forms {

   void p2_epsilonvar_0_2_affine_q2::integrateAll( const std::array< Point3D, 3 >& coords, Matrix< real_t, 6, 6 >& elMat ) const
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

   void p2_epsilonvar_0_2_affine_q2::integrateAll( const std::array< Point3D, 4 >& coords, Matrix< real_t, 10, 10 >& elMat ) const
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
      real_t Scalar_Variable_Coefficient_3D_0_0 = 0;
      real_t Scalar_Variable_Coefficient_3D_1_0 = 0;
      real_t Scalar_Variable_Coefficient_3D_2_0 = 0;
      real_t Scalar_Variable_Coefficient_3D_3_0 = 0;
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
      Scalar_Variable_Coefficient_3D( p_affine_0_0 + q_p_0_0*(-p_affine_0_0 + p_affine_1_0) + q_p_0_1*(-p_affine_0_0 + p_affine_2_0) + q_p_0_2*(-p_affine_0_0 + p_affine_3_0), p_affine_0_1 + q_p_0_0*(-p_affine_0_1 + p_affine_1_1) + q_p_0_1*(-p_affine_0_1 + p_affine_2_1) + q_p_0_2*(-p_affine_0_1 + p_affine_3_1), p_affine_0_2 + q_p_0_0*(-p_affine_0_2 + p_affine_1_2) + q_p_0_1*(-p_affine_0_2 + p_affine_2_2) + q_p_0_2*(-p_affine_0_2 + p_affine_3_2), &Scalar_Variable_Coefficient_3D_0_0 );
      Scalar_Variable_Coefficient_3D( p_affine_0_0 + q_p_1_0*(-p_affine_0_0 + p_affine_1_0) + q_p_1_1*(-p_affine_0_0 + p_affine_2_0) + q_p_1_2*(-p_affine_0_0 + p_affine_3_0), p_affine_0_1 + q_p_1_0*(-p_affine_0_1 + p_affine_1_1) + q_p_1_1*(-p_affine_0_1 + p_affine_2_1) + q_p_1_2*(-p_affine_0_1 + p_affine_3_1), p_affine_0_2 + q_p_1_0*(-p_affine_0_2 + p_affine_1_2) + q_p_1_1*(-p_affine_0_2 + p_affine_2_2) + q_p_1_2*(-p_affine_0_2 + p_affine_3_2), &Scalar_Variable_Coefficient_3D_1_0 );
      Scalar_Variable_Coefficient_3D( p_affine_0_0 + q_p_2_0*(-p_affine_0_0 + p_affine_1_0) + q_p_2_1*(-p_affine_0_0 + p_affine_2_0) + q_p_2_2*(-p_affine_0_0 + p_affine_3_0), p_affine_0_1 + q_p_2_0*(-p_affine_0_1 + p_affine_1_1) + q_p_2_1*(-p_affine_0_1 + p_affine_2_1) + q_p_2_2*(-p_affine_0_1 + p_affine_3_1), p_affine_0_2 + q_p_2_0*(-p_affine_0_2 + p_affine_1_2) + q_p_2_1*(-p_affine_0_2 + p_affine_2_2) + q_p_2_2*(-p_affine_0_2 + p_affine_3_2), &Scalar_Variable_Coefficient_3D_2_0 );
      Scalar_Variable_Coefficient_3D( p_affine_0_0 + q_p_3_0*(-p_affine_0_0 + p_affine_1_0) + q_p_3_1*(-p_affine_0_0 + p_affine_2_0) + q_p_3_2*(-p_affine_0_0 + p_affine_3_0), p_affine_0_1 + q_p_3_0*(-p_affine_0_1 + p_affine_1_1) + q_p_3_1*(-p_affine_0_1 + p_affine_2_1) + q_p_3_2*(-p_affine_0_1 + p_affine_3_1), p_affine_0_2 + q_p_3_0*(-p_affine_0_2 + p_affine_1_2) + q_p_3_1*(-p_affine_0_2 + p_affine_2_2) + q_p_3_2*(-p_affine_0_2 + p_affine_3_2), &Scalar_Variable_Coefficient_3D_3_0 );
      real_t tmp_0 = -p_affine_0_1;
      real_t tmp_1 = p_affine_1_1 + tmp_0;
      real_t tmp_2 = -p_affine_0_2;
      real_t tmp_3 = p_affine_2_2 + tmp_2;
      real_t tmp_4 = p_affine_2_1 + tmp_0;
      real_t tmp_5 = p_affine_1_2 + tmp_2;
      real_t tmp_6 = tmp_1*tmp_3 - tmp_4*tmp_5;
      real_t tmp_7 = 4.0*q_p_0_0;
      real_t tmp_8 = 4.0*q_p_0_1;
      real_t tmp_9 = 4.0*q_p_0_2;
      real_t tmp_10 = p_affine_3_2 + tmp_2;
      real_t tmp_11 = -p_affine_0_0;
      real_t tmp_12 = p_affine_1_0 + tmp_11;
      real_t tmp_13 = tmp_12*tmp_4;
      real_t tmp_14 = p_affine_2_0 + tmp_11;
      real_t tmp_15 = p_affine_3_1 + tmp_0;
      real_t tmp_16 = tmp_14*tmp_15;
      real_t tmp_17 = p_affine_3_0 + tmp_11;
      real_t tmp_18 = tmp_1*tmp_17;
      real_t tmp_19 = tmp_12*tmp_15;
      real_t tmp_20 = tmp_1*tmp_14;
      real_t tmp_21 = tmp_17*tmp_4;
      real_t tmp_22 = tmp_10*tmp_13 - tmp_10*tmp_20 + tmp_16*tmp_5 + tmp_18*tmp_3 - tmp_19*tmp_3 - tmp_21*tmp_5;
      real_t tmp_23 = 1.0 / (tmp_22);
      real_t tmp_24 = 0.5*tmp_23;
      real_t tmp_25 = tmp_24*(tmp_7 + tmp_8 + tmp_9 - 3.0);
      real_t tmp_26 = -tmp_1*tmp_10 + tmp_15*tmp_5;
      real_t tmp_27 = tmp_10*tmp_4 - tmp_15*tmp_3;
      real_t tmp_28 = tmp_25*tmp_26 + tmp_25*tmp_27 + tmp_25*tmp_6;
      real_t tmp_29 = tmp_13 - tmp_20;
      real_t tmp_30 = tmp_18 - tmp_19;
      real_t tmp_31 = tmp_16 - tmp_21;
      real_t tmp_32 = tmp_25*tmp_29 + tmp_25*tmp_30 + tmp_25*tmp_31;
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
      real_t tmp_46 = 4*tmp_45;
      real_t tmp_47 = Scalar_Variable_Coefficient_3D_0_0*w_p_0;
      real_t tmp_48 = tmp_46*tmp_47;
      real_t tmp_49 = tmp_32*tmp_48;
      real_t tmp_50 = 4.0*q_p_1_0;
      real_t tmp_51 = 4.0*q_p_1_1;
      real_t tmp_52 = 4.0*q_p_1_2;
      real_t tmp_53 = tmp_24*(tmp_50 + tmp_51 + tmp_52 - 3.0);
      real_t tmp_54 = tmp_26*tmp_53 + tmp_27*tmp_53 + tmp_53*tmp_6;
      real_t tmp_55 = tmp_29*tmp_53 + tmp_30*tmp_53 + tmp_31*tmp_53;
      real_t tmp_56 = Scalar_Variable_Coefficient_3D_1_0*w_p_1;
      real_t tmp_57 = tmp_46*tmp_56;
      real_t tmp_58 = tmp_55*tmp_57;
      real_t tmp_59 = 4.0*q_p_2_0;
      real_t tmp_60 = 4.0*q_p_2_1;
      real_t tmp_61 = 4.0*q_p_2_2;
      real_t tmp_62 = tmp_24*(tmp_59 + tmp_60 + tmp_61 - 3.0);
      real_t tmp_63 = tmp_26*tmp_62 + tmp_27*tmp_62 + tmp_6*tmp_62;
      real_t tmp_64 = tmp_29*tmp_62 + tmp_30*tmp_62 + tmp_31*tmp_62;
      real_t tmp_65 = Scalar_Variable_Coefficient_3D_2_0*w_p_2;
      real_t tmp_66 = tmp_46*tmp_65;
      real_t tmp_67 = tmp_64*tmp_66;
      real_t tmp_68 = 4.0*q_p_3_0;
      real_t tmp_69 = 4.0*q_p_3_1;
      real_t tmp_70 = 4.0*q_p_3_2;
      real_t tmp_71 = tmp_24*(tmp_68 + tmp_69 + tmp_70 - 3.0);
      real_t tmp_72 = tmp_26*tmp_71 + tmp_27*tmp_71 + tmp_6*tmp_71;
      real_t tmp_73 = tmp_29*tmp_71 + tmp_30*tmp_71 + tmp_31*tmp_71;
      real_t tmp_74 = Scalar_Variable_Coefficient_3D_3_0*w_p_3;
      real_t tmp_75 = tmp_46*tmp_74;
      real_t tmp_76 = tmp_73*tmp_75;
      real_t tmp_77 = tmp_32*tmp_47;
      real_t tmp_78 = tmp_7 - 1.0;
      real_t tmp_79 = 2.0*tmp_23;
      real_t tmp_80 = tmp_27*tmp_79;
      real_t tmp_81 = tmp_45*tmp_80;
      real_t tmp_82 = tmp_78*tmp_81;
      real_t tmp_83 = tmp_55*tmp_56;
      real_t tmp_84 = tmp_50 - 1.0;
      real_t tmp_85 = tmp_81*tmp_84;
      real_t tmp_86 = tmp_64*tmp_65;
      real_t tmp_87 = tmp_59 - 1.0;
      real_t tmp_88 = tmp_81*tmp_87;
      real_t tmp_89 = tmp_73*tmp_74;
      real_t tmp_90 = tmp_68 - 1.0;
      real_t tmp_91 = tmp_81*tmp_90;
      real_t tmp_92 = tmp_8 - 1.0;
      real_t tmp_93 = tmp_26*tmp_79;
      real_t tmp_94 = tmp_45*tmp_93;
      real_t tmp_95 = tmp_92*tmp_94;
      real_t tmp_96 = tmp_51 - 1.0;
      real_t tmp_97 = tmp_94*tmp_96;
      real_t tmp_98 = tmp_60 - 1.0;
      real_t tmp_99 = tmp_94*tmp_98;
      real_t tmp_100 = tmp_69 - 1.0;
      real_t tmp_101 = tmp_100*tmp_94;
      real_t tmp_102 = tmp_9 - 1.0;
      real_t tmp_103 = tmp_6*tmp_79;
      real_t tmp_104 = tmp_103*tmp_45;
      real_t tmp_105 = tmp_102*tmp_104;
      real_t tmp_106 = tmp_52 - 1.0;
      real_t tmp_107 = tmp_104*tmp_106;
      real_t tmp_108 = tmp_61 - 1.0;
      real_t tmp_109 = tmp_104*tmp_108;
      real_t tmp_110 = tmp_70 - 1.0;
      real_t tmp_111 = tmp_104*tmp_110;
      real_t tmp_112 = q_p_0_1*tmp_103;
      real_t tmp_113 = q_p_0_2*tmp_93;
      real_t tmp_114 = tmp_112 + tmp_113;
      real_t tmp_115 = q_p_1_1*tmp_103;
      real_t tmp_116 = q_p_1_2*tmp_93;
      real_t tmp_117 = tmp_115 + tmp_116;
      real_t tmp_118 = q_p_2_1*tmp_103;
      real_t tmp_119 = q_p_2_2*tmp_93;
      real_t tmp_120 = tmp_118 + tmp_119;
      real_t tmp_121 = q_p_3_1*tmp_103;
      real_t tmp_122 = q_p_3_2*tmp_93;
      real_t tmp_123 = tmp_121 + tmp_122;
      real_t tmp_124 = q_p_0_0*tmp_103;
      real_t tmp_125 = q_p_0_2*tmp_80;
      real_t tmp_126 = tmp_124 + tmp_125;
      real_t tmp_127 = q_p_1_0*tmp_103;
      real_t tmp_128 = q_p_1_2*tmp_80;
      real_t tmp_129 = tmp_127 + tmp_128;
      real_t tmp_130 = q_p_2_0*tmp_103;
      real_t tmp_131 = q_p_2_2*tmp_80;
      real_t tmp_132 = tmp_130 + tmp_131;
      real_t tmp_133 = q_p_3_0*tmp_103;
      real_t tmp_134 = q_p_3_2*tmp_80;
      real_t tmp_135 = tmp_133 + tmp_134;
      real_t tmp_136 = q_p_0_0*tmp_93;
      real_t tmp_137 = q_p_0_1*tmp_80;
      real_t tmp_138 = tmp_136 + tmp_137;
      real_t tmp_139 = q_p_1_0*tmp_93;
      real_t tmp_140 = q_p_1_1*tmp_80;
      real_t tmp_141 = tmp_139 + tmp_140;
      real_t tmp_142 = q_p_2_0*tmp_93;
      real_t tmp_143 = q_p_2_1*tmp_80;
      real_t tmp_144 = tmp_142 + tmp_143;
      real_t tmp_145 = q_p_3_0*tmp_93;
      real_t tmp_146 = q_p_3_1*tmp_80;
      real_t tmp_147 = tmp_145 + tmp_146;
      real_t tmp_148 = -tmp_8;
      real_t tmp_149 = 4.0 - tmp_7;
      real_t tmp_150 = -8.0*q_p_0_2 + tmp_148 + tmp_149;
      real_t tmp_151 = tmp_24*tmp_6;
      real_t tmp_152 = -tmp_113 - tmp_125 + tmp_150*tmp_151;
      real_t tmp_153 = -tmp_51;
      real_t tmp_154 = 4.0 - tmp_50;
      real_t tmp_155 = -8.0*q_p_1_2 + tmp_153 + tmp_154;
      real_t tmp_156 = -tmp_116 - tmp_128 + tmp_151*tmp_155;
      real_t tmp_157 = -tmp_60;
      real_t tmp_158 = 4.0 - tmp_59;
      real_t tmp_159 = -8.0*q_p_2_2 + tmp_157 + tmp_158;
      real_t tmp_160 = -tmp_119 - tmp_131 + tmp_151*tmp_159;
      real_t tmp_161 = -tmp_69;
      real_t tmp_162 = 4.0 - tmp_68;
      real_t tmp_163 = -8.0*q_p_3_2 + tmp_161 + tmp_162;
      real_t tmp_164 = -tmp_122 - tmp_134 + tmp_151*tmp_163;
      real_t tmp_165 = -tmp_9;
      real_t tmp_166 = -8.0*q_p_0_1 + tmp_149 + tmp_165;
      real_t tmp_167 = tmp_24*tmp_26;
      real_t tmp_168 = -tmp_112 - tmp_137 + tmp_166*tmp_167;
      real_t tmp_169 = -tmp_52;
      real_t tmp_170 = -8.0*q_p_1_1 + tmp_154 + tmp_169;
      real_t tmp_171 = -tmp_115 - tmp_140 + tmp_167*tmp_170;
      real_t tmp_172 = -tmp_61;
      real_t tmp_173 = -8.0*q_p_2_1 + tmp_158 + tmp_172;
      real_t tmp_174 = -tmp_118 - tmp_143 + tmp_167*tmp_173;
      real_t tmp_175 = -tmp_70;
      real_t tmp_176 = -8.0*q_p_3_1 + tmp_162 + tmp_175;
      real_t tmp_177 = -tmp_121 - tmp_146 + tmp_167*tmp_176;
      real_t tmp_178 = -8.0*q_p_0_0 + tmp_148 + tmp_165 + 4.0;
      real_t tmp_179 = tmp_24*tmp_27;
      real_t tmp_180 = -tmp_124 - tmp_136 + tmp_178*tmp_179;
      real_t tmp_181 = -8.0*q_p_1_0 + tmp_153 + tmp_169 + 4.0;
      real_t tmp_182 = -tmp_127 - tmp_139 + tmp_179*tmp_181;
      real_t tmp_183 = -8.0*q_p_2_0 + tmp_157 + tmp_172 + 4.0;
      real_t tmp_184 = -tmp_130 - tmp_142 + tmp_179*tmp_183;
      real_t tmp_185 = -8.0*q_p_3_0 + tmp_161 + tmp_175 + 4.0;
      real_t tmp_186 = -tmp_133 - tmp_145 + tmp_179*tmp_185;
      real_t tmp_187 = tmp_28*tmp_47;
      real_t tmp_188 = tmp_31*tmp_79;
      real_t tmp_189 = tmp_188*tmp_45;
      real_t tmp_190 = tmp_189*tmp_78;
      real_t tmp_191 = tmp_54*tmp_56;
      real_t tmp_192 = tmp_189*tmp_84;
      real_t tmp_193 = tmp_63*tmp_65;
      real_t tmp_194 = tmp_189*tmp_87;
      real_t tmp_195 = tmp_72*tmp_74;
      real_t tmp_196 = tmp_189*tmp_90;
      real_t tmp_197 = tmp_45*tmp_47;
      real_t tmp_198 = 1.0 / (tmp_22*tmp_22);
      real_t tmp_199 = 16.0*tmp_198;
      real_t tmp_200 = tmp_197*tmp_199;
      real_t tmp_201 = tmp_27*tmp_31;
      real_t tmp_202 = tmp_199*tmp_45;
      real_t tmp_203 = tmp_201*tmp_202;
      real_t tmp_204 = 1.0*tmp_198;
      real_t tmp_205 = tmp_204*tmp_31;
      real_t tmp_206 = tmp_205*tmp_26;
      real_t tmp_207 = tmp_197*tmp_92;
      real_t tmp_208 = tmp_207*tmp_78;
      real_t tmp_209 = tmp_206*tmp_45;
      real_t tmp_210 = tmp_56*tmp_84;
      real_t tmp_211 = tmp_210*tmp_96;
      real_t tmp_212 = tmp_65*tmp_87;
      real_t tmp_213 = tmp_212*tmp_98;
      real_t tmp_214 = tmp_74*tmp_90;
      real_t tmp_215 = tmp_100*tmp_214;
      real_t tmp_216 = tmp_205*tmp_6;
      real_t tmp_217 = tmp_102*tmp_197*tmp_78;
      real_t tmp_218 = tmp_216*tmp_45;
      real_t tmp_219 = tmp_106*tmp_210;
      real_t tmp_220 = tmp_108*tmp_212;
      real_t tmp_221 = tmp_110*tmp_214;
      real_t tmp_222 = tmp_190*tmp_47;
      real_t tmp_223 = tmp_192*tmp_56;
      real_t tmp_224 = tmp_194*tmp_65;
      real_t tmp_225 = tmp_196*tmp_74;
      real_t tmp_226 = tmp_30*tmp_79;
      real_t tmp_227 = tmp_226*tmp_45;
      real_t tmp_228 = tmp_227*tmp_96;
      real_t tmp_229 = tmp_227*tmp_98;
      real_t tmp_230 = tmp_100*tmp_227;
      real_t tmp_231 = tmp_204*tmp_30;
      real_t tmp_232 = tmp_231*tmp_27;
      real_t tmp_233 = tmp_232*tmp_45;
      real_t tmp_234 = tmp_26*tmp_30;
      real_t tmp_235 = tmp_202*tmp_234;
      real_t tmp_236 = tmp_231*tmp_6;
      real_t tmp_237 = tmp_102*tmp_207;
      real_t tmp_238 = tmp_236*tmp_45;
      real_t tmp_239 = tmp_106*tmp_56*tmp_96;
      real_t tmp_240 = tmp_108*tmp_65*tmp_98;
      real_t tmp_241 = tmp_100*tmp_110*tmp_74;
      real_t tmp_242 = tmp_114*tmp_197;
      real_t tmp_243 = tmp_226*tmp_92;
      real_t tmp_244 = tmp_228*tmp_56;
      real_t tmp_245 = tmp_229*tmp_65;
      real_t tmp_246 = tmp_230*tmp_74;
      real_t tmp_247 = tmp_197*tmp_243;
      real_t tmp_248 = tmp_29*tmp_79;
      real_t tmp_249 = tmp_248*tmp_45;
      real_t tmp_250 = tmp_106*tmp_249;
      real_t tmp_251 = tmp_108*tmp_249;
      real_t tmp_252 = tmp_110*tmp_249;
      real_t tmp_253 = tmp_204*tmp_29;
      real_t tmp_254 = tmp_253*tmp_27;
      real_t tmp_255 = tmp_254*tmp_45;
      real_t tmp_256 = tmp_253*tmp_26;
      real_t tmp_257 = tmp_256*tmp_45;
      real_t tmp_258 = tmp_29*tmp_6;
      real_t tmp_259 = tmp_202*tmp_258;
      real_t tmp_260 = tmp_102*tmp_248;
      real_t tmp_261 = tmp_250*tmp_56;
      real_t tmp_262 = tmp_251*tmp_65;
      real_t tmp_263 = tmp_252*tmp_74;
      real_t tmp_264 = tmp_197*tmp_260;
      real_t tmp_265 = q_p_0_1*tmp_248;
      real_t tmp_266 = q_p_0_2*tmp_226;
      real_t tmp_267 = tmp_265 + tmp_266;
      real_t tmp_268 = tmp_267*tmp_48;
      real_t tmp_269 = q_p_1_1*tmp_248;
      real_t tmp_270 = q_p_1_2*tmp_226;
      real_t tmp_271 = tmp_269 + tmp_270;
      real_t tmp_272 = tmp_271*tmp_57;
      real_t tmp_273 = q_p_2_1*tmp_248;
      real_t tmp_274 = q_p_2_2*tmp_226;
      real_t tmp_275 = tmp_273 + tmp_274;
      real_t tmp_276 = tmp_275*tmp_66;
      real_t tmp_277 = q_p_3_1*tmp_248;
      real_t tmp_278 = q_p_3_2*tmp_226;
      real_t tmp_279 = tmp_277 + tmp_278;
      real_t tmp_280 = tmp_279*tmp_75;
      real_t tmp_281 = tmp_267*tmp_47;
      real_t tmp_282 = tmp_271*tmp_56;
      real_t tmp_283 = tmp_275*tmp_65;
      real_t tmp_284 = tmp_279*tmp_74;
      real_t tmp_285 = q_p_0_0*tmp_248;
      real_t tmp_286 = q_p_0_2*tmp_188;
      real_t tmp_287 = tmp_285 + tmp_286;
      real_t tmp_288 = tmp_287*tmp_48;
      real_t tmp_289 = q_p_1_0*tmp_248;
      real_t tmp_290 = q_p_1_2*tmp_188;
      real_t tmp_291 = tmp_289 + tmp_290;
      real_t tmp_292 = tmp_291*tmp_57;
      real_t tmp_293 = q_p_2_0*tmp_248;
      real_t tmp_294 = q_p_2_2*tmp_188;
      real_t tmp_295 = tmp_293 + tmp_294;
      real_t tmp_296 = tmp_295*tmp_66;
      real_t tmp_297 = q_p_3_0*tmp_248;
      real_t tmp_298 = q_p_3_2*tmp_188;
      real_t tmp_299 = tmp_297 + tmp_298;
      real_t tmp_300 = tmp_299*tmp_75;
      real_t tmp_301 = tmp_287*tmp_47;
      real_t tmp_302 = tmp_291*tmp_56;
      real_t tmp_303 = tmp_295*tmp_65;
      real_t tmp_304 = tmp_299*tmp_74;
      real_t tmp_305 = q_p_0_0*tmp_226;
      real_t tmp_306 = q_p_0_1*tmp_188;
      real_t tmp_307 = tmp_305 + tmp_306;
      real_t tmp_308 = tmp_307*tmp_48;
      real_t tmp_309 = q_p_1_0*tmp_226;
      real_t tmp_310 = q_p_1_1*tmp_188;
      real_t tmp_311 = tmp_309 + tmp_310;
      real_t tmp_312 = tmp_311*tmp_57;
      real_t tmp_313 = q_p_2_0*tmp_226;
      real_t tmp_314 = q_p_2_1*tmp_188;
      real_t tmp_315 = tmp_313 + tmp_314;
      real_t tmp_316 = tmp_315*tmp_66;
      real_t tmp_317 = q_p_3_0*tmp_226;
      real_t tmp_318 = q_p_3_1*tmp_188;
      real_t tmp_319 = tmp_317 + tmp_318;
      real_t tmp_320 = tmp_319*tmp_75;
      real_t tmp_321 = tmp_307*tmp_47;
      real_t tmp_322 = tmp_311*tmp_56;
      real_t tmp_323 = tmp_315*tmp_65;
      real_t tmp_324 = tmp_319*tmp_74;
      real_t tmp_325 = tmp_24*tmp_29;
      real_t tmp_326 = tmp_150*tmp_325 - tmp_266 - tmp_286;
      real_t tmp_327 = tmp_326*tmp_48;
      real_t tmp_328 = tmp_155*tmp_325 - tmp_270 - tmp_290;
      real_t tmp_329 = tmp_328*tmp_57;
      real_t tmp_330 = tmp_159*tmp_325 - tmp_274 - tmp_294;
      real_t tmp_331 = tmp_330*tmp_66;
      real_t tmp_332 = tmp_163*tmp_325 - tmp_278 - tmp_298;
      real_t tmp_333 = tmp_332*tmp_75;
      real_t tmp_334 = tmp_326*tmp_47;
      real_t tmp_335 = tmp_328*tmp_56;
      real_t tmp_336 = tmp_330*tmp_65;
      real_t tmp_337 = tmp_332*tmp_74;
      real_t tmp_338 = tmp_24*tmp_30;
      real_t tmp_339 = tmp_166*tmp_338 - tmp_265 - tmp_306;
      real_t tmp_340 = tmp_339*tmp_48;
      real_t tmp_341 = tmp_170*tmp_338 - tmp_269 - tmp_310;
      real_t tmp_342 = tmp_341*tmp_57;
      real_t tmp_343 = tmp_173*tmp_338 - tmp_273 - tmp_314;
      real_t tmp_344 = tmp_343*tmp_66;
      real_t tmp_345 = tmp_176*tmp_338 - tmp_277 - tmp_318;
      real_t tmp_346 = tmp_345*tmp_75;
      real_t tmp_347 = tmp_339*tmp_47;
      real_t tmp_348 = tmp_341*tmp_56;
      real_t tmp_349 = tmp_343*tmp_65;
      real_t tmp_350 = tmp_345*tmp_74;
      real_t tmp_351 = tmp_24*tmp_31;
      real_t tmp_352 = tmp_178*tmp_351 - tmp_285 - tmp_305;
      real_t tmp_353 = tmp_352*tmp_48;
      real_t tmp_354 = tmp_181*tmp_351 - tmp_289 - tmp_309;
      real_t tmp_355 = tmp_354*tmp_57;
      real_t tmp_356 = tmp_183*tmp_351 - tmp_293 - tmp_313;
      real_t tmp_357 = tmp_356*tmp_66;
      real_t tmp_358 = tmp_185*tmp_351 - tmp_297 - tmp_317;
      real_t tmp_359 = tmp_358*tmp_75;
      real_t tmp_360 = tmp_352*tmp_47;
      real_t tmp_361 = tmp_354*tmp_56;
      real_t tmp_362 = tmp_356*tmp_65;
      real_t tmp_363 = tmp_358*tmp_74;
      real_t a_0_0 = tmp_28*tmp_49 + tmp_54*tmp_58 + tmp_63*tmp_67 + tmp_72*tmp_76;
      real_t a_0_1 = tmp_77*tmp_82 + tmp_83*tmp_85 + tmp_86*tmp_88 + tmp_89*tmp_91;
      real_t a_0_2 = tmp_101*tmp_89 + tmp_77*tmp_95 + tmp_83*tmp_97 + tmp_86*tmp_99;
      real_t a_0_3 = tmp_105*tmp_77 + tmp_107*tmp_83 + tmp_109*tmp_86 + tmp_111*tmp_89;
      real_t a_0_4 = tmp_114*tmp_49 + tmp_117*tmp_58 + tmp_120*tmp_67 + tmp_123*tmp_76;
      real_t a_0_5 = tmp_126*tmp_49 + tmp_129*tmp_58 + tmp_132*tmp_67 + tmp_135*tmp_76;
      real_t a_0_6 = tmp_138*tmp_49 + tmp_141*tmp_58 + tmp_144*tmp_67 + tmp_147*tmp_76;
      real_t a_0_7 = tmp_152*tmp_49 + tmp_156*tmp_58 + tmp_160*tmp_67 + tmp_164*tmp_76;
      real_t a_0_8 = tmp_168*tmp_49 + tmp_171*tmp_58 + tmp_174*tmp_67 + tmp_177*tmp_76;
      real_t a_0_9 = tmp_180*tmp_49 + tmp_182*tmp_58 + tmp_184*tmp_67 + tmp_186*tmp_76;
      real_t a_1_0 = tmp_187*tmp_190 + tmp_191*tmp_192 + tmp_193*tmp_194 + tmp_195*tmp_196;
      real_t a_1_1 = tmp_200*tmp_201*((q_p_0_0 - 0.25)*(q_p_0_0 - 0.25)) + tmp_203*tmp_56*((q_p_1_0 - 0.25)*(q_p_1_0 - 0.25)) + tmp_203*tmp_65*((q_p_2_0 - 0.25)*(q_p_2_0 - 0.25)) + tmp_203*tmp_74*((q_p_3_0 - 0.25)*(q_p_3_0 - 0.25));
      real_t a_1_2 = tmp_206*tmp_208 + tmp_209*tmp_211 + tmp_209*tmp_213 + tmp_209*tmp_215;
      real_t a_1_3 = tmp_216*tmp_217 + tmp_218*tmp_219 + tmp_218*tmp_220 + tmp_218*tmp_221;
      real_t a_1_4 = tmp_114*tmp_222 + tmp_117*tmp_223 + tmp_120*tmp_224 + tmp_123*tmp_225;
      real_t a_1_5 = tmp_126*tmp_222 + tmp_129*tmp_223 + tmp_132*tmp_224 + tmp_135*tmp_225;
      real_t a_1_6 = tmp_138*tmp_222 + tmp_141*tmp_223 + tmp_144*tmp_224 + tmp_147*tmp_225;
      real_t a_1_7 = tmp_152*tmp_222 + tmp_156*tmp_223 + tmp_160*tmp_224 + tmp_164*tmp_225;
      real_t a_1_8 = tmp_168*tmp_222 + tmp_171*tmp_223 + tmp_174*tmp_224 + tmp_177*tmp_225;
      real_t a_1_9 = tmp_180*tmp_222 + tmp_182*tmp_223 + tmp_184*tmp_224 + tmp_186*tmp_225;
      real_t a_2_0 = tmp_187*tmp_227*tmp_92 + tmp_191*tmp_228 + tmp_193*tmp_229 + tmp_195*tmp_230;
      real_t a_2_1 = tmp_208*tmp_232 + tmp_211*tmp_233 + tmp_213*tmp_233 + tmp_215*tmp_233;
      real_t a_2_2 = tmp_200*tmp_234*((q_p_0_1 - 0.25)*(q_p_0_1 - 0.25)) + tmp_235*tmp_56*((q_p_1_1 - 0.25)*(q_p_1_1 - 0.25)) + tmp_235*tmp_65*((q_p_2_1 - 0.25)*(q_p_2_1 - 0.25)) + tmp_235*tmp_74*((q_p_3_1 - 0.25)*(q_p_3_1 - 0.25));
      real_t a_2_3 = tmp_236*tmp_237 + tmp_238*tmp_239 + tmp_238*tmp_240 + tmp_238*tmp_241;
      real_t a_2_4 = tmp_117*tmp_244 + tmp_120*tmp_245 + tmp_123*tmp_246 + tmp_242*tmp_243;
      real_t a_2_5 = tmp_126*tmp_247 + tmp_129*tmp_244 + tmp_132*tmp_245 + tmp_135*tmp_246;
      real_t a_2_6 = tmp_138*tmp_247 + tmp_141*tmp_244 + tmp_144*tmp_245 + tmp_147*tmp_246;
      real_t a_2_7 = tmp_152*tmp_247 + tmp_156*tmp_244 + tmp_160*tmp_245 + tmp_164*tmp_246;
      real_t a_2_8 = tmp_168*tmp_247 + tmp_171*tmp_244 + tmp_174*tmp_245 + tmp_177*tmp_246;
      real_t a_2_9 = tmp_180*tmp_247 + tmp_182*tmp_244 + tmp_184*tmp_245 + tmp_186*tmp_246;
      real_t a_3_0 = tmp_102*tmp_187*tmp_249 + tmp_191*tmp_250 + tmp_193*tmp_251 + tmp_195*tmp_252;
      real_t a_3_1 = tmp_217*tmp_254 + tmp_219*tmp_255 + tmp_220*tmp_255 + tmp_221*tmp_255;
      real_t a_3_2 = tmp_237*tmp_256 + tmp_239*tmp_257 + tmp_240*tmp_257 + tmp_241*tmp_257;
      real_t a_3_3 = tmp_200*tmp_258*((q_p_0_2 - 0.25)*(q_p_0_2 - 0.25)) + tmp_259*tmp_56*((q_p_1_2 - 0.25)*(q_p_1_2 - 0.25)) + tmp_259*tmp_65*((q_p_2_2 - 0.25)*(q_p_2_2 - 0.25)) + tmp_259*tmp_74*((q_p_3_2 - 0.25)*(q_p_3_2 - 0.25));
      real_t a_3_4 = tmp_117*tmp_261 + tmp_120*tmp_262 + tmp_123*tmp_263 + tmp_242*tmp_260;
      real_t a_3_5 = tmp_126*tmp_264 + tmp_129*tmp_261 + tmp_132*tmp_262 + tmp_135*tmp_263;
      real_t a_3_6 = tmp_138*tmp_264 + tmp_141*tmp_261 + tmp_144*tmp_262 + tmp_147*tmp_263;
      real_t a_3_7 = tmp_152*tmp_264 + tmp_156*tmp_261 + tmp_160*tmp_262 + tmp_164*tmp_263;
      real_t a_3_8 = tmp_168*tmp_264 + tmp_171*tmp_261 + tmp_174*tmp_262 + tmp_177*tmp_263;
      real_t a_3_9 = tmp_180*tmp_264 + tmp_182*tmp_261 + tmp_184*tmp_262 + tmp_186*tmp_263;
      real_t a_4_0 = tmp_268*tmp_28 + tmp_272*tmp_54 + tmp_276*tmp_63 + tmp_280*tmp_72;
      real_t a_4_1 = tmp_281*tmp_82 + tmp_282*tmp_85 + tmp_283*tmp_88 + tmp_284*tmp_91;
      real_t a_4_2 = tmp_101*tmp_284 + tmp_281*tmp_95 + tmp_282*tmp_97 + tmp_283*tmp_99;
      real_t a_4_3 = tmp_105*tmp_281 + tmp_107*tmp_282 + tmp_109*tmp_283 + tmp_111*tmp_284;
      real_t a_4_4 = tmp_114*tmp_268 + tmp_117*tmp_272 + tmp_120*tmp_276 + tmp_123*tmp_280;
      real_t a_4_5 = tmp_126*tmp_268 + tmp_129*tmp_272 + tmp_132*tmp_276 + tmp_135*tmp_280;
      real_t a_4_6 = tmp_138*tmp_268 + tmp_141*tmp_272 + tmp_144*tmp_276 + tmp_147*tmp_280;
      real_t a_4_7 = tmp_152*tmp_268 + tmp_156*tmp_272 + tmp_160*tmp_276 + tmp_164*tmp_280;
      real_t a_4_8 = tmp_168*tmp_268 + tmp_171*tmp_272 + tmp_174*tmp_276 + tmp_177*tmp_280;
      real_t a_4_9 = tmp_180*tmp_268 + tmp_182*tmp_272 + tmp_184*tmp_276 + tmp_186*tmp_280;
      real_t a_5_0 = tmp_28*tmp_288 + tmp_292*tmp_54 + tmp_296*tmp_63 + tmp_300*tmp_72;
      real_t a_5_1 = tmp_301*tmp_82 + tmp_302*tmp_85 + tmp_303*tmp_88 + tmp_304*tmp_91;
      real_t a_5_2 = tmp_101*tmp_304 + tmp_301*tmp_95 + tmp_302*tmp_97 + tmp_303*tmp_99;
      real_t a_5_3 = tmp_105*tmp_301 + tmp_107*tmp_302 + tmp_109*tmp_303 + tmp_111*tmp_304;
      real_t a_5_4 = tmp_114*tmp_288 + tmp_117*tmp_292 + tmp_120*tmp_296 + tmp_123*tmp_300;
      real_t a_5_5 = tmp_126*tmp_288 + tmp_129*tmp_292 + tmp_132*tmp_296 + tmp_135*tmp_300;
      real_t a_5_6 = tmp_138*tmp_288 + tmp_141*tmp_292 + tmp_144*tmp_296 + tmp_147*tmp_300;
      real_t a_5_7 = tmp_152*tmp_288 + tmp_156*tmp_292 + tmp_160*tmp_296 + tmp_164*tmp_300;
      real_t a_5_8 = tmp_168*tmp_288 + tmp_171*tmp_292 + tmp_174*tmp_296 + tmp_177*tmp_300;
      real_t a_5_9 = tmp_180*tmp_288 + tmp_182*tmp_292 + tmp_184*tmp_296 + tmp_186*tmp_300;
      real_t a_6_0 = tmp_28*tmp_308 + tmp_312*tmp_54 + tmp_316*tmp_63 + tmp_320*tmp_72;
      real_t a_6_1 = tmp_321*tmp_82 + tmp_322*tmp_85 + tmp_323*tmp_88 + tmp_324*tmp_91;
      real_t a_6_2 = tmp_101*tmp_324 + tmp_321*tmp_95 + tmp_322*tmp_97 + tmp_323*tmp_99;
      real_t a_6_3 = tmp_105*tmp_321 + tmp_107*tmp_322 + tmp_109*tmp_323 + tmp_111*tmp_324;
      real_t a_6_4 = tmp_114*tmp_308 + tmp_117*tmp_312 + tmp_120*tmp_316 + tmp_123*tmp_320;
      real_t a_6_5 = tmp_126*tmp_308 + tmp_129*tmp_312 + tmp_132*tmp_316 + tmp_135*tmp_320;
      real_t a_6_6 = tmp_138*tmp_308 + tmp_141*tmp_312 + tmp_144*tmp_316 + tmp_147*tmp_320;
      real_t a_6_7 = tmp_152*tmp_308 + tmp_156*tmp_312 + tmp_160*tmp_316 + tmp_164*tmp_320;
      real_t a_6_8 = tmp_168*tmp_308 + tmp_171*tmp_312 + tmp_174*tmp_316 + tmp_177*tmp_320;
      real_t a_6_9 = tmp_180*tmp_308 + tmp_182*tmp_312 + tmp_184*tmp_316 + tmp_186*tmp_320;
      real_t a_7_0 = tmp_28*tmp_327 + tmp_329*tmp_54 + tmp_331*tmp_63 + tmp_333*tmp_72;
      real_t a_7_1 = tmp_334*tmp_82 + tmp_335*tmp_85 + tmp_336*tmp_88 + tmp_337*tmp_91;
      real_t a_7_2 = tmp_101*tmp_337 + tmp_334*tmp_95 + tmp_335*tmp_97 + tmp_336*tmp_99;
      real_t a_7_3 = tmp_105*tmp_334 + tmp_107*tmp_335 + tmp_109*tmp_336 + tmp_111*tmp_337;
      real_t a_7_4 = tmp_114*tmp_327 + tmp_117*tmp_329 + tmp_120*tmp_331 + tmp_123*tmp_333;
      real_t a_7_5 = tmp_126*tmp_327 + tmp_129*tmp_329 + tmp_132*tmp_331 + tmp_135*tmp_333;
      real_t a_7_6 = tmp_138*tmp_327 + tmp_141*tmp_329 + tmp_144*tmp_331 + tmp_147*tmp_333;
      real_t a_7_7 = tmp_152*tmp_327 + tmp_156*tmp_329 + tmp_160*tmp_331 + tmp_164*tmp_333;
      real_t a_7_8 = tmp_168*tmp_327 + tmp_171*tmp_329 + tmp_174*tmp_331 + tmp_177*tmp_333;
      real_t a_7_9 = tmp_180*tmp_327 + tmp_182*tmp_329 + tmp_184*tmp_331 + tmp_186*tmp_333;
      real_t a_8_0 = tmp_28*tmp_340 + tmp_342*tmp_54 + tmp_344*tmp_63 + tmp_346*tmp_72;
      real_t a_8_1 = tmp_347*tmp_82 + tmp_348*tmp_85 + tmp_349*tmp_88 + tmp_350*tmp_91;
      real_t a_8_2 = tmp_101*tmp_350 + tmp_347*tmp_95 + tmp_348*tmp_97 + tmp_349*tmp_99;
      real_t a_8_3 = tmp_105*tmp_347 + tmp_107*tmp_348 + tmp_109*tmp_349 + tmp_111*tmp_350;
      real_t a_8_4 = tmp_114*tmp_340 + tmp_117*tmp_342 + tmp_120*tmp_344 + tmp_123*tmp_346;
      real_t a_8_5 = tmp_126*tmp_340 + tmp_129*tmp_342 + tmp_132*tmp_344 + tmp_135*tmp_346;
      real_t a_8_6 = tmp_138*tmp_340 + tmp_141*tmp_342 + tmp_144*tmp_344 + tmp_147*tmp_346;
      real_t a_8_7 = tmp_152*tmp_340 + tmp_156*tmp_342 + tmp_160*tmp_344 + tmp_164*tmp_346;
      real_t a_8_8 = tmp_168*tmp_340 + tmp_171*tmp_342 + tmp_174*tmp_344 + tmp_177*tmp_346;
      real_t a_8_9 = tmp_180*tmp_340 + tmp_182*tmp_342 + tmp_184*tmp_344 + tmp_186*tmp_346;
      real_t a_9_0 = tmp_28*tmp_353 + tmp_355*tmp_54 + tmp_357*tmp_63 + tmp_359*tmp_72;
      real_t a_9_1 = tmp_360*tmp_82 + tmp_361*tmp_85 + tmp_362*tmp_88 + tmp_363*tmp_91;
      real_t a_9_2 = tmp_101*tmp_363 + tmp_360*tmp_95 + tmp_361*tmp_97 + tmp_362*tmp_99;
      real_t a_9_3 = tmp_105*tmp_360 + tmp_107*tmp_361 + tmp_109*tmp_362 + tmp_111*tmp_363;
      real_t a_9_4 = tmp_114*tmp_353 + tmp_117*tmp_355 + tmp_120*tmp_357 + tmp_123*tmp_359;
      real_t a_9_5 = tmp_126*tmp_353 + tmp_129*tmp_355 + tmp_132*tmp_357 + tmp_135*tmp_359;
      real_t a_9_6 = tmp_138*tmp_353 + tmp_141*tmp_355 + tmp_144*tmp_357 + tmp_147*tmp_359;
      real_t a_9_7 = tmp_152*tmp_353 + tmp_156*tmp_355 + tmp_160*tmp_357 + tmp_164*tmp_359;
      real_t a_9_8 = tmp_168*tmp_353 + tmp_171*tmp_355 + tmp_174*tmp_357 + tmp_177*tmp_359;
      real_t a_9_9 = tmp_180*tmp_353 + tmp_182*tmp_355 + tmp_184*tmp_357 + tmp_186*tmp_359;
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

   void p2_epsilonvar_0_2_affine_q2::Scalar_Variable_Coefficient_3D( real_t in_0, real_t in_1, real_t in_2, real_t * out_0 ) const
   {
      *out_0 = callback3D( Point3D( {in_0, in_1, in_2} ) );
   }

} // namespace forms
} // namespace hyteg
