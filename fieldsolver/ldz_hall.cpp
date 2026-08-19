/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "fs_common.h"
#include "ldz_hall.hpp"
#include <limits>
#include <fftw3.h>
#include "array.hpp"

#ifdef DEBUG_VLASIATOR
   #define DEBUG_FSOLVER
#endif

using namespace std;

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBY Background By
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermComponents
 *
 */
template<typename REAL> inline
REAL JXBX_000_100(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBY,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[a_zz] * BGBZ) / dz + (pC[a_z] * BGBZ) / dz - (pC[a_yz] * BGBZ) / (2 * dz) -
          (pC[c_xzz] * BGBZ) / (6 * dx) + (pC[c_xz] * BGBZ) / (2 * dx) - (pC[c_xyz] * BGBZ) / (4 * dx) +
          (pC[c_xy] * BGBZ) / (2 * dx) - (pC[c_x] * BGBZ) / dx - (pC[a_yz] * BGBY) / (2 * dy) - (pC[a_yy] * BGBY) / dy +
          (pC[a_y] * BGBY) / dy + (pC[b_xz] * BGBY) / (2 * dx) - (pC[b_xyz] * BGBY) / (4 * dx) -
          (pC[b_xyy] * BGBY) / (6 * dx) + (pC[b_xy] * BGBY) / (2 * dx) - (pC[b_x] * BGBY) / dx -
          (pC[a_zz] * pC[c_zz]) / (6 * dz) + (pC[a_z] * pC[c_zz]) / (6 * dz) - (pC[a_yz] * pC[c_zz]) / (12 * dz) +
          (pC[a_zz] * pC[c_z]) / (2 * dz) - (pC[a_z] * pC[c_z]) / (2 * dz) + (pC[a_yz] * pC[c_z]) / (4 * dz) -
          (pC[a_zz] * pC[c_yz]) / (4 * dz) + (pC[a_z] * pC[c_yz]) / (4 * dz) - (pC[a_yz] * pC[c_yz]) / (8 * dz) +
          (pC[a_zz] * pC[c_y]) / (2 * dz) - (pC[a_z] * pC[c_y]) / (2 * dz) + (pC[a_yz] * pC[c_y]) / (4 * dz) +
          (pC[a_xzz] * pC[c_xz]) / (24 * dz) - (pC[a_xz] * pC[c_xz]) / (24 * dz) + (pC[a_xyz] * pC[c_xz]) / (48 * dz) -
          (pC[a_xzz] * pC[c_x]) / (12 * dz) + (pC[a_xz] * pC[c_x]) / (12 * dz) - (pC[a_xyz] * pC[c_x]) / (24 * dz) -
          (pC[a_zz] * pC[c_0]) / dz + (pC[a_z] * pC[c_0]) / dz - (pC[a_yz] * pC[c_0]) / (2 * dz) +
          (pC[a_yz] * pC[b_z]) / (4 * dy) + (pC[a_yy] * pC[b_z]) / (2 * dy) - (pC[a_y] * pC[b_z]) / (2 * dy) -
          (pC[a_yz] * pC[b_yz]) / (8 * dy) - (pC[a_yy] * pC[b_yz]) / (4 * dy) + (pC[a_y] * pC[b_yz]) / (4 * dy) -
          (pC[a_yz] * pC[b_yy]) / (12 * dy) - (pC[a_yy] * pC[b_yy]) / (6 * dy) + (pC[a_y] * pC[b_yy]) / (6 * dy) +
          (pC[a_yz] * pC[b_y]) / (4 * dy) + (pC[a_yy] * pC[b_y]) / (2 * dy) - (pC[a_y] * pC[b_y]) / (2 * dy) +
          (pC[a_xyz] * pC[b_xy]) / (48 * dy) + (pC[a_xyy] * pC[b_xy]) / (24 * dy) - (pC[a_xy] * pC[b_xy]) / (24 * dy) -
          (pC[a_xyz] * pC[b_x]) / (24 * dy) - (pC[a_xyy] * pC[b_x]) / (12 * dy) + (pC[a_xy] * pC[b_x]) / (12 * dy) -
          (pC[a_yz] * pC[b_0]) / (2 * dy) - (pC[a_yy] * pC[b_0]) / dy + (pC[a_y] * pC[b_0]) / dy -
          (pC[c_xzz] * pC[c_zz]) / (36 * dx) + (pC[c_xz] * pC[c_zz]) / (12 * dx) - (pC[c_xyz] * pC[c_zz]) / (24 * dx) +
          (pC[c_xy] * pC[c_zz]) / (12 * dx) - (pC[c_x] * pC[c_zz]) / (6 * dx) + (pC[c_xzz] * pC[c_z]) / (12 * dx) -
          (pC[c_xz] * pC[c_z]) / (4 * dx) + (pC[c_xyz] * pC[c_z]) / (8 * dx) - (pC[c_xy] * pC[c_z]) / (4 * dx) +
          (pC[c_x] * pC[c_z]) / (2 * dx) - (pC[c_xzz] * pC[c_yz]) / (24 * dx) + (pC[c_xz] * pC[c_yz]) / (8 * dx) -
          (pC[c_xyz] * pC[c_yz]) / (16 * dx) + (pC[c_xy] * pC[c_yz]) / (8 * dx) - (pC[c_x] * pC[c_yz]) / (4 * dx) +
          (pC[c_xzz] * pC[c_y]) / (12 * dx) - (pC[c_xz] * pC[c_y]) / (4 * dx) + (pC[c_xyz] * pC[c_y]) / (8 * dx) -
          (pC[c_xy] * pC[c_y]) / (4 * dx) + (pC[c_x] * pC[c_y]) / (2 * dx) - (pC[c_0] * pC[c_xzz]) / (6 * dx) -
          (pC[c_xxz] * pC[c_xz]) / (24 * dx) + (pC[c_xx] * pC[c_xz]) / (12 * dx) + (pC[c_0] * pC[c_xz]) / (2 * dx) -
          (pC[c_0] * pC[c_xyz]) / (4 * dx) + (pC[c_0] * pC[c_xy]) / (2 * dx) + (pC[c_x] * pC[c_xxz]) / (12 * dx) -
          (pC[c_x] * pC[c_xx]) / (6 * dx) - (pC[c_0] * pC[c_x]) / dx - (pC[b_xz] * pC[b_z]) / (4 * dx) +
          (pC[b_xyz] * pC[b_z]) / (8 * dx) + (pC[b_xyy] * pC[b_z]) / (12 * dx) - (pC[b_xy] * pC[b_z]) / (4 * dx) +
          (pC[b_x] * pC[b_z]) / (2 * dx) + (pC[b_xz] * pC[b_yz]) / (8 * dx) - (pC[b_xyz] * pC[b_yz]) / (16 * dx) -
          (pC[b_xyy] * pC[b_yz]) / (24 * dx) + (pC[b_xy] * pC[b_yz]) / (8 * dx) - (pC[b_x] * pC[b_yz]) / (4 * dx) +
          (pC[b_xz] * pC[b_yy]) / (12 * dx) - (pC[b_xyz] * pC[b_yy]) / (24 * dx) - (pC[b_xyy] * pC[b_yy]) / (36 * dx) +
          (pC[b_xy] * pC[b_yy]) / (12 * dx) - (pC[b_x] * pC[b_yy]) / (6 * dx) - (pC[b_xz] * pC[b_y]) / (4 * dx) +
          (pC[b_xyz] * pC[b_y]) / (8 * dx) + (pC[b_xyy] * pC[b_y]) / (12 * dx) - (pC[b_xy] * pC[b_y]) / (4 * dx) +
          (pC[b_x] * pC[b_y]) / (2 * dx) + (pC[b_0] * pC[b_xz]) / (2 * dx) - (pC[b_0] * pC[b_xyz]) / (4 * dx) -
          (pC[b_0] * pC[b_xyy]) / (6 * dx) - (pC[b_xxy] * pC[b_xy]) / (24 * dx) + (pC[b_xx] * pC[b_xy]) / (12 * dx) +
          (pC[b_0] * pC[b_xy]) / (2 * dx) + (pC[b_x] * pC[b_xxy]) / (12 * dx) - (pC[b_x] * pC[b_xx]) / (6 * dx) -
          (pC[b_0] * pC[b_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBY Background By
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermComponents
 *
 */
template<typename REAL> inline
REAL JXBX_010_110(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBY,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[a_zz] * BGBZ) / dz + (pC[a_z] * BGBZ) / dz + (pC[a_yz] * BGBZ) / (2 * dz) -
          (pC[c_xzz] * BGBZ) / (6 * dx) + (pC[c_xz] * BGBZ) / (2 * dx) + (pC[c_xyz] * BGBZ) / (4 * dx) -
          (pC[c_xy] * BGBZ) / (2 * dx) - (pC[c_x] * BGBZ) / dx - (pC[a_yz] * BGBY) / (2 * dy) + (pC[a_yy] * BGBY) / dy +
          (pC[a_y] * BGBY) / dy + (pC[b_xz] * BGBY) / (2 * dx) + (pC[b_xyz] * BGBY) / (4 * dx) -
          (pC[b_xyy] * BGBY) / (6 * dx) - (pC[b_xy] * BGBY) / (2 * dx) - (pC[b_x] * BGBY) / dx -
          (pC[a_zz] * pC[c_zz]) / (6 * dz) + (pC[a_z] * pC[c_zz]) / (6 * dz) + (pC[a_yz] * pC[c_zz]) / (12 * dz) +
          (pC[a_zz] * pC[c_z]) / (2 * dz) - (pC[a_z] * pC[c_z]) / (2 * dz) - (pC[a_yz] * pC[c_z]) / (4 * dz) +
          (pC[a_zz] * pC[c_yz]) / (4 * dz) - (pC[a_z] * pC[c_yz]) / (4 * dz) - (pC[a_yz] * pC[c_yz]) / (8 * dz) -
          (pC[a_zz] * pC[c_y]) / (2 * dz) + (pC[a_z] * pC[c_y]) / (2 * dz) + (pC[a_yz] * pC[c_y]) / (4 * dz) +
          (pC[a_xzz] * pC[c_xz]) / (24 * dz) - (pC[a_xz] * pC[c_xz]) / (24 * dz) - (pC[a_xyz] * pC[c_xz]) / (48 * dz) -
          (pC[a_xzz] * pC[c_x]) / (12 * dz) + (pC[a_xz] * pC[c_x]) / (12 * dz) + (pC[a_xyz] * pC[c_x]) / (24 * dz) -
          (pC[a_zz] * pC[c_0]) / dz + (pC[a_z] * pC[c_0]) / dz + (pC[a_yz] * pC[c_0]) / (2 * dz) +
          (pC[a_yz] * pC[b_z]) / (4 * dy) - (pC[a_yy] * pC[b_z]) / (2 * dy) - (pC[a_y] * pC[b_z]) / (2 * dy) +
          (pC[a_yz] * pC[b_yz]) / (8 * dy) - (pC[a_yy] * pC[b_yz]) / (4 * dy) - (pC[a_y] * pC[b_yz]) / (4 * dy) -
          (pC[a_yz] * pC[b_yy]) / (12 * dy) + (pC[a_yy] * pC[b_yy]) / (6 * dy) + (pC[a_y] * pC[b_yy]) / (6 * dy) -
          (pC[a_yz] * pC[b_y]) / (4 * dy) + (pC[a_yy] * pC[b_y]) / (2 * dy) + (pC[a_y] * pC[b_y]) / (2 * dy) -
          (pC[a_xyz] * pC[b_xy]) / (48 * dy) + (pC[a_xyy] * pC[b_xy]) / (24 * dy) + (pC[a_xy] * pC[b_xy]) / (24 * dy) -
          (pC[a_xyz] * pC[b_x]) / (24 * dy) + (pC[a_xyy] * pC[b_x]) / (12 * dy) + (pC[a_xy] * pC[b_x]) / (12 * dy) -
          (pC[a_yz] * pC[b_0]) / (2 * dy) + (pC[a_yy] * pC[b_0]) / dy + (pC[a_y] * pC[b_0]) / dy -
          (pC[c_xzz] * pC[c_zz]) / (36 * dx) + (pC[c_xz] * pC[c_zz]) / (12 * dx) + (pC[c_xyz] * pC[c_zz]) / (24 * dx) -
          (pC[c_xy] * pC[c_zz]) / (12 * dx) - (pC[c_x] * pC[c_zz]) / (6 * dx) + (pC[c_xzz] * pC[c_z]) / (12 * dx) -
          (pC[c_xz] * pC[c_z]) / (4 * dx) - (pC[c_xyz] * pC[c_z]) / (8 * dx) + (pC[c_xy] * pC[c_z]) / (4 * dx) +
          (pC[c_x] * pC[c_z]) / (2 * dx) + (pC[c_xzz] * pC[c_yz]) / (24 * dx) - (pC[c_xz] * pC[c_yz]) / (8 * dx) -
          (pC[c_xyz] * pC[c_yz]) / (16 * dx) + (pC[c_xy] * pC[c_yz]) / (8 * dx) + (pC[c_x] * pC[c_yz]) / (4 * dx) -
          (pC[c_xzz] * pC[c_y]) / (12 * dx) + (pC[c_xz] * pC[c_y]) / (4 * dx) + (pC[c_xyz] * pC[c_y]) / (8 * dx) -
          (pC[c_xy] * pC[c_y]) / (4 * dx) - (pC[c_x] * pC[c_y]) / (2 * dx) - (pC[c_0] * pC[c_xzz]) / (6 * dx) -
          (pC[c_xxz] * pC[c_xz]) / (24 * dx) + (pC[c_xx] * pC[c_xz]) / (12 * dx) + (pC[c_0] * pC[c_xz]) / (2 * dx) +
          (pC[c_0] * pC[c_xyz]) / (4 * dx) - (pC[c_0] * pC[c_xy]) / (2 * dx) + (pC[c_x] * pC[c_xxz]) / (12 * dx) -
          (pC[c_x] * pC[c_xx]) / (6 * dx) - (pC[c_0] * pC[c_x]) / dx - (pC[b_xz] * pC[b_z]) / (4 * dx) -
          (pC[b_xyz] * pC[b_z]) / (8 * dx) + (pC[b_xyy] * pC[b_z]) / (12 * dx) + (pC[b_xy] * pC[b_z]) / (4 * dx) +
          (pC[b_x] * pC[b_z]) / (2 * dx) - (pC[b_xz] * pC[b_yz]) / (8 * dx) - (pC[b_xyz] * pC[b_yz]) / (16 * dx) +
          (pC[b_xyy] * pC[b_yz]) / (24 * dx) + (pC[b_xy] * pC[b_yz]) / (8 * dx) + (pC[b_x] * pC[b_yz]) / (4 * dx) +
          (pC[b_xz] * pC[b_yy]) / (12 * dx) + (pC[b_xyz] * pC[b_yy]) / (24 * dx) - (pC[b_xyy] * pC[b_yy]) / (36 * dx) -
          (pC[b_xy] * pC[b_yy]) / (12 * dx) - (pC[b_x] * pC[b_yy]) / (6 * dx) + (pC[b_xz] * pC[b_y]) / (4 * dx) +
          (pC[b_xyz] * pC[b_y]) / (8 * dx) - (pC[b_xyy] * pC[b_y]) / (12 * dx) - (pC[b_xy] * pC[b_y]) / (4 * dx) -
          (pC[b_x] * pC[b_y]) / (2 * dx) + (pC[b_0] * pC[b_xz]) / (2 * dx) + (pC[b_0] * pC[b_xyz]) / (4 * dx) -
          (pC[b_0] * pC[b_xyy]) / (6 * dx) - (pC[b_xxy] * pC[b_xy]) / (24 * dx) - (pC[b_xx] * pC[b_xy]) / (12 * dx) -
          (pC[b_0] * pC[b_xy]) / (2 * dx) - (pC[b_x] * pC[b_xxy]) / (12 * dx) - (pC[b_x] * pC[b_xx]) / (6 * dx) -
          (pC[b_0] * pC[b_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBY Background By
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermComponents
 *
 */
template<typename REAL> inline
REAL JXBX_001_101(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBY,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return (pC[a_zz] * BGBZ) / dz + (pC[a_z] * BGBZ) / dz - (pC[a_yz] * BGBZ) / (2 * dz) -
          (pC[c_xzz] * BGBZ) / (6 * dx) - (pC[c_xz] * BGBZ) / (2 * dx) + (pC[c_xyz] * BGBZ) / (4 * dx) +
          (pC[c_xy] * BGBZ) / (2 * dx) - (pC[c_x] * BGBZ) / dx + (pC[a_yz] * BGBY) / (2 * dy) - (pC[a_yy] * BGBY) / dy +
          (pC[a_y] * BGBY) / dy - (pC[b_xz] * BGBY) / (2 * dx) + (pC[b_xyz] * BGBY) / (4 * dx) -
          (pC[b_xyy] * BGBY) / (6 * dx) + (pC[b_xy] * BGBY) / (2 * dx) - (pC[b_x] * BGBY) / dx +
          (pC[a_zz] * pC[c_zz]) / (6 * dz) + (pC[a_z] * pC[c_zz]) / (6 * dz) - (pC[a_yz] * pC[c_zz]) / (12 * dz) +
          (pC[a_zz] * pC[c_z]) / (2 * dz) + (pC[a_z] * pC[c_z]) / (2 * dz) - (pC[a_yz] * pC[c_z]) / (4 * dz) -
          (pC[a_zz] * pC[c_yz]) / (4 * dz) - (pC[a_z] * pC[c_yz]) / (4 * dz) + (pC[a_yz] * pC[c_yz]) / (8 * dz) -
          (pC[a_zz] * pC[c_y]) / (2 * dz) - (pC[a_z] * pC[c_y]) / (2 * dz) + (pC[a_yz] * pC[c_y]) / (4 * dz) +
          (pC[a_xzz] * pC[c_xz]) / (24 * dz) + (pC[a_xz] * pC[c_xz]) / (24 * dz) - (pC[a_xyz] * pC[c_xz]) / (48 * dz) +
          (pC[a_xzz] * pC[c_x]) / (12 * dz) + (pC[a_xz] * pC[c_x]) / (12 * dz) - (pC[a_xyz] * pC[c_x]) / (24 * dz) +
          (pC[a_zz] * pC[c_0]) / dz + (pC[a_z] * pC[c_0]) / dz - (pC[a_yz] * pC[c_0]) / (2 * dz) +
          (pC[a_yz] * pC[b_z]) / (4 * dy) - (pC[a_yy] * pC[b_z]) / (2 * dy) + (pC[a_y] * pC[b_z]) / (2 * dy) -
          (pC[a_yz] * pC[b_yz]) / (8 * dy) + (pC[a_yy] * pC[b_yz]) / (4 * dy) - (pC[a_y] * pC[b_yz]) / (4 * dy) +
          (pC[a_yz] * pC[b_yy]) / (12 * dy) - (pC[a_yy] * pC[b_yy]) / (6 * dy) + (pC[a_y] * pC[b_yy]) / (6 * dy) -
          (pC[a_yz] * pC[b_y]) / (4 * dy) + (pC[a_yy] * pC[b_y]) / (2 * dy) - (pC[a_y] * pC[b_y]) / (2 * dy) -
          (pC[a_xyz] * pC[b_xy]) / (48 * dy) + (pC[a_xyy] * pC[b_xy]) / (24 * dy) - (pC[a_xy] * pC[b_xy]) / (24 * dy) +
          (pC[a_xyz] * pC[b_x]) / (24 * dy) - (pC[a_xyy] * pC[b_x]) / (12 * dy) + (pC[a_xy] * pC[b_x]) / (12 * dy) +
          (pC[a_yz] * pC[b_0]) / (2 * dy) - (pC[a_yy] * pC[b_0]) / dy + (pC[a_y] * pC[b_0]) / dy -
          (pC[c_xzz] * pC[c_zz]) / (36 * dx) - (pC[c_xz] * pC[c_zz]) / (12 * dx) + (pC[c_xyz] * pC[c_zz]) / (24 * dx) +
          (pC[c_xy] * pC[c_zz]) / (12 * dx) - (pC[c_x] * pC[c_zz]) / (6 * dx) - (pC[c_xzz] * pC[c_z]) / (12 * dx) -
          (pC[c_xz] * pC[c_z]) / (4 * dx) + (pC[c_xyz] * pC[c_z]) / (8 * dx) + (pC[c_xy] * pC[c_z]) / (4 * dx) -
          (pC[c_x] * pC[c_z]) / (2 * dx) + (pC[c_xzz] * pC[c_yz]) / (24 * dx) + (pC[c_xz] * pC[c_yz]) / (8 * dx) -
          (pC[c_xyz] * pC[c_yz]) / (16 * dx) - (pC[c_xy] * pC[c_yz]) / (8 * dx) + (pC[c_x] * pC[c_yz]) / (4 * dx) +
          (pC[c_xzz] * pC[c_y]) / (12 * dx) + (pC[c_xz] * pC[c_y]) / (4 * dx) - (pC[c_xyz] * pC[c_y]) / (8 * dx) -
          (pC[c_xy] * pC[c_y]) / (4 * dx) + (pC[c_x] * pC[c_y]) / (2 * dx) - (pC[c_0] * pC[c_xzz]) / (6 * dx) -
          (pC[c_xxz] * pC[c_xz]) / (24 * dx) - (pC[c_xx] * pC[c_xz]) / (12 * dx) - (pC[c_0] * pC[c_xz]) / (2 * dx) +
          (pC[c_0] * pC[c_xyz]) / (4 * dx) + (pC[c_0] * pC[c_xy]) / (2 * dx) - (pC[c_x] * pC[c_xxz]) / (12 * dx) -
          (pC[c_x] * pC[c_xx]) / (6 * dx) - (pC[c_0] * pC[c_x]) / dx - (pC[b_xz] * pC[b_z]) / (4 * dx) +
          (pC[b_xyz] * pC[b_z]) / (8 * dx) - (pC[b_xyy] * pC[b_z]) / (12 * dx) + (pC[b_xy] * pC[b_z]) / (4 * dx) -
          (pC[b_x] * pC[b_z]) / (2 * dx) + (pC[b_xz] * pC[b_yz]) / (8 * dx) - (pC[b_xyz] * pC[b_yz]) / (16 * dx) +
          (pC[b_xyy] * pC[b_yz]) / (24 * dx) - (pC[b_xy] * pC[b_yz]) / (8 * dx) + (pC[b_x] * pC[b_yz]) / (4 * dx) -
          (pC[b_xz] * pC[b_yy]) / (12 * dx) + (pC[b_xyz] * pC[b_yy]) / (24 * dx) - (pC[b_xyy] * pC[b_yy]) / (36 * dx) +
          (pC[b_xy] * pC[b_yy]) / (12 * dx) - (pC[b_x] * pC[b_yy]) / (6 * dx) + (pC[b_xz] * pC[b_y]) / (4 * dx) -
          (pC[b_xyz] * pC[b_y]) / (8 * dx) + (pC[b_xyy] * pC[b_y]) / (12 * dx) - (pC[b_xy] * pC[b_y]) / (4 * dx) +
          (pC[b_x] * pC[b_y]) / (2 * dx) - (pC[b_0] * pC[b_xz]) / (2 * dx) + (pC[b_0] * pC[b_xyz]) / (4 * dx) -
          (pC[b_0] * pC[b_xyy]) / (6 * dx) - (pC[b_xxy] * pC[b_xy]) / (24 * dx) + (pC[b_xx] * pC[b_xy]) / (12 * dx) +
          (pC[b_0] * pC[b_xy]) / (2 * dx) + (pC[b_x] * pC[b_xxy]) / (12 * dx) - (pC[b_x] * pC[b_xx]) / (6 * dx) -
          (pC[b_0] * pC[b_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBY Background By
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermComponents
 *
 */
template<typename REAL> inline
REAL JXBX_011_111(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBY,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return (pC[a_zz] * BGBZ) / dz + (pC[a_z] * BGBZ) / dz + (pC[a_yz] * BGBZ) / (2 * dz) -
          (pC[c_xzz] * BGBZ) / (6 * dx) - (pC[c_xz] * BGBZ) / (2 * dx) - (pC[c_xyz] * BGBZ) / (4 * dx) -
          (pC[c_xy] * BGBZ) / (2 * dx) - (pC[c_x] * BGBZ) / dx + (pC[a_yz] * BGBY) / (2 * dy) + (pC[a_yy] * BGBY) / dy +
          (pC[a_y] * BGBY) / dy - (pC[b_xz] * BGBY) / (2 * dx) - (pC[b_xyz] * BGBY) / (4 * dx) -
          (pC[b_xyy] * BGBY) / (6 * dx) - (pC[b_xy] * BGBY) / (2 * dx) - (pC[b_x] * BGBY) / dx +
          (pC[a_zz] * pC[c_zz]) / (6 * dz) + (pC[a_z] * pC[c_zz]) / (6 * dz) + (pC[a_yz] * pC[c_zz]) / (12 * dz) +
          (pC[a_zz] * pC[c_z]) / (2 * dz) + (pC[a_z] * pC[c_z]) / (2 * dz) + (pC[a_yz] * pC[c_z]) / (4 * dz) +
          (pC[a_zz] * pC[c_yz]) / (4 * dz) + (pC[a_z] * pC[c_yz]) / (4 * dz) + (pC[a_yz] * pC[c_yz]) / (8 * dz) +
          (pC[a_zz] * pC[c_y]) / (2 * dz) + (pC[a_z] * pC[c_y]) / (2 * dz) + (pC[a_yz] * pC[c_y]) / (4 * dz) +
          (pC[a_xzz] * pC[c_xz]) / (24 * dz) + (pC[a_xz] * pC[c_xz]) / (24 * dz) + (pC[a_xyz] * pC[c_xz]) / (48 * dz) +
          (pC[a_xzz] * pC[c_x]) / (12 * dz) + (pC[a_xz] * pC[c_x]) / (12 * dz) + (pC[a_xyz] * pC[c_x]) / (24 * dz) +
          (pC[a_zz] * pC[c_0]) / dz + (pC[a_z] * pC[c_0]) / dz + (pC[a_yz] * pC[c_0]) / (2 * dz) +
          (pC[a_yz] * pC[b_z]) / (4 * dy) + (pC[a_yy] * pC[b_z]) / (2 * dy) + (pC[a_y] * pC[b_z]) / (2 * dy) +
          (pC[a_yz] * pC[b_yz]) / (8 * dy) + (pC[a_yy] * pC[b_yz]) / (4 * dy) + (pC[a_y] * pC[b_yz]) / (4 * dy) +
          (pC[a_yz] * pC[b_yy]) / (12 * dy) + (pC[a_yy] * pC[b_yy]) / (6 * dy) + (pC[a_y] * pC[b_yy]) / (6 * dy) +
          (pC[a_yz] * pC[b_y]) / (4 * dy) + (pC[a_yy] * pC[b_y]) / (2 * dy) + (pC[a_y] * pC[b_y]) / (2 * dy) +
          (pC[a_xyz] * pC[b_xy]) / (48 * dy) + (pC[a_xyy] * pC[b_xy]) / (24 * dy) + (pC[a_xy] * pC[b_xy]) / (24 * dy) +
          (pC[a_xyz] * pC[b_x]) / (24 * dy) + (pC[a_xyy] * pC[b_x]) / (12 * dy) + (pC[a_xy] * pC[b_x]) / (12 * dy) +
          (pC[a_yz] * pC[b_0]) / (2 * dy) + (pC[a_yy] * pC[b_0]) / dy + (pC[a_y] * pC[b_0]) / dy -
          (pC[c_xzz] * pC[c_zz]) / (36 * dx) - (pC[c_xz] * pC[c_zz]) / (12 * dx) - (pC[c_xyz] * pC[c_zz]) / (24 * dx) -
          (pC[c_xy] * pC[c_zz]) / (12 * dx) - (pC[c_x] * pC[c_zz]) / (6 * dx) - (pC[c_xzz] * pC[c_z]) / (12 * dx) -
          (pC[c_xz] * pC[c_z]) / (4 * dx) - (pC[c_xyz] * pC[c_z]) / (8 * dx) - (pC[c_xy] * pC[c_z]) / (4 * dx) -
          (pC[c_x] * pC[c_z]) / (2 * dx) - (pC[c_xzz] * pC[c_yz]) / (24 * dx) - (pC[c_xz] * pC[c_yz]) / (8 * dx) -
          (pC[c_xyz] * pC[c_yz]) / (16 * dx) - (pC[c_xy] * pC[c_yz]) / (8 * dx) - (pC[c_x] * pC[c_yz]) / (4 * dx) -
          (pC[c_xzz] * pC[c_y]) / (12 * dx) - (pC[c_xz] * pC[c_y]) / (4 * dx) - (pC[c_xyz] * pC[c_y]) / (8 * dx) -
          (pC[c_xy] * pC[c_y]) / (4 * dx) - (pC[c_x] * pC[c_y]) / (2 * dx) - (pC[c_0] * pC[c_xzz]) / (6 * dx) -
          (pC[c_xxz] * pC[c_xz]) / (24 * dx) - (pC[c_xx] * pC[c_xz]) / (12 * dx) - (pC[c_0] * pC[c_xz]) / (2 * dx) -
          (pC[c_0] * pC[c_xyz]) / (4 * dx) - (pC[c_0] * pC[c_xy]) / (2 * dx) - (pC[c_x] * pC[c_xxz]) / (12 * dx) -
          (pC[c_x] * pC[c_xx]) / (6 * dx) - (pC[c_0] * pC[c_x]) / dx - (pC[b_xz] * pC[b_z]) / (4 * dx) -
          (pC[b_xyz] * pC[b_z]) / (8 * dx) - (pC[b_xyy] * pC[b_z]) / (12 * dx) - (pC[b_xy] * pC[b_z]) / (4 * dx) -
          (pC[b_x] * pC[b_z]) / (2 * dx) - (pC[b_xz] * pC[b_yz]) / (8 * dx) - (pC[b_xyz] * pC[b_yz]) / (16 * dx) -
          (pC[b_xyy] * pC[b_yz]) / (24 * dx) - (pC[b_xy] * pC[b_yz]) / (8 * dx) - (pC[b_x] * pC[b_yz]) / (4 * dx) -
          (pC[b_xz] * pC[b_yy]) / (12 * dx) - (pC[b_xyz] * pC[b_yy]) / (24 * dx) - (pC[b_xyy] * pC[b_yy]) / (36 * dx) -
          (pC[b_xy] * pC[b_yy]) / (12 * dx) - (pC[b_x] * pC[b_yy]) / (6 * dx) - (pC[b_xz] * pC[b_y]) / (4 * dx) -
          (pC[b_xyz] * pC[b_y]) / (8 * dx) - (pC[b_xyy] * pC[b_y]) / (12 * dx) - (pC[b_xy] * pC[b_y]) / (4 * dx) -
          (pC[b_x] * pC[b_y]) / (2 * dx) - (pC[b_0] * pC[b_xz]) / (2 * dx) - (pC[b_0] * pC[b_xyz]) / (4 * dx) -
          (pC[b_0] * pC[b_xyy]) / (6 * dx) - (pC[b_xxy] * pC[b_xy]) / (24 * dx) - (pC[b_xx] * pC[b_xy]) / (12 * dx) -
          (pC[b_0] * pC[b_xy]) / (2 * dx) - (pC[b_x] * pC[b_xxy]) / (12 * dx) - (pC[b_x] * pC[b_xx]) / (6 * dx) -
          (pC[b_0] * pC[b_x]) / dx;
}

// Y
/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermYComponents
 *
 */
template<typename REAL> inline
REAL JXBY_000_010(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[b_zz] * BGBZ) / dz + (pC[b_z] * BGBZ) / dz - (pC[b_xz] * BGBZ) / (2 * dz) -
          (pC[c_yzz] * BGBZ) / (6 * dy) + (pC[c_yz] * BGBZ) / (2 * dy) - (pC[c_y] * BGBZ) / dy -
          (pC[c_xyz] * BGBZ) / (4 * dy) + (pC[c_xy] * BGBZ) / (2 * dy) + (pC[a_yz] * BGBX) / (2 * dy) -
          (pC[a_y] * BGBX) / dy - (pC[a_xyz] * BGBX) / (4 * dy) + (pC[a_xy] * BGBX) / (2 * dy) -
          (pC[a_xxy] * BGBX) / (6 * dy) - (pC[b_xz] * BGBX) / (2 * dx) - (pC[b_xx] * BGBX) / dx +
          (pC[b_x] * BGBX) / dx - (pC[b_zz] * pC[c_zz]) / (6 * dz) + (pC[b_z] * pC[c_zz]) / (6 * dz) -
          (pC[b_xz] * pC[c_zz]) / (12 * dz) + (pC[b_zz] * pC[c_z]) / (2 * dz) - (pC[b_z] * pC[c_z]) / (2 * dz) +
          (pC[b_xz] * pC[c_z]) / (4 * dz) + (pC[b_yzz] * pC[c_yz]) / (24 * dz) - (pC[b_yz] * pC[c_yz]) / (24 * dz) +
          (pC[b_xyz] * pC[c_yz]) / (48 * dz) - (pC[b_yzz] * pC[c_y]) / (12 * dz) + (pC[b_yz] * pC[c_y]) / (12 * dz) -
          (pC[b_xyz] * pC[c_y]) / (24 * dz) - (pC[b_zz] * pC[c_xz]) / (4 * dz) + (pC[b_z] * pC[c_xz]) / (4 * dz) -
          (pC[b_xz] * pC[c_xz]) / (8 * dz) + (pC[b_zz] * pC[c_x]) / (2 * dz) - (pC[b_z] * pC[c_x]) / (2 * dz) +
          (pC[b_xz] * pC[c_x]) / (4 * dz) - (pC[b_zz] * pC[c_0]) / dz + (pC[b_z] * pC[c_0]) / dz -
          (pC[b_xz] * pC[c_0]) / (2 * dz) - (pC[c_yzz] * pC[c_zz]) / (36 * dy) + (pC[c_yz] * pC[c_zz]) / (12 * dy) -
          (pC[c_y] * pC[c_zz]) / (6 * dy) - (pC[c_xyz] * pC[c_zz]) / (24 * dy) + (pC[c_xy] * pC[c_zz]) / (12 * dy) +
          (pC[c_yzz] * pC[c_z]) / (12 * dy) - (pC[c_yz] * pC[c_z]) / (4 * dy) + (pC[c_y] * pC[c_z]) / (2 * dy) +
          (pC[c_xyz] * pC[c_z]) / (8 * dy) - (pC[c_xy] * pC[c_z]) / (4 * dy) - (pC[c_xz] * pC[c_yzz]) / (24 * dy) +
          (pC[c_x] * pC[c_yzz]) / (12 * dy) - (pC[c_0] * pC[c_yzz]) / (6 * dy) - (pC[c_yyz] * pC[c_yz]) / (24 * dy) +
          (pC[c_yy] * pC[c_yz]) / (12 * dy) + (pC[c_xz] * pC[c_yz]) / (8 * dy) - (pC[c_x] * pC[c_yz]) / (4 * dy) +
          (pC[c_0] * pC[c_yz]) / (2 * dy) + (pC[c_y] * pC[c_yyz]) / (12 * dy) - (pC[c_y] * pC[c_yy]) / (6 * dy) -
          (pC[c_xz] * pC[c_y]) / (4 * dy) + (pC[c_x] * pC[c_y]) / (2 * dy) - (pC[c_0] * pC[c_y]) / dy -
          (pC[c_xyz] * pC[c_xz]) / (16 * dy) + (pC[c_xy] * pC[c_xz]) / (8 * dy) + (pC[c_x] * pC[c_xyz]) / (8 * dy) -
          (pC[c_0] * pC[c_xyz]) / (4 * dy) - (pC[c_x] * pC[c_xy]) / (4 * dy) + (pC[c_0] * pC[c_xy]) / (2 * dy) -
          (pC[a_yz] * pC[a_z]) / (4 * dy) + (pC[a_y] * pC[a_z]) / (2 * dy) + (pC[a_xyz] * pC[a_z]) / (8 * dy) -
          (pC[a_xy] * pC[a_z]) / (4 * dy) + (pC[a_xxy] * pC[a_z]) / (12 * dy) + (pC[a_xz] * pC[a_yz]) / (8 * dy) +
          (pC[a_xx] * pC[a_yz]) / (12 * dy) - (pC[a_x] * pC[a_yz]) / (4 * dy) + (pC[a_0] * pC[a_yz]) / (2 * dy) -
          (pC[a_y] * pC[a_yy]) / (6 * dy) + (pC[a_xy] * pC[a_yy]) / (12 * dy) - (pC[a_xz] * pC[a_y]) / (4 * dy) +
          (pC[a_xyy] * pC[a_y]) / (12 * dy) - (pC[a_xx] * pC[a_y]) / (6 * dy) + (pC[a_x] * pC[a_y]) / (2 * dy) -
          (pC[a_0] * pC[a_y]) / dy - (pC[a_xyz] * pC[a_xz]) / (16 * dy) + (pC[a_xy] * pC[a_xz]) / (8 * dy) -
          (pC[a_xxy] * pC[a_xz]) / (24 * dy) - (pC[a_xx] * pC[a_xyz]) / (24 * dy) + (pC[a_x] * pC[a_xyz]) / (8 * dy) -
          (pC[a_0] * pC[a_xyz]) / (4 * dy) - (pC[a_xy] * pC[a_xyy]) / (24 * dy) + (pC[a_xx] * pC[a_xy]) / (12 * dy) -
          (pC[a_x] * pC[a_xy]) / (4 * dy) + (pC[a_0] * pC[a_xy]) / (2 * dy) - (pC[a_xx] * pC[a_xxy]) / (36 * dy) +
          (pC[a_x] * pC[a_xxy]) / (12 * dy) - (pC[a_0] * pC[a_xxy]) / (6 * dy) + (pC[a_z] * pC[b_xz]) / (4 * dx) -
          (pC[a_xz] * pC[b_xz]) / (8 * dx) - (pC[a_xx] * pC[b_xz]) / (12 * dx) + (pC[a_x] * pC[b_xz]) / (4 * dx) -
          (pC[a_0] * pC[b_xz]) / (2 * dx) - (pC[a_y] * pC[b_xyz]) / (24 * dx) + (pC[a_xy] * pC[b_xyz]) / (48 * dx) +
          (pC[a_y] * pC[b_xy]) / (12 * dx) - (pC[a_xy] * pC[b_xy]) / (24 * dx) - (pC[a_y] * pC[b_xxy]) / (12 * dx) +
          (pC[a_xy] * pC[b_xxy]) / (24 * dx) + (pC[a_z] * pC[b_xx]) / (2 * dx) - (pC[a_xz] * pC[b_xx]) / (4 * dx) -
          (pC[a_xx] * pC[b_xx]) / (6 * dx) + (pC[a_x] * pC[b_xx]) / (2 * dx) - (pC[a_0] * pC[b_xx]) / dx -
          (pC[a_z] * pC[b_x]) / (2 * dx) + (pC[a_xz] * pC[b_x]) / (4 * dx) + (pC[a_xx] * pC[b_x]) / (6 * dx) -
          (pC[a_x] * pC[b_x]) / (2 * dx) + (pC[a_0] * pC[b_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermYComponents
 *
 */
template<typename REAL> inline
REAL JXBY_100_110(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[b_zz] * BGBZ) / dz + (pC[b_z] * BGBZ) / dz + (pC[b_xz] * BGBZ) / (2 * dz) -
          (pC[c_yzz] * BGBZ) / (6 * dy) + (pC[c_yz] * BGBZ) / (2 * dy) - (pC[c_y] * BGBZ) / dy +
          (pC[c_xyz] * BGBZ) / (4 * dy) - (pC[c_xy] * BGBZ) / (2 * dy) + (pC[a_yz] * BGBX) / (2 * dy) -
          (pC[a_y] * BGBX) / dy + (pC[a_xyz] * BGBX) / (4 * dy) - (pC[a_xy] * BGBX) / (2 * dy) -
          (pC[a_xxy] * BGBX) / (6 * dy) - (pC[b_xz] * BGBX) / (2 * dx) + (pC[b_xx] * BGBX) / dx +
          (pC[b_x] * BGBX) / dx - (pC[b_zz] * pC[c_zz]) / (6 * dz) + (pC[b_z] * pC[c_zz]) / (6 * dz) +
          (pC[b_xz] * pC[c_zz]) / (12 * dz) + (pC[b_zz] * pC[c_z]) / (2 * dz) - (pC[b_z] * pC[c_z]) / (2 * dz) -
          (pC[b_xz] * pC[c_z]) / (4 * dz) + (pC[b_yzz] * pC[c_yz]) / (24 * dz) - (pC[b_yz] * pC[c_yz]) / (24 * dz) -
          (pC[b_xyz] * pC[c_yz]) / (48 * dz) - (pC[b_yzz] * pC[c_y]) / (12 * dz) + (pC[b_yz] * pC[c_y]) / (12 * dz) +
          (pC[b_xyz] * pC[c_y]) / (24 * dz) + (pC[b_zz] * pC[c_xz]) / (4 * dz) - (pC[b_z] * pC[c_xz]) / (4 * dz) -
          (pC[b_xz] * pC[c_xz]) / (8 * dz) - (pC[b_zz] * pC[c_x]) / (2 * dz) + (pC[b_z] * pC[c_x]) / (2 * dz) +
          (pC[b_xz] * pC[c_x]) / (4 * dz) - (pC[b_zz] * pC[c_0]) / dz + (pC[b_z] * pC[c_0]) / dz +
          (pC[b_xz] * pC[c_0]) / (2 * dz) - (pC[c_yzz] * pC[c_zz]) / (36 * dy) + (pC[c_yz] * pC[c_zz]) / (12 * dy) -
          (pC[c_y] * pC[c_zz]) / (6 * dy) + (pC[c_xyz] * pC[c_zz]) / (24 * dy) - (pC[c_xy] * pC[c_zz]) / (12 * dy) +
          (pC[c_yzz] * pC[c_z]) / (12 * dy) - (pC[c_yz] * pC[c_z]) / (4 * dy) + (pC[c_y] * pC[c_z]) / (2 * dy) -
          (pC[c_xyz] * pC[c_z]) / (8 * dy) + (pC[c_xy] * pC[c_z]) / (4 * dy) + (pC[c_xz] * pC[c_yzz]) / (24 * dy) -
          (pC[c_x] * pC[c_yzz]) / (12 * dy) - (pC[c_0] * pC[c_yzz]) / (6 * dy) - (pC[c_yyz] * pC[c_yz]) / (24 * dy) +
          (pC[c_yy] * pC[c_yz]) / (12 * dy) - (pC[c_xz] * pC[c_yz]) / (8 * dy) + (pC[c_x] * pC[c_yz]) / (4 * dy) +
          (pC[c_0] * pC[c_yz]) / (2 * dy) + (pC[c_y] * pC[c_yyz]) / (12 * dy) - (pC[c_y] * pC[c_yy]) / (6 * dy) +
          (pC[c_xz] * pC[c_y]) / (4 * dy) - (pC[c_x] * pC[c_y]) / (2 * dy) - (pC[c_0] * pC[c_y]) / dy -
          (pC[c_xyz] * pC[c_xz]) / (16 * dy) + (pC[c_xy] * pC[c_xz]) / (8 * dy) + (pC[c_x] * pC[c_xyz]) / (8 * dy) +
          (pC[c_0] * pC[c_xyz]) / (4 * dy) - (pC[c_x] * pC[c_xy]) / (4 * dy) - (pC[c_0] * pC[c_xy]) / (2 * dy) -
          (pC[a_yz] * pC[a_z]) / (4 * dy) + (pC[a_y] * pC[a_z]) / (2 * dy) - (pC[a_xyz] * pC[a_z]) / (8 * dy) +
          (pC[a_xy] * pC[a_z]) / (4 * dy) + (pC[a_xxy] * pC[a_z]) / (12 * dy) - (pC[a_xz] * pC[a_yz]) / (8 * dy) +
          (pC[a_xx] * pC[a_yz]) / (12 * dy) + (pC[a_x] * pC[a_yz]) / (4 * dy) + (pC[a_0] * pC[a_yz]) / (2 * dy) -
          (pC[a_y] * pC[a_yy]) / (6 * dy) - (pC[a_xy] * pC[a_yy]) / (12 * dy) + (pC[a_xz] * pC[a_y]) / (4 * dy) -
          (pC[a_xyy] * pC[a_y]) / (12 * dy) - (pC[a_xx] * pC[a_y]) / (6 * dy) - (pC[a_x] * pC[a_y]) / (2 * dy) -
          (pC[a_0] * pC[a_y]) / dy - (pC[a_xyz] * pC[a_xz]) / (16 * dy) + (pC[a_xy] * pC[a_xz]) / (8 * dy) +
          (pC[a_xxy] * pC[a_xz]) / (24 * dy) + (pC[a_xx] * pC[a_xyz]) / (24 * dy) + (pC[a_x] * pC[a_xyz]) / (8 * dy) +
          (pC[a_0] * pC[a_xyz]) / (4 * dy) - (pC[a_xy] * pC[a_xyy]) / (24 * dy) - (pC[a_xx] * pC[a_xy]) / (12 * dy) -
          (pC[a_x] * pC[a_xy]) / (4 * dy) - (pC[a_0] * pC[a_xy]) / (2 * dy) - (pC[a_xx] * pC[a_xxy]) / (36 * dy) -
          (pC[a_x] * pC[a_xxy]) / (12 * dy) - (pC[a_0] * pC[a_xxy]) / (6 * dy) + (pC[a_z] * pC[b_xz]) / (4 * dx) +
          (pC[a_xz] * pC[b_xz]) / (8 * dx) - (pC[a_xx] * pC[b_xz]) / (12 * dx) - (pC[a_x] * pC[b_xz]) / (4 * dx) -
          (pC[a_0] * pC[b_xz]) / (2 * dx) - (pC[a_y] * pC[b_xyz]) / (24 * dx) - (pC[a_xy] * pC[b_xyz]) / (48 * dx) +
          (pC[a_y] * pC[b_xy]) / (12 * dx) + (pC[a_xy] * pC[b_xy]) / (24 * dx) + (pC[a_y] * pC[b_xxy]) / (12 * dx) +
          (pC[a_xy] * pC[b_xxy]) / (24 * dx) - (pC[a_z] * pC[b_xx]) / (2 * dx) - (pC[a_xz] * pC[b_xx]) / (4 * dx) +
          (pC[a_xx] * pC[b_xx]) / (6 * dx) + (pC[a_x] * pC[b_xx]) / (2 * dx) + (pC[a_0] * pC[b_xx]) / dx -
          (pC[a_z] * pC[b_x]) / (2 * dx) - (pC[a_xz] * pC[b_x]) / (4 * dx) + (pC[a_xx] * pC[b_x]) / (6 * dx) +
          (pC[a_x] * pC[b_x]) / (2 * dx) + (pC[a_0] * pC[b_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermYComponents
 *
 */
template<typename REAL> inline
REAL JXBY_001_011(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return (pC[b_zz] * BGBZ) / dz + (pC[b_z] * BGBZ) / dz - (pC[b_xz] * BGBZ) / (2 * dz) -
          (pC[c_yzz] * BGBZ) / (6 * dy) - (pC[c_yz] * BGBZ) / (2 * dy) - (pC[c_y] * BGBZ) / dy +
          (pC[c_xyz] * BGBZ) / (4 * dy) + (pC[c_xy] * BGBZ) / (2 * dy) - (pC[a_yz] * BGBX) / (2 * dy) -
          (pC[a_y] * BGBX) / dy + (pC[a_xyz] * BGBX) / (4 * dy) + (pC[a_xy] * BGBX) / (2 * dy) -
          (pC[a_xxy] * BGBX) / (6 * dy) + (pC[b_xz] * BGBX) / (2 * dx) - (pC[b_xx] * BGBX) / dx +
          (pC[b_x] * BGBX) / dx + (pC[b_zz] * pC[c_zz]) / (6 * dz) + (pC[b_z] * pC[c_zz]) / (6 * dz) -
          (pC[b_xz] * pC[c_zz]) / (12 * dz) + (pC[b_zz] * pC[c_z]) / (2 * dz) + (pC[b_z] * pC[c_z]) / (2 * dz) -
          (pC[b_xz] * pC[c_z]) / (4 * dz) + (pC[b_yzz] * pC[c_yz]) / (24 * dz) + (pC[b_yz] * pC[c_yz]) / (24 * dz) -
          (pC[b_xyz] * pC[c_yz]) / (48 * dz) + (pC[b_yzz] * pC[c_y]) / (12 * dz) + (pC[b_yz] * pC[c_y]) / (12 * dz) -
          (pC[b_xyz] * pC[c_y]) / (24 * dz) - (pC[b_zz] * pC[c_xz]) / (4 * dz) - (pC[b_z] * pC[c_xz]) / (4 * dz) +
          (pC[b_xz] * pC[c_xz]) / (8 * dz) - (pC[b_zz] * pC[c_x]) / (2 * dz) - (pC[b_z] * pC[c_x]) / (2 * dz) +
          (pC[b_xz] * pC[c_x]) / (4 * dz) + (pC[b_zz] * pC[c_0]) / dz + (pC[b_z] * pC[c_0]) / dz -
          (pC[b_xz] * pC[c_0]) / (2 * dz) - (pC[c_yzz] * pC[c_zz]) / (36 * dy) - (pC[c_yz] * pC[c_zz]) / (12 * dy) -
          (pC[c_y] * pC[c_zz]) / (6 * dy) + (pC[c_xyz] * pC[c_zz]) / (24 * dy) + (pC[c_xy] * pC[c_zz]) / (12 * dy) -
          (pC[c_yzz] * pC[c_z]) / (12 * dy) - (pC[c_yz] * pC[c_z]) / (4 * dy) - (pC[c_y] * pC[c_z]) / (2 * dy) +
          (pC[c_xyz] * pC[c_z]) / (8 * dy) + (pC[c_xy] * pC[c_z]) / (4 * dy) + (pC[c_xz] * pC[c_yzz]) / (24 * dy) +
          (pC[c_x] * pC[c_yzz]) / (12 * dy) - (pC[c_0] * pC[c_yzz]) / (6 * dy) - (pC[c_yyz] * pC[c_yz]) / (24 * dy) -
          (pC[c_yy] * pC[c_yz]) / (12 * dy) + (pC[c_xz] * pC[c_yz]) / (8 * dy) + (pC[c_x] * pC[c_yz]) / (4 * dy) -
          (pC[c_0] * pC[c_yz]) / (2 * dy) - (pC[c_y] * pC[c_yyz]) / (12 * dy) - (pC[c_y] * pC[c_yy]) / (6 * dy) +
          (pC[c_xz] * pC[c_y]) / (4 * dy) + (pC[c_x] * pC[c_y]) / (2 * dy) - (pC[c_0] * pC[c_y]) / dy -
          (pC[c_xyz] * pC[c_xz]) / (16 * dy) - (pC[c_xy] * pC[c_xz]) / (8 * dy) - (pC[c_x] * pC[c_xyz]) / (8 * dy) +
          (pC[c_0] * pC[c_xyz]) / (4 * dy) - (pC[c_x] * pC[c_xy]) / (4 * dy) + (pC[c_0] * pC[c_xy]) / (2 * dy) -
          (pC[a_yz] * pC[a_z]) / (4 * dy) - (pC[a_y] * pC[a_z]) / (2 * dy) + (pC[a_xyz] * pC[a_z]) / (8 * dy) +
          (pC[a_xy] * pC[a_z]) / (4 * dy) - (pC[a_xxy] * pC[a_z]) / (12 * dy) + (pC[a_xz] * pC[a_yz]) / (8 * dy) -
          (pC[a_xx] * pC[a_yz]) / (12 * dy) + (pC[a_x] * pC[a_yz]) / (4 * dy) - (pC[a_0] * pC[a_yz]) / (2 * dy) -
          (pC[a_y] * pC[a_yy]) / (6 * dy) + (pC[a_xy] * pC[a_yy]) / (12 * dy) + (pC[a_xz] * pC[a_y]) / (4 * dy) +
          (pC[a_xyy] * pC[a_y]) / (12 * dy) - (pC[a_xx] * pC[a_y]) / (6 * dy) + (pC[a_x] * pC[a_y]) / (2 * dy) -
          (pC[a_0] * pC[a_y]) / dy - (pC[a_xyz] * pC[a_xz]) / (16 * dy) - (pC[a_xy] * pC[a_xz]) / (8 * dy) +
          (pC[a_xxy] * pC[a_xz]) / (24 * dy) + (pC[a_xx] * pC[a_xyz]) / (24 * dy) - (pC[a_x] * pC[a_xyz]) / (8 * dy) +
          (pC[a_0] * pC[a_xyz]) / (4 * dy) - (pC[a_xy] * pC[a_xyy]) / (24 * dy) + (pC[a_xx] * pC[a_xy]) / (12 * dy) -
          (pC[a_x] * pC[a_xy]) / (4 * dy) + (pC[a_0] * pC[a_xy]) / (2 * dy) - (pC[a_xx] * pC[a_xxy]) / (36 * dy) +
          (pC[a_x] * pC[a_xxy]) / (12 * dy) - (pC[a_0] * pC[a_xxy]) / (6 * dy) + (pC[a_z] * pC[b_xz]) / (4 * dx) -
          (pC[a_xz] * pC[b_xz]) / (8 * dx) + (pC[a_xx] * pC[b_xz]) / (12 * dx) - (pC[a_x] * pC[b_xz]) / (4 * dx) +
          (pC[a_0] * pC[b_xz]) / (2 * dx) + (pC[a_y] * pC[b_xyz]) / (24 * dx) - (pC[a_xy] * pC[b_xyz]) / (48 * dx) +
          (pC[a_y] * pC[b_xy]) / (12 * dx) - (pC[a_xy] * pC[b_xy]) / (24 * dx) - (pC[a_y] * pC[b_xxy]) / (12 * dx) +
          (pC[a_xy] * pC[b_xxy]) / (24 * dx) - (pC[a_z] * pC[b_xx]) / (2 * dx) + (pC[a_xz] * pC[b_xx]) / (4 * dx) -
          (pC[a_xx] * pC[b_xx]) / (6 * dx) + (pC[a_x] * pC[b_xx]) / (2 * dx) - (pC[a_0] * pC[b_xx]) / dx +
          (pC[a_z] * pC[b_x]) / (2 * dx) - (pC[a_xz] * pC[b_x]) / (4 * dx) + (pC[a_xx] * pC[b_x]) / (6 * dx) -
          (pC[a_x] * pC[b_x]) / (2 * dx) + (pC[a_0] * pC[b_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBZ Background Bz
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermYComponents
 *
 */
template<typename REAL> inline
REAL JXBY_101_111(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBZ,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return (pC[b_zz] * BGBZ) / dz + (pC[b_z] * BGBZ) / dz + (pC[b_xz] * BGBZ) / (2 * dz) -
          (pC[c_yzz] * BGBZ) / (6 * dy) - (pC[c_yz] * BGBZ) / (2 * dy) - (pC[c_y] * BGBZ) / dy -
          (pC[c_xyz] * BGBZ) / (4 * dy) - (pC[c_xy] * BGBZ) / (2 * dy) - (pC[a_yz] * BGBX) / (2 * dy) -
          (pC[a_y] * BGBX) / dy - (pC[a_xyz] * BGBX) / (4 * dy) - (pC[a_xy] * BGBX) / (2 * dy) -
          (pC[a_xxy] * BGBX) / (6 * dy) + (pC[b_xz] * BGBX) / (2 * dx) + (pC[b_xx] * BGBX) / dx +
          (pC[b_x] * BGBX) / dx + (pC[b_zz] * pC[c_zz]) / (6 * dz) + (pC[b_z] * pC[c_zz]) / (6 * dz) +
          (pC[b_xz] * pC[c_zz]) / (12 * dz) + (pC[b_zz] * pC[c_z]) / (2 * dz) + (pC[b_z] * pC[c_z]) / (2 * dz) +
          (pC[b_xz] * pC[c_z]) / (4 * dz) + (pC[b_yzz] * pC[c_yz]) / (24 * dz) + (pC[b_yz] * pC[c_yz]) / (24 * dz) +
          (pC[b_xyz] * pC[c_yz]) / (48 * dz) + (pC[b_yzz] * pC[c_y]) / (12 * dz) + (pC[b_yz] * pC[c_y]) / (12 * dz) +
          (pC[b_xyz] * pC[c_y]) / (24 * dz) + (pC[b_zz] * pC[c_xz]) / (4 * dz) + (pC[b_z] * pC[c_xz]) / (4 * dz) +
          (pC[b_xz] * pC[c_xz]) / (8 * dz) + (pC[b_zz] * pC[c_x]) / (2 * dz) + (pC[b_z] * pC[c_x]) / (2 * dz) +
          (pC[b_xz] * pC[c_x]) / (4 * dz) + (pC[b_zz] * pC[c_0]) / dz + (pC[b_z] * pC[c_0]) / dz +
          (pC[b_xz] * pC[c_0]) / (2 * dz) - (pC[c_yzz] * pC[c_zz]) / (36 * dy) - (pC[c_yz] * pC[c_zz]) / (12 * dy) -
          (pC[c_y] * pC[c_zz]) / (6 * dy) - (pC[c_xyz] * pC[c_zz]) / (24 * dy) - (pC[c_xy] * pC[c_zz]) / (12 * dy) -
          (pC[c_yzz] * pC[c_z]) / (12 * dy) - (pC[c_yz] * pC[c_z]) / (4 * dy) - (pC[c_y] * pC[c_z]) / (2 * dy) -
          (pC[c_xyz] * pC[c_z]) / (8 * dy) - (pC[c_xy] * pC[c_z]) / (4 * dy) - (pC[c_xz] * pC[c_yzz]) / (24 * dy) -
          (pC[c_x] * pC[c_yzz]) / (12 * dy) - (pC[c_0] * pC[c_yzz]) / (6 * dy) - (pC[c_yyz] * pC[c_yz]) / (24 * dy) -
          (pC[c_yy] * pC[c_yz]) / (12 * dy) - (pC[c_xz] * pC[c_yz]) / (8 * dy) - (pC[c_x] * pC[c_yz]) / (4 * dy) -
          (pC[c_0] * pC[c_yz]) / (2 * dy) - (pC[c_y] * pC[c_yyz]) / (12 * dy) - (pC[c_y] * pC[c_yy]) / (6 * dy) -
          (pC[c_xz] * pC[c_y]) / (4 * dy) - (pC[c_x] * pC[c_y]) / (2 * dy) - (pC[c_0] * pC[c_y]) / dy -
          (pC[c_xyz] * pC[c_xz]) / (16 * dy) - (pC[c_xy] * pC[c_xz]) / (8 * dy) - (pC[c_x] * pC[c_xyz]) / (8 * dy) -
          (pC[c_0] * pC[c_xyz]) / (4 * dy) - (pC[c_x] * pC[c_xy]) / (4 * dy) - (pC[c_0] * pC[c_xy]) / (2 * dy) -
          (pC[a_yz] * pC[a_z]) / (4 * dy) - (pC[a_y] * pC[a_z]) / (2 * dy) - (pC[a_xyz] * pC[a_z]) / (8 * dy) -
          (pC[a_xy] * pC[a_z]) / (4 * dy) - (pC[a_xxy] * pC[a_z]) / (12 * dy) - (pC[a_xz] * pC[a_yz]) / (8 * dy) -
          (pC[a_xx] * pC[a_yz]) / (12 * dy) - (pC[a_x] * pC[a_yz]) / (4 * dy) - (pC[a_0] * pC[a_yz]) / (2 * dy) -
          (pC[a_y] * pC[a_yy]) / (6 * dy) - (pC[a_xy] * pC[a_yy]) / (12 * dy) - (pC[a_xz] * pC[a_y]) / (4 * dy) -
          (pC[a_xyy] * pC[a_y]) / (12 * dy) - (pC[a_xx] * pC[a_y]) / (6 * dy) - (pC[a_x] * pC[a_y]) / (2 * dy) -
          (pC[a_0] * pC[a_y]) / dy - (pC[a_xyz] * pC[a_xz]) / (16 * dy) - (pC[a_xy] * pC[a_xz]) / (8 * dy) -
          (pC[a_xxy] * pC[a_xz]) / (24 * dy) - (pC[a_xx] * pC[a_xyz]) / (24 * dy) - (pC[a_x] * pC[a_xyz]) / (8 * dy) -
          (pC[a_0] * pC[a_xyz]) / (4 * dy) - (pC[a_xy] * pC[a_xyy]) / (24 * dy) - (pC[a_xx] * pC[a_xy]) / (12 * dy) -
          (pC[a_x] * pC[a_xy]) / (4 * dy) - (pC[a_0] * pC[a_xy]) / (2 * dy) - (pC[a_xx] * pC[a_xxy]) / (36 * dy) -
          (pC[a_x] * pC[a_xxy]) / (12 * dy) - (pC[a_0] * pC[a_xxy]) / (6 * dy) + (pC[a_z] * pC[b_xz]) / (4 * dx) +
          (pC[a_xz] * pC[b_xz]) / (8 * dx) + (pC[a_xx] * pC[b_xz]) / (12 * dx) + (pC[a_x] * pC[b_xz]) / (4 * dx) +
          (pC[a_0] * pC[b_xz]) / (2 * dx) + (pC[a_y] * pC[b_xyz]) / (24 * dx) + (pC[a_xy] * pC[b_xyz]) / (48 * dx) +
          (pC[a_y] * pC[b_xy]) / (12 * dx) + (pC[a_xy] * pC[b_xy]) / (24 * dx) + (pC[a_y] * pC[b_xxy]) / (12 * dx) +
          (pC[a_xy] * pC[b_xxy]) / (24 * dx) + (pC[a_z] * pC[b_xx]) / (2 * dx) + (pC[a_xz] * pC[b_xx]) / (4 * dx) +
          (pC[a_xx] * pC[b_xx]) / (6 * dx) + (pC[a_x] * pC[b_xx]) / (2 * dx) + (pC[a_0] * pC[b_xx]) / dx +
          (pC[a_z] * pC[b_x]) / (2 * dx) + (pC[a_xz] * pC[b_x]) / (4 * dx) + (pC[a_xx] * pC[b_x]) / (6 * dx) +
          (pC[a_x] * pC[b_x]) / (2 * dx) + (pC[a_0] * pC[b_x]) / dx;
}

// Z
/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBY Background By
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermZComponents
 *
 */
template<typename REAL> inline
REAL JXBZ_000_001(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBY,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[b_z] * BGBY) / dz + (pC[b_yz] * BGBY) / (2 * dz) - (pC[b_yyz] * BGBY) / (6 * dz) +
          (pC[b_xz] * BGBY) / (2 * dz) - (pC[b_xyz] * BGBY) / (4 * dz) - (pC[c_yy] * BGBY) / dy +
          (pC[c_y] * BGBY) / dy - (pC[c_xy] * BGBY) / (2 * dy) - (pC[a_z] * BGBX) / dz + (pC[a_yz] * BGBX) / (2 * dz) +
          (pC[a_xz] * BGBX) / (2 * dz) - (pC[a_xyz] * BGBX) / (4 * dz) - (pC[a_xxz] * BGBX) / (6 * dz) -
          (pC[c_xy] * BGBX) / (2 * dx) - (pC[c_xx] * BGBX) / dx + (pC[c_x] * BGBX) / dx -
          (pC[b_z] * pC[b_zz]) / (6 * dz) + (pC[b_yz] * pC[b_zz]) / (12 * dz) + (pC[b_yzz] * pC[b_z]) / (12 * dz) -
          (pC[b_yy] * pC[b_z]) / (6 * dz) + (pC[b_y] * pC[b_z]) / (2 * dz) - (pC[b_xy] * pC[b_z]) / (4 * dz) +
          (pC[b_x] * pC[b_z]) / (2 * dz) - (pC[b_0] * pC[b_z]) / dz - (pC[b_yz] * pC[b_yzz]) / (24 * dz) +
          (pC[b_yy] * pC[b_yz]) / (12 * dz) - (pC[b_y] * pC[b_yz]) / (4 * dz) + (pC[b_xy] * pC[b_yz]) / (8 * dz) -
          (pC[b_x] * pC[b_yz]) / (4 * dz) + (pC[b_0] * pC[b_yz]) / (2 * dz) - (pC[b_yy] * pC[b_yyz]) / (36 * dz) +
          (pC[b_y] * pC[b_yyz]) / (12 * dz) - (pC[b_xy] * pC[b_yyz]) / (24 * dz) + (pC[b_x] * pC[b_yyz]) / (12 * dz) -
          (pC[b_0] * pC[b_yyz]) / (6 * dz) + (pC[b_xz] * pC[b_yy]) / (12 * dz) - (pC[b_xyz] * pC[b_yy]) / (24 * dz) -
          (pC[b_xz] * pC[b_y]) / (4 * dz) + (pC[b_xyz] * pC[b_y]) / (8 * dz) + (pC[b_xy] * pC[b_xz]) / (8 * dz) -
          (pC[b_x] * pC[b_xz]) / (4 * dz) + (pC[b_0] * pC[b_xz]) / (2 * dz) - (pC[b_xy] * pC[b_xyz]) / (16 * dz) +
          (pC[b_x] * pC[b_xyz]) / (8 * dz) - (pC[b_0] * pC[b_xyz]) / (4 * dz) - (pC[a_z] * pC[a_zz]) / (6 * dz) +
          (pC[a_xz] * pC[a_zz]) / (12 * dz) + (pC[a_y] * pC[a_z]) / (2 * dz) + (pC[a_xzz] * pC[a_z]) / (12 * dz) -
          (pC[a_xy] * pC[a_z]) / (4 * dz) - (pC[a_xx] * pC[a_z]) / (6 * dz) + (pC[a_x] * pC[a_z]) / (2 * dz) -
          (pC[a_0] * pC[a_z]) / dz - (pC[a_y] * pC[a_yz]) / (4 * dz) + (pC[a_xy] * pC[a_yz]) / (8 * dz) +
          (pC[a_xx] * pC[a_yz]) / (12 * dz) - (pC[a_x] * pC[a_yz]) / (4 * dz) + (pC[a_0] * pC[a_yz]) / (2 * dz) -
          (pC[a_xz] * pC[a_y]) / (4 * dz) + (pC[a_xyz] * pC[a_y]) / (8 * dz) + (pC[a_xxz] * pC[a_y]) / (12 * dz) -
          (pC[a_xz] * pC[a_xzz]) / (24 * dz) + (pC[a_xy] * pC[a_xz]) / (8 * dz) + (pC[a_xx] * pC[a_xz]) / (12 * dz) -
          (pC[a_x] * pC[a_xz]) / (4 * dz) + (pC[a_0] * pC[a_xz]) / (2 * dz) - (pC[a_xy] * pC[a_xyz]) / (16 * dz) -
          (pC[a_xx] * pC[a_xyz]) / (24 * dz) + (pC[a_x] * pC[a_xyz]) / (8 * dz) - (pC[a_0] * pC[a_xyz]) / (4 * dz) -
          (pC[a_xxz] * pC[a_xy]) / (24 * dz) - (pC[a_xx] * pC[a_xxz]) / (36 * dz) + (pC[a_x] * pC[a_xxz]) / (12 * dz) -
          (pC[a_0] * pC[a_xxz]) / (6 * dz) + (pC[b_z] * pC[c_yz]) / (12 * dy) - (pC[b_yz] * pC[c_yz]) / (24 * dy) -
          (pC[b_z] * pC[c_yyz]) / (12 * dy) + (pC[b_yz] * pC[c_yyz]) / (24 * dy) - (pC[b_yy] * pC[c_yy]) / (6 * dy) +
          (pC[b_y] * pC[c_yy]) / (2 * dy) - (pC[b_xy] * pC[c_yy]) / (4 * dy) + (pC[b_x] * pC[c_yy]) / (2 * dy) -
          (pC[b_0] * pC[c_yy]) / dy + (pC[b_yy] * pC[c_y]) / (6 * dy) - (pC[b_y] * pC[c_y]) / (2 * dy) +
          (pC[b_xy] * pC[c_y]) / (4 * dy) - (pC[b_x] * pC[c_y]) / (2 * dy) + (pC[b_0] * pC[c_y]) / dy -
          (pC[b_z] * pC[c_xyz]) / (24 * dy) + (pC[b_yz] * pC[c_xyz]) / (48 * dy) - (pC[b_yy] * pC[c_xy]) / (12 * dy) +
          (pC[b_y] * pC[c_xy]) / (4 * dy) - (pC[b_xy] * pC[c_xy]) / (8 * dy) + (pC[b_x] * pC[c_xy]) / (4 * dy) -
          (pC[b_0] * pC[c_xy]) / (2 * dy) + (pC[a_z] * pC[c_xz]) / (12 * dx) - (pC[a_xz] * pC[c_xz]) / (24 * dx) -
          (pC[a_z] * pC[c_xyz]) / (24 * dx) + (pC[a_xz] * pC[c_xyz]) / (48 * dx) + (pC[a_y] * pC[c_xy]) / (4 * dx) -
          (pC[a_xy] * pC[c_xy]) / (8 * dx) - (pC[a_xx] * pC[c_xy]) / (12 * dx) + (pC[a_x] * pC[c_xy]) / (4 * dx) -
          (pC[a_0] * pC[c_xy]) / (2 * dx) - (pC[a_z] * pC[c_xxz]) / (12 * dx) + (pC[a_xz] * pC[c_xxz]) / (24 * dx) +
          (pC[a_y] * pC[c_xx]) / (2 * dx) - (pC[a_xy] * pC[c_xx]) / (4 * dx) - (pC[a_xx] * pC[c_xx]) / (6 * dx) +
          (pC[a_x] * pC[c_xx]) / (2 * dx) - (pC[a_0] * pC[c_xx]) / dx - (pC[a_y] * pC[c_x]) / (2 * dx) +
          (pC[a_xy] * pC[c_x]) / (4 * dx) + (pC[a_xx] * pC[c_x]) / (6 * dx) - (pC[a_x] * pC[c_x]) / (2 * dx) +
          (pC[a_0] * pC[c_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBY Background By
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermZComponents
 *
 */
template<typename REAL> inline
REAL JXBZ_100_101(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBY,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[b_z] * BGBY) / dz + (pC[b_yz] * BGBY) / (2 * dz) - (pC[b_yyz] * BGBY) / (6 * dz) -
          (pC[b_xz] * BGBY) / (2 * dz) + (pC[b_xyz] * BGBY) / (4 * dz) - (pC[c_yy] * BGBY) / dy +
          (pC[c_y] * BGBY) / dy + (pC[c_xy] * BGBY) / (2 * dy) - (pC[a_z] * BGBX) / dz + (pC[a_yz] * BGBX) / (2 * dz) -
          (pC[a_xz] * BGBX) / (2 * dz) + (pC[a_xyz] * BGBX) / (4 * dz) - (pC[a_xxz] * BGBX) / (6 * dz) -
          (pC[c_xy] * BGBX) / (2 * dx) + (pC[c_xx] * BGBX) / dx + (pC[c_x] * BGBX) / dx -
          (pC[b_z] * pC[b_zz]) / (6 * dz) + (pC[b_yz] * pC[b_zz]) / (12 * dz) + (pC[b_yzz] * pC[b_z]) / (12 * dz) -
          (pC[b_yy] * pC[b_z]) / (6 * dz) + (pC[b_y] * pC[b_z]) / (2 * dz) + (pC[b_xy] * pC[b_z]) / (4 * dz) -
          (pC[b_x] * pC[b_z]) / (2 * dz) - (pC[b_0] * pC[b_z]) / dz - (pC[b_yz] * pC[b_yzz]) / (24 * dz) +
          (pC[b_yy] * pC[b_yz]) / (12 * dz) - (pC[b_y] * pC[b_yz]) / (4 * dz) - (pC[b_xy] * pC[b_yz]) / (8 * dz) +
          (pC[b_x] * pC[b_yz]) / (4 * dz) + (pC[b_0] * pC[b_yz]) / (2 * dz) - (pC[b_yy] * pC[b_yyz]) / (36 * dz) +
          (pC[b_y] * pC[b_yyz]) / (12 * dz) + (pC[b_xy] * pC[b_yyz]) / (24 * dz) - (pC[b_x] * pC[b_yyz]) / (12 * dz) -
          (pC[b_0] * pC[b_yyz]) / (6 * dz) - (pC[b_xz] * pC[b_yy]) / (12 * dz) + (pC[b_xyz] * pC[b_yy]) / (24 * dz) +
          (pC[b_xz] * pC[b_y]) / (4 * dz) - (pC[b_xyz] * pC[b_y]) / (8 * dz) + (pC[b_xy] * pC[b_xz]) / (8 * dz) -
          (pC[b_x] * pC[b_xz]) / (4 * dz) - (pC[b_0] * pC[b_xz]) / (2 * dz) - (pC[b_xy] * pC[b_xyz]) / (16 * dz) +
          (pC[b_x] * pC[b_xyz]) / (8 * dz) + (pC[b_0] * pC[b_xyz]) / (4 * dz) - (pC[a_z] * pC[a_zz]) / (6 * dz) -
          (pC[a_xz] * pC[a_zz]) / (12 * dz) + (pC[a_y] * pC[a_z]) / (2 * dz) - (pC[a_xzz] * pC[a_z]) / (12 * dz) +
          (pC[a_xy] * pC[a_z]) / (4 * dz) - (pC[a_xx] * pC[a_z]) / (6 * dz) - (pC[a_x] * pC[a_z]) / (2 * dz) -
          (pC[a_0] * pC[a_z]) / dz - (pC[a_y] * pC[a_yz]) / (4 * dz) - (pC[a_xy] * pC[a_yz]) / (8 * dz) +
          (pC[a_xx] * pC[a_yz]) / (12 * dz) + (pC[a_x] * pC[a_yz]) / (4 * dz) + (pC[a_0] * pC[a_yz]) / (2 * dz) +
          (pC[a_xz] * pC[a_y]) / (4 * dz) - (pC[a_xyz] * pC[a_y]) / (8 * dz) + (pC[a_xxz] * pC[a_y]) / (12 * dz) -
          (pC[a_xz] * pC[a_xzz]) / (24 * dz) + (pC[a_xy] * pC[a_xz]) / (8 * dz) - (pC[a_xx] * pC[a_xz]) / (12 * dz) -
          (pC[a_x] * pC[a_xz]) / (4 * dz) - (pC[a_0] * pC[a_xz]) / (2 * dz) - (pC[a_xy] * pC[a_xyz]) / (16 * dz) +
          (pC[a_xx] * pC[a_xyz]) / (24 * dz) + (pC[a_x] * pC[a_xyz]) / (8 * dz) + (pC[a_0] * pC[a_xyz]) / (4 * dz) +
          (pC[a_xxz] * pC[a_xy]) / (24 * dz) - (pC[a_xx] * pC[a_xxz]) / (36 * dz) - (pC[a_x] * pC[a_xxz]) / (12 * dz) -
          (pC[a_0] * pC[a_xxz]) / (6 * dz) + (pC[b_z] * pC[c_yz]) / (12 * dy) - (pC[b_yz] * pC[c_yz]) / (24 * dy) -
          (pC[b_z] * pC[c_yyz]) / (12 * dy) + (pC[b_yz] * pC[c_yyz]) / (24 * dy) - (pC[b_yy] * pC[c_yy]) / (6 * dy) +
          (pC[b_y] * pC[c_yy]) / (2 * dy) + (pC[b_xy] * pC[c_yy]) / (4 * dy) - (pC[b_x] * pC[c_yy]) / (2 * dy) -
          (pC[b_0] * pC[c_yy]) / dy + (pC[b_yy] * pC[c_y]) / (6 * dy) - (pC[b_y] * pC[c_y]) / (2 * dy) -
          (pC[b_xy] * pC[c_y]) / (4 * dy) + (pC[b_x] * pC[c_y]) / (2 * dy) + (pC[b_0] * pC[c_y]) / dy +
          (pC[b_z] * pC[c_xyz]) / (24 * dy) - (pC[b_yz] * pC[c_xyz]) / (48 * dy) + (pC[b_yy] * pC[c_xy]) / (12 * dy) -
          (pC[b_y] * pC[c_xy]) / (4 * dy) - (pC[b_xy] * pC[c_xy]) / (8 * dy) + (pC[b_x] * pC[c_xy]) / (4 * dy) +
          (pC[b_0] * pC[c_xy]) / (2 * dy) + (pC[a_z] * pC[c_xz]) / (12 * dx) + (pC[a_xz] * pC[c_xz]) / (24 * dx) -
          (pC[a_z] * pC[c_xyz]) / (24 * dx) - (pC[a_xz] * pC[c_xyz]) / (48 * dx) + (pC[a_y] * pC[c_xy]) / (4 * dx) +
          (pC[a_xy] * pC[c_xy]) / (8 * dx) - (pC[a_xx] * pC[c_xy]) / (12 * dx) - (pC[a_x] * pC[c_xy]) / (4 * dx) -
          (pC[a_0] * pC[c_xy]) / (2 * dx) + (pC[a_z] * pC[c_xxz]) / (12 * dx) + (pC[a_xz] * pC[c_xxz]) / (24 * dx) -
          (pC[a_y] * pC[c_xx]) / (2 * dx) - (pC[a_xy] * pC[c_xx]) / (4 * dx) + (pC[a_xx] * pC[c_xx]) / (6 * dx) +
          (pC[a_x] * pC[c_xx]) / (2 * dx) + (pC[a_0] * pC[c_xx]) / dx - (pC[a_y] * pC[c_x]) / (2 * dx) -
          (pC[a_xy] * pC[c_x]) / (4 * dx) + (pC[a_xx] * pC[c_x]) / (6 * dx) + (pC[a_x] * pC[c_x]) / (2 * dx) +
          (pC[a_0] * pC[c_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBY Background By
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermZComponents
 *
 */
template<typename REAL> inline
REAL JXBZ_010_011(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBY,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[b_z] * BGBY) / dz - (pC[b_yz] * BGBY) / (2 * dz) - (pC[b_yyz] * BGBY) / (6 * dz) +
          (pC[b_xz] * BGBY) / (2 * dz) + (pC[b_xyz] * BGBY) / (4 * dz) + (pC[c_yy] * BGBY) / dy +
          (pC[c_y] * BGBY) / dy - (pC[c_xy] * BGBY) / (2 * dy) - (pC[a_z] * BGBX) / dz - (pC[a_yz] * BGBX) / (2 * dz) +
          (pC[a_xz] * BGBX) / (2 * dz) + (pC[a_xyz] * BGBX) / (4 * dz) - (pC[a_xxz] * BGBX) / (6 * dz) +
          (pC[c_xy] * BGBX) / (2 * dx) - (pC[c_xx] * BGBX) / dx + (pC[c_x] * BGBX) / dx -
          (pC[b_z] * pC[b_zz]) / (6 * dz) - (pC[b_yz] * pC[b_zz]) / (12 * dz) - (pC[b_yzz] * pC[b_z]) / (12 * dz) -
          (pC[b_yy] * pC[b_z]) / (6 * dz) - (pC[b_y] * pC[b_z]) / (2 * dz) + (pC[b_xy] * pC[b_z]) / (4 * dz) +
          (pC[b_x] * pC[b_z]) / (2 * dz) - (pC[b_0] * pC[b_z]) / dz - (pC[b_yz] * pC[b_yzz]) / (24 * dz) -
          (pC[b_yy] * pC[b_yz]) / (12 * dz) - (pC[b_y] * pC[b_yz]) / (4 * dz) + (pC[b_xy] * pC[b_yz]) / (8 * dz) +
          (pC[b_x] * pC[b_yz]) / (4 * dz) - (pC[b_0] * pC[b_yz]) / (2 * dz) - (pC[b_yy] * pC[b_yyz]) / (36 * dz) -
          (pC[b_y] * pC[b_yyz]) / (12 * dz) + (pC[b_xy] * pC[b_yyz]) / (24 * dz) + (pC[b_x] * pC[b_yyz]) / (12 * dz) -
          (pC[b_0] * pC[b_yyz]) / (6 * dz) + (pC[b_xz] * pC[b_yy]) / (12 * dz) + (pC[b_xyz] * pC[b_yy]) / (24 * dz) +
          (pC[b_xz] * pC[b_y]) / (4 * dz) + (pC[b_xyz] * pC[b_y]) / (8 * dz) - (pC[b_xy] * pC[b_xz]) / (8 * dz) -
          (pC[b_x] * pC[b_xz]) / (4 * dz) + (pC[b_0] * pC[b_xz]) / (2 * dz) - (pC[b_xy] * pC[b_xyz]) / (16 * dz) -
          (pC[b_x] * pC[b_xyz]) / (8 * dz) + (pC[b_0] * pC[b_xyz]) / (4 * dz) - (pC[a_z] * pC[a_zz]) / (6 * dz) +
          (pC[a_xz] * pC[a_zz]) / (12 * dz) - (pC[a_y] * pC[a_z]) / (2 * dz) + (pC[a_xzz] * pC[a_z]) / (12 * dz) +
          (pC[a_xy] * pC[a_z]) / (4 * dz) - (pC[a_xx] * pC[a_z]) / (6 * dz) + (pC[a_x] * pC[a_z]) / (2 * dz) -
          (pC[a_0] * pC[a_z]) / dz - (pC[a_y] * pC[a_yz]) / (4 * dz) + (pC[a_xy] * pC[a_yz]) / (8 * dz) -
          (pC[a_xx] * pC[a_yz]) / (12 * dz) + (pC[a_x] * pC[a_yz]) / (4 * dz) - (pC[a_0] * pC[a_yz]) / (2 * dz) +
          (pC[a_xz] * pC[a_y]) / (4 * dz) + (pC[a_xyz] * pC[a_y]) / (8 * dz) - (pC[a_xxz] * pC[a_y]) / (12 * dz) -
          (pC[a_xz] * pC[a_xzz]) / (24 * dz) - (pC[a_xy] * pC[a_xz]) / (8 * dz) + (pC[a_xx] * pC[a_xz]) / (12 * dz) -
          (pC[a_x] * pC[a_xz]) / (4 * dz) + (pC[a_0] * pC[a_xz]) / (2 * dz) - (pC[a_xy] * pC[a_xyz]) / (16 * dz) +
          (pC[a_xx] * pC[a_xyz]) / (24 * dz) - (pC[a_x] * pC[a_xyz]) / (8 * dz) + (pC[a_0] * pC[a_xyz]) / (4 * dz) +
          (pC[a_xxz] * pC[a_xy]) / (24 * dz) - (pC[a_xx] * pC[a_xxz]) / (36 * dz) + (pC[a_x] * pC[a_xxz]) / (12 * dz) -
          (pC[a_0] * pC[a_xxz]) / (6 * dz) + (pC[b_z] * pC[c_yz]) / (12 * dy) + (pC[b_yz] * pC[c_yz]) / (24 * dy) +
          (pC[b_z] * pC[c_yyz]) / (12 * dy) + (pC[b_yz] * pC[c_yyz]) / (24 * dy) + (pC[b_yy] * pC[c_yy]) / (6 * dy) +
          (pC[b_y] * pC[c_yy]) / (2 * dy) - (pC[b_xy] * pC[c_yy]) / (4 * dy) - (pC[b_x] * pC[c_yy]) / (2 * dy) +
          (pC[b_0] * pC[c_yy]) / dy + (pC[b_yy] * pC[c_y]) / (6 * dy) + (pC[b_y] * pC[c_y]) / (2 * dy) -
          (pC[b_xy] * pC[c_y]) / (4 * dy) - (pC[b_x] * pC[c_y]) / (2 * dy) + (pC[b_0] * pC[c_y]) / dy -
          (pC[b_z] * pC[c_xyz]) / (24 * dy) - (pC[b_yz] * pC[c_xyz]) / (48 * dy) - (pC[b_yy] * pC[c_xy]) / (12 * dy) -
          (pC[b_y] * pC[c_xy]) / (4 * dy) + (pC[b_xy] * pC[c_xy]) / (8 * dy) + (pC[b_x] * pC[c_xy]) / (4 * dy) -
          (pC[b_0] * pC[c_xy]) / (2 * dy) + (pC[a_z] * pC[c_xz]) / (12 * dx) - (pC[a_xz] * pC[c_xz]) / (24 * dx) +
          (pC[a_z] * pC[c_xyz]) / (24 * dx) - (pC[a_xz] * pC[c_xyz]) / (48 * dx) + (pC[a_y] * pC[c_xy]) / (4 * dx) -
          (pC[a_xy] * pC[c_xy]) / (8 * dx) + (pC[a_xx] * pC[c_xy]) / (12 * dx) - (pC[a_x] * pC[c_xy]) / (4 * dx) +
          (pC[a_0] * pC[c_xy]) / (2 * dx) - (pC[a_z] * pC[c_xxz]) / (12 * dx) + (pC[a_xz] * pC[c_xxz]) / (24 * dx) -
          (pC[a_y] * pC[c_xx]) / (2 * dx) + (pC[a_xy] * pC[c_xx]) / (4 * dx) - (pC[a_xx] * pC[c_xx]) / (6 * dx) +
          (pC[a_x] * pC[c_xx]) / (2 * dx) - (pC[a_0] * pC[c_xx]) / dx + (pC[a_y] * pC[c_x]) / (2 * dx) -
          (pC[a_xy] * pC[c_x]) / (4 * dx) + (pC[a_xx] * pC[c_x]) / (6 * dx) - (pC[a_x] * pC[c_x]) / (2 * dx) +
          (pC[a_0] * pC[c_x]) / dx;
}

/*! \brief Low-level Hall component computation
 *
 * Hall term computation following Balsara reconstruction, edge-averaged.
 *
 * \param pC Reconstruction coefficients
 * \param BGBX Background Bx
 * \param BGBY Background By
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateEdgeHallTermZComponents
 *
 */
template<typename REAL> inline
REAL JXBZ_110_111(
   const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC,
   creal BGBX,
   creal BGBY,
   const std::array<Real, 3>& gridSpacing
) {
   using namespace Rec;
   const auto dx = gridSpacing[0];
   const auto dy = gridSpacing[1];
   const auto dz = gridSpacing[2];
   return -(pC[b_z] * BGBY) / dz - (pC[b_yz] * BGBY) / (2 * dz) - (pC[b_yyz] * BGBY) / (6 * dz) -
          (pC[b_xz] * BGBY) / (2 * dz) - (pC[b_xyz] * BGBY) / (4 * dz) + (pC[c_yy] * BGBY) / dy +
          (pC[c_y] * BGBY) / dy + (pC[c_xy] * BGBY) / (2 * dy) - (pC[a_z] * BGBX) / dz - (pC[a_yz] * BGBX) / (2 * dz) -
          (pC[a_xz] * BGBX) / (2 * dz) - (pC[a_xyz] * BGBX) / (4 * dz) - (pC[a_xxz] * BGBX) / (6 * dz) +
          (pC[c_xy] * BGBX) / (2 * dx) + (pC[c_xx] * BGBX) / dx + (pC[c_x] * BGBX) / dx -
          (pC[b_z] * pC[b_zz]) / (6 * dz) - (pC[b_yz] * pC[b_zz]) / (12 * dz) - (pC[b_yzz] * pC[b_z]) / (12 * dz) -
          (pC[b_yy] * pC[b_z]) / (6 * dz) - (pC[b_y] * pC[b_z]) / (2 * dz) - (pC[b_xy] * pC[b_z]) / (4 * dz) -
          (pC[b_x] * pC[b_z]) / (2 * dz) - (pC[b_0] * pC[b_z]) / dz - (pC[b_yz] * pC[b_yzz]) / (24 * dz) -
          (pC[b_yy] * pC[b_yz]) / (12 * dz) - (pC[b_y] * pC[b_yz]) / (4 * dz) - (pC[b_xy] * pC[b_yz]) / (8 * dz) -
          (pC[b_x] * pC[b_yz]) / (4 * dz) - (pC[b_0] * pC[b_yz]) / (2 * dz) - (pC[b_yy] * pC[b_yyz]) / (36 * dz) -
          (pC[b_y] * pC[b_yyz]) / (12 * dz) - (pC[b_xy] * pC[b_yyz]) / (24 * dz) - (pC[b_x] * pC[b_yyz]) / (12 * dz) -
          (pC[b_0] * pC[b_yyz]) / (6 * dz) - (pC[b_xz] * pC[b_yy]) / (12 * dz) - (pC[b_xyz] * pC[b_yy]) / (24 * dz) -
          (pC[b_xz] * pC[b_y]) / (4 * dz) - (pC[b_xyz] * pC[b_y]) / (8 * dz) - (pC[b_xy] * pC[b_xz]) / (8 * dz) -
          (pC[b_x] * pC[b_xz]) / (4 * dz) - (pC[b_0] * pC[b_xz]) / (2 * dz) - (pC[b_xy] * pC[b_xyz]) / (16 * dz) -
          (pC[b_x] * pC[b_xyz]) / (8 * dz) - (pC[b_0] * pC[b_xyz]) / (4 * dz) - (pC[a_z] * pC[a_zz]) / (6 * dz) -
          (pC[a_xz] * pC[a_zz]) / (12 * dz) - (pC[a_y] * pC[a_z]) / (2 * dz) - (pC[a_xzz] * pC[a_z]) / (12 * dz) -
          (pC[a_xy] * pC[a_z]) / (4 * dz) - (pC[a_xx] * pC[a_z]) / (6 * dz) - (pC[a_x] * pC[a_z]) / (2 * dz) -
          (pC[a_0] * pC[a_z]) / dz - (pC[a_y] * pC[a_yz]) / (4 * dz) - (pC[a_xy] * pC[a_yz]) / (8 * dz) -
          (pC[a_xx] * pC[a_yz]) / (12 * dz) - (pC[a_x] * pC[a_yz]) / (4 * dz) - (pC[a_0] * pC[a_yz]) / (2 * dz) -
          (pC[a_xz] * pC[a_y]) / (4 * dz) - (pC[a_xyz] * pC[a_y]) / (8 * dz) - (pC[a_xxz] * pC[a_y]) / (12 * dz) -
          (pC[a_xz] * pC[a_xzz]) / (24 * dz) - (pC[a_xy] * pC[a_xz]) / (8 * dz) - (pC[a_xx] * pC[a_xz]) / (12 * dz) -
          (pC[a_x] * pC[a_xz]) / (4 * dz) - (pC[a_0] * pC[a_xz]) / (2 * dz) - (pC[a_xy] * pC[a_xyz]) / (16 * dz) -
          (pC[a_xx] * pC[a_xyz]) / (24 * dz) - (pC[a_x] * pC[a_xyz]) / (8 * dz) - (pC[a_0] * pC[a_xyz]) / (4 * dz) -
          (pC[a_xxz] * pC[a_xy]) / (24 * dz) - (pC[a_xx] * pC[a_xxz]) / (36 * dz) - (pC[a_x] * pC[a_xxz]) / (12 * dz) -
          (pC[a_0] * pC[a_xxz]) / (6 * dz) + (pC[b_z] * pC[c_yz]) / (12 * dy) + (pC[b_yz] * pC[c_yz]) / (24 * dy) +
          (pC[b_z] * pC[c_yyz]) / (12 * dy) + (pC[b_yz] * pC[c_yyz]) / (24 * dy) + (pC[b_yy] * pC[c_yy]) / (6 * dy) +
          (pC[b_y] * pC[c_yy]) / (2 * dy) + (pC[b_xy] * pC[c_yy]) / (4 * dy) + (pC[b_x] * pC[c_yy]) / (2 * dy) +
          (pC[b_0] * pC[c_yy]) / dy + (pC[b_yy] * pC[c_y]) / (6 * dy) + (pC[b_y] * pC[c_y]) / (2 * dy) +
          (pC[b_xy] * pC[c_y]) / (4 * dy) + (pC[b_x] * pC[c_y]) / (2 * dy) + (pC[b_0] * pC[c_y]) / dy +
          (pC[b_z] * pC[c_xyz]) / (24 * dy) + (pC[b_yz] * pC[c_xyz]) / (48 * dy) + (pC[b_yy] * pC[c_xy]) / (12 * dy) +
          (pC[b_y] * pC[c_xy]) / (4 * dy) + (pC[b_xy] * pC[c_xy]) / (8 * dy) + (pC[b_x] * pC[c_xy]) / (4 * dy) +
          (pC[b_0] * pC[c_xy]) / (2 * dy) + (pC[a_z] * pC[c_xz]) / (12 * dx) + (pC[a_xz] * pC[c_xz]) / (24 * dx) +
          (pC[a_z] * pC[c_xyz]) / (24 * dx) + (pC[a_xz] * pC[c_xyz]) / (48 * dx) + (pC[a_y] * pC[c_xy]) / (4 * dx) +
          (pC[a_xy] * pC[c_xy]) / (8 * dx) + (pC[a_xx] * pC[c_xy]) / (12 * dx) + (pC[a_x] * pC[c_xy]) / (4 * dx) +
          (pC[a_0] * pC[c_xy]) / (2 * dx) + (pC[a_z] * pC[c_xxz]) / (12 * dx) + (pC[a_xz] * pC[c_xxz]) / (24 * dx) +
          (pC[a_y] * pC[c_xx]) / (2 * dx) + (pC[a_xy] * pC[c_xx]) / (4 * dx) + (pC[a_xx] * pC[c_xx]) / (6 * dx) +
          (pC[a_x] * pC[c_xx]) / (2 * dx) + (pC[a_0] * pC[c_xx]) / dx + (pC[a_y] * pC[c_x]) / (2 * dx) +
          (pC[a_xy] * pC[c_x]) / (4 * dx) + (pC[a_xx] * pC[c_x]) / (6 * dx) + (pC[a_x] * pC[c_x]) / (2 * dx) +
          (pC[a_0] * pC[c_x]) / dx;
}

template <typename REAL>
inline REAL JXB(fsgrids::ehall term, const std::array<REAL, Rec::N_REC_COEFFICIENTS>& pC, Real BGBX, Real BGBY,
                Real BGBZ, const std::array<Real, 3>& gridSpacing) {
   switch (term) {
   case fsgrids::EXHALL_000_100: {
      return JXBX_000_100(pC, BGBY, BGBZ, gridSpacing);
   }
   case fsgrids::EXHALL_010_110: {
      return JXBX_010_110(pC, BGBY, BGBZ, gridSpacing);
   }
   case fsgrids::EXHALL_001_101: {
      return JXBX_001_101(pC, BGBY, BGBZ, gridSpacing);
   }
   case fsgrids::EXHALL_011_111: {
      return JXBX_011_111(pC, BGBY, BGBZ, gridSpacing);
   }
   case fsgrids::EYHALL_000_010: {
      return JXBY_000_010(pC, BGBX, BGBZ, gridSpacing);
   }
   case fsgrids::EYHALL_100_110: {
      return JXBY_100_110(pC, BGBX, BGBZ, gridSpacing);
   }
   case fsgrids::EYHALL_001_011: {
      return JXBY_001_011(pC, BGBX, BGBZ, gridSpacing);
   }
   case fsgrids::EYHALL_101_111: {
      return JXBY_101_111(pC, BGBX, BGBZ, gridSpacing);
   }
   case fsgrids::EZHALL_000_001: {
      return JXBZ_000_001(pC, BGBX, BGBY, gridSpacing);
   }
   case fsgrids::EZHALL_100_101: {
      return JXBZ_100_101(pC, BGBX, BGBY, gridSpacing);
   }
   case fsgrids::EZHALL_010_011: {
      return JXBZ_010_011(pC, BGBX, BGBY, gridSpacing);
   }
   case fsgrids::EZHALL_110_111: {
      return JXBZ_110_111(pC, BGBX, BGBY, gridSpacing);
   }
   case fsgrids::N_EHALL: {
      break;
   }
   }
   return 0.0;
}

/*! \brief Low-level function computing the Hall term numerator x components.
 *
 * Calls the lower-level inline templates and scales the components properly.
 *
 * \param perbs fsGrid holding the perturbed B quantities
 * \param ehalls fsGrid holding the Hall contributions to the electric field
 * \param moments fsGrid holding the moment quantities
 * \param dperbs fsGrid holding the derivatives of perturbed B
 * \param bgbs fsGrid holding the background B quantities
 * \param gridSpacing fsgrid cell size in x,y,z
 * \param perturbedCoefficients Reconstruction coefficients
 * \param stencil fsGrid stencil for current cell
 *
 * \sa calculateHallTerm JXBX_000_100 JXBX_001_101 JXBX_010_110 JXBX_011_111
 *
 */
void calculateEdgeHallTermComponents(fsgrids::perbspan perbs,
                                     fsgrids::ehallspan ehalls,
                                     fsgrids::constmomentsspan moments,
                                     fsgrids::constdperbspan dperbs,
                                     fsgrids::constbgbspan bgbs,
                                     fsgrids::consttechnicalspan technical, FieldSolverGrid &fsgrid, const fsgrid::FsStencil &stencil,
                                     const std::array<Real, 3>& gridSpacing,
                                     const std::array<Real, Rec::N_REC_COEFFICIENTS>& perturbedCoefficients) {
   const auto ooo = stencil.ooo();
   const auto& bgb = bgbs[ooo];
   const auto& perb = perbs[ooo];
   const auto& dperb = dperbs[ooo];
   const auto& moment = moments[ooo];
   auto& ehall = ehalls[ooo];

   const Real bgbx = bgb[fsgrids::bgbfield::BGBX];
   const Real bgby = bgb[fsgrids::bgbfield::BGBY];
   const Real bgbz = bgb[fsgrids::bgbfield::BGBZ];

   auto computeHallRhoq = [&moments, &moment](const std::array<size_t, 4>& indices) {
      const auto min = Parameters::hallMinimumRhoq;
      const auto max = std::numeric_limits<Real>::max();

      return std::clamp(
          Parameters::ohmHallTerm == 1
              ? moment[fsgrids::moments::RHOQ]
              : FOURTH * (moments[indices[0]][fsgrids::moments::RHOQ] + moments[indices[1]][fsgrids::moments::RHOQ] +
                          moments[indices[2]][fsgrids::moments::RHOQ] + moments[indices[3]][fsgrids::moments::RHOQ]),
          min, max);
   };

   switch (Parameters::ohmHallTerm) {
   case 0:
      cerr << __FILE__ << __LINE__ << "You shouldn't be in a Hall term function if Parameters::ohmHallTerm == 0."
           << endl;
      break;

   case 1: {
      const Real Bx = perb[fsgrids::bfield::PERBX] + bgbx;
      const Real By = perb[fsgrids::bfield::PERBY] + bgby;
      const Real Bz = perb[fsgrids::bfield::PERBZ] + bgbz;

      const Real invHallRhoqMU0 = 1.0 / (physicalconstants::MU_0 * computeHallRhoq({}));

      const Real ydx = (bgb[fsgrids::bgbfield::dBGBydx] + dperb[fsgrids::dperb::dPERBydx]) / gridSpacing[0];
      const Real zdx = (bgb[fsgrids::bgbfield::dBGBzdx] + dperb[fsgrids::dperb::dPERBzdx]) / gridSpacing[0];
      const Real xdy = (bgb[fsgrids::bgbfield::dBGBxdy] + dperb[fsgrids::dperb::dPERBxdy]) / gridSpacing[1];
      const Real zdy = (bgb[fsgrids::bgbfield::dBGBzdy] + dperb[fsgrids::dperb::dPERBzdy]) / gridSpacing[1];
      const Real xdz = (bgb[fsgrids::bgbfield::dBGBxdz] + dperb[fsgrids::dperb::dPERBxdz]) / gridSpacing[2];
      const Real ydz = (bgb[fsgrids::bgbfield::dBGBydz] + dperb[fsgrids::dperb::dPERBydz]) / gridSpacing[2];

      const Real EXHall = (Bz * (xdz - zdx) - By * (ydx - xdy)) * invHallRhoqMU0;
      ehall[fsgrids::ehall::EXHALL_000_100] = EXHall;
      ehall[fsgrids::ehall::EXHALL_010_110] = EXHall;
      ehall[fsgrids::ehall::EXHALL_001_101] = EXHall;
      ehall[fsgrids::ehall::EXHALL_011_111] = EXHall;

      const Real EYHall = (Bx * (ydx - xdy) - Bz * (zdy - ydz)) * invHallRhoqMU0;
      ehall[fsgrids::ehall::EYHALL_000_010] = EYHall;
      ehall[fsgrids::ehall::EYHALL_100_110] = EYHall;
      ehall[fsgrids::ehall::EYHALL_101_111] = EYHall;
      ehall[fsgrids::ehall::EYHALL_001_011] = EYHall;

      const Real EZHall = (By * (zdy - ydz) - Bx * (xdz - zdx)) * invHallRhoqMU0;
      ehall[fsgrids::ehall::EZHALL_000_001] = EZHall;
      ehall[fsgrids::ehall::EZHALL_100_101] = EZHall;
      ehall[fsgrids::ehall::EZHALL_110_111] = EZHall;
      ehall[fsgrids::ehall::EZHALL_010_011] = EZHall;

      break;
   }
   case 2: {
      auto computeEHall = [&perturbedCoefficients, &bgbx, &bgby, &bgbz, &gridSpacing](fsgrids::ehall term,
                                                                                      Real hallRhoq) {
         return JXB(term, perturbedCoefficients, bgbx, bgby, bgbz, gridSpacing) / (physicalconstants::MU_0 * hallRhoq);
      };

      const auto omo = stencil.omo();
      const auto opo = stencil.opo();
      const auto moo = stencil.moo();
      const auto poo = stencil.poo();
      const auto oom = stencil.oom();
      const auto oop = stencil.oop();
      // clang-format off
      const std::array<std::array<size_t, 4>, 12> indices = {
          std::array{
              ooo,
              omo,
              oom,
              stencil.omm(),
          },
          std::array{
              ooo,
              moo,
              oom,
              stencil.mom(),
          },
          std::array{
              ooo,
              moo,
              omo,
              stencil.mmo(),
          },
          std::array{
              ooo,
              poo,
              oom,
              stencil.pom(),
          },
          std::array{
              ooo,
              poo,
              omo,
              stencil.pmo(),
          },
          std::array{
              ooo,
              opo,
              oom,
              stencil.opm(),
          },
          std::array{
              ooo,
              moo,
              opo,
              stencil.mpo(),
          },
          std::array{
              ooo,
              poo,
              opo,
              stencil.ppo(),
          },
          std::array{
              ooo,
              omo,
              oop,
              stencil.omp(),
          },
          std::array{
              ooo,
              moo,
              oop,
              stencil.mop(),
          },
          std::array{
              ooo,
              poo,
              oop,
              stencil.pop(),
          },
          std::array{
              ooo,
              opo,
              oop,
              stencil.opp(),
          },
      };
      // clang-format on

      const std::array<fsgrids::ehall, 12> terms = {
          fsgrids::ehall::EXHALL_000_100, fsgrids::ehall::EYHALL_000_010, fsgrids::ehall::EZHALL_000_001,
          fsgrids::ehall::EYHALL_100_110, fsgrids::ehall::EZHALL_100_101, fsgrids::ehall::EXHALL_010_110,
          fsgrids::ehall::EZHALL_010_011, fsgrids::ehall::EZHALL_110_111, fsgrids::ehall::EXHALL_001_101,
          fsgrids::ehall::EYHALL_001_011, fsgrids::ehall::EYHALL_101_111, fsgrids::ehall::EXHALL_011_111,
      };

      for (size_t index = 0; index < terms.size(); index++) {
         ehall[terms[index]] = computeEHall(terms[index], computeHallRhoq(indices[index]));
      }

      break;
   }

   default:
      cerr << __FILE__ << ":" << __LINE__ << "You are welcome to code higher-order Hall term correction terms." << endl;
      break;
   }
}

void filterHallTerm(fsgrids::ehallspan ehalls,
                    fsgrids::consttechnicalspan technical,
                    FieldSolverGrid &fsgrid) {
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   const std::array<fsgrid::FsSize_t, 3> &globalSize = fsgrid.getGlobalSize();

   Array3D EHallComponent_global(globalSize[0], globalSize[1], globalSize[2]);
   Array3D EHallComponent_filtered(globalSize[0], globalSize[1], globalSize[2]);

   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("Collect Hall term x components"),
      technical,

      [&EHallComponent_global, ehalls]
      (const fsgrid::Coordinates &coordinates,
      const fsgrid::FsStencil& stencil,
      cuint sysBoundaryFlag,
      cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         EHallComponent_global(globalIndices[0], globalIndices[1], globalIndices[2]) = ehalls[stencil.ooo()][fsgrids::ehall::EXHALL_000_100];
      });

   if (myRank == MASTER_RANK) {
      MPI_Reduce(MPI_IN_PLACE, &EHallComponent_global.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }
   else {
      MPI_Reduce(&EHallComponent_global.data[0], nullptr, globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }

   if (myRank == MASTER_RANK){

      ComplexArray3D complex_buffer(globalSize[0],globalSize[1],globalSize[2]);

      auto base_addr  = &complex_buffer.data[0];
      fftw_plan fwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_plan rwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_BACKWARD, FFTW_ESTIMATE);

      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               complex_buffer(i,j,k)[0] = EHallComponent_global(i,j,k);
               complex_buffer(i,j,k)[1] = 0.0;
            }
         }
      }

      // Execute the FTT
      fftw_execute(fwd_plan);

   // Perform point by point operation in k-space
      const auto dxyz = fsgrid.getGridSpacing();
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         double kx = i / double(globalSize[0]);
               kx = kx <= 0.5 ? kx : kx-1.;
               kx *= 2. * M_PI / dxyz[0];
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            double ky = j / double(globalSize[1]);
                  ky = ky <= 0.5 ? ky : ky-1.;
                  ky *= 2. * M_PI / dxyz[1];
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               double kz = k / double(globalSize[2]);
                     kz = kz <= 0.5 ? kz : kz-1.;
                     kz *= 2. * M_PI / dxyz[2];

               double ksquared = kx*kx + ky*ky + kz*kz;

               if(ksquared < std::pow(10,-9)) {
                  complex_buffer(i,j,k)[0] = complex_buffer(i,j,k)[0];
                  complex_buffer(i,j,k)[1] = complex_buffer(i,j,k)[1];
               } else {
                  complex_buffer(i,j,k)[0] = 0.;
                  complex_buffer(i,j,k)[1] = 0.;
               }
            }
         }
      }

      fftw_execute(rwd_plan);

      double max_real_mag = 0.;
      double max_imag_mag = 0.;
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               max_real_mag = std::max(max_real_mag, std::abs(complex_buffer(i,j,k)[0]));
               max_imag_mag = std::max(max_imag_mag, std::abs(complex_buffer(i,j,k)[1]));
               EHallComponent_filtered(i,j,k) = complex_buffer(i,j,k)[0] / double(globalSize[0]*globalSize[1]*globalSize[2]); // Division comes from unnormalized nature of the FFTW transformations
            }
         }
      }

      fftw_destroy_plan(fwd_plan);
      fftw_destroy_plan(rwd_plan);
   }

   // Use MPI_Broadcast from MASTER_RANK to update everyone on the value of phi
   MPI_Bcast(&EHallComponent_filtered.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);

   // Distribute local values of Phi
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("set local values of Phi"),
      technical,

      [&ehalls,&EHallComponent_filtered,myRank]
      (const fsgrid::Coordinates &coordinates,
      const fsgrid::FsStencil& stencil,
      cuint sysBoundaryFlag,
      cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         const Real val = EHallComponent_filtered(globalIndices[0], globalIndices[1], globalIndices[2]);
         
         ehalls[stencil.ooo()][fsgrids::ehall::EXHALL_000_100] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EXHALL_010_110] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EXHALL_001_101] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EXHALL_011_111] = val;
      });

   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("Collect Hall term x components"),
      technical,

      [&EHallComponent_global, ehalls]
      (const fsgrid::Coordinates &coordinates,
      const fsgrid::FsStencil& stencil,
      cuint sysBoundaryFlag,
      cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         EHallComponent_global(globalIndices[0], globalIndices[1], globalIndices[2]) = ehalls[stencil.ooo()][fsgrids::ehall::EYHALL_000_010];
      });

   if (myRank == MASTER_RANK) {
      MPI_Reduce(MPI_IN_PLACE, &EHallComponent_global.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }
   else {
      MPI_Reduce(&EHallComponent_global.data[0], nullptr, globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }

   if (myRank == MASTER_RANK){

      ComplexArray3D complex_buffer(globalSize[0],globalSize[1],globalSize[2]);

      auto base_addr  = &complex_buffer.data[0];
      fftw_plan fwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_plan rwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_BACKWARD, FFTW_ESTIMATE);

      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               complex_buffer(i,j,k)[0] = EHallComponent_global(i,j,k);
               complex_buffer(i,j,k)[1] = 0.0;
            }
         }
      }

      // Execute the FTT
      fftw_execute(fwd_plan);

      // Perform point by point operation in k-space
      const auto dxyz = fsgrid.getGridSpacing();
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         double kx = i / double(globalSize[0]);
               kx = kx <= 0.5 ? kx : kx-1.;
               kx *= 2. * M_PI / dxyz[0];
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            double ky = j / double(globalSize[1]);
                  ky = ky <= 0.5 ? ky : ky-1.;
                  ky *= 2. * M_PI / dxyz[1];
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               double kz = k / double(globalSize[2]);
                     kz = kz <= 0.5 ? kz : kz-1.;
                     kz *= 2. * M_PI / dxyz[2];

               double ksquared = kx*kx + ky*ky + kz*kz;

               if(ksquared < std::pow(10,-9)) {
                  complex_buffer(i,j,k)[0] = complex_buffer(i,j,k)[0];
                  complex_buffer(i,j,k)[1] = complex_buffer(i,j,k)[1];
               } else {
                  complex_buffer(i,j,k)[0] = 0.;
                  complex_buffer(i,j,k)[1] = 0.;
               }
            }
         }
      }

   // Perform inverse FFT on rho_complex
      fftw_execute(rwd_plan);

   // Copy out the real part of complex_buffer, which holds phi at that point
      double max_real_mag = 0.;
      double max_imag_mag = 0.;
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               max_real_mag = std::max(max_real_mag, std::abs(complex_buffer(i,j,k)[0]));
               max_imag_mag = std::max(max_imag_mag, std::abs(complex_buffer(i,j,k)[1]));
               EHallComponent_filtered(i,j,k) = complex_buffer(i,j,k)[0] / double(globalSize[0]*globalSize[1]*globalSize[2]); // Division comes from unnormalized nature of the FFTW transformations
            }
         }
      }

      fftw_destroy_plan(fwd_plan);
      fftw_destroy_plan(rwd_plan);
   }

   MPI_Bcast(&EHallComponent_filtered.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("set new E hall y filtered components"),
      technical,

      [&ehalls,&EHallComponent_filtered,myRank]
      (const fsgrid::Coordinates &coordinates,
      const fsgrid::FsStencil& stencil,
      cuint sysBoundaryFlag,
      cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         const Real val = EHallComponent_filtered(globalIndices[0], globalIndices[1], globalIndices[2]);
         
         ehalls[stencil.ooo()][fsgrids::ehall::EYHALL_000_010] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EYHALL_100_110] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EYHALL_101_111] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EYHALL_001_011] = val;
      });

   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("Collect Hall term y components"),
      technical,

      [&EHallComponent_global, ehalls]
      (const fsgrid::Coordinates &coordinates,
      const fsgrid::FsStencil& stencil,
      cuint sysBoundaryFlag,
      cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         EHallComponent_global(globalIndices[0], globalIndices[1], globalIndices[2]) = ehalls[stencil.ooo()][fsgrids::ehall::EZHALL_000_001];
      });

   if (myRank == MASTER_RANK) {
      MPI_Reduce(MPI_IN_PLACE, &EHallComponent_global.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }
   else {
      MPI_Reduce(&EHallComponent_global.data[0], nullptr, globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }

   if (myRank == MASTER_RANK){

      ComplexArray3D complex_buffer(globalSize[0],globalSize[1],globalSize[2]);

      auto base_addr  = &complex_buffer.data[0];
      fftw_plan fwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_plan rwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_BACKWARD, FFTW_ESTIMATE);

      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               complex_buffer(i,j,k)[0] = EHallComponent_global(i,j,k);
               complex_buffer(i,j,k)[1] = 0.0;
            }
         }
      }

      // Execute the FTT
      fftw_execute(fwd_plan);

   // Perform point by point operation in k-space
      bool triggered = false;
      const auto dxyz = fsgrid.getGridSpacing();
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         double kx = i / double(globalSize[0]);
               kx = kx <= 0.5 ? kx : kx-1.;
               kx *= 2. * M_PI / dxyz[0];
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            double ky = j / double(globalSize[1]);
                  ky = ky <= 0.5 ? ky : ky-1.;
                  ky *= 2. * M_PI / dxyz[1];
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               double kz = k / double(globalSize[2]);
                     kz = kz <= 0.5 ? kz : kz-1.;
                     kz *= 2. * M_PI / dxyz[2];

               double ksquared = kx*kx + ky*ky + kz*kz;

               if(ksquared < std::pow(10,-9)) {
                  complex_buffer(i,j,k)[0] = complex_buffer(i,j,k)[0];
                  complex_buffer(i,j,k)[1] = complex_buffer(i,j,k)[1];
               } else {
                  complex_buffer(i,j,k)[0] = 0.;
                  complex_buffer(i,j,k)[1] = 0.;
                  triggered = true;
               }
            }
         }
      }
      if (triggered){
         fprintf(stderr, "0 triggered");
      }

   // Perform inverse FFT on rho_complex
      fftw_execute(rwd_plan);

      // Copy out the real part of complex_buffer, which holds phi at that point
      double max_real_mag = 0.;
      double max_imag_mag = 0.;
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               max_real_mag = std::max(max_real_mag, std::abs(complex_buffer(i,j,k)[0]));
               max_imag_mag = std::max(max_imag_mag, std::abs(complex_buffer(i,j,k)[1]));
               EHallComponent_filtered(i,j,k) = complex_buffer(i,j,k)[0] / double(globalSize[0]*globalSize[1]*globalSize[2]); // Division comes from unnormalized nature of the FFTW transformations
            }
         }
      }

      fftw_destroy_plan(fwd_plan);
      fftw_destroy_plan(rwd_plan);
   }

   // Use MPI_Broadcast from MASTER_RANK to update everyone on the value of phi
   MPI_Bcast(&EHallComponent_filtered.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);

   // Distribute local values of Phi
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("set local values of Phi"),
      technical,

      [&ehalls,&EHallComponent_filtered,myRank]
      (const fsgrid::Coordinates &coordinates,
      const fsgrid::FsStencil& stencil,
      cuint sysBoundaryFlag,
      cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         const Real val = EHallComponent_filtered(globalIndices[0], globalIndices[1], globalIndices[2]);
         
         ehalls[stencil.ooo()][fsgrids::ehall::EZHALL_000_001] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EZHALL_100_101] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EZHALL_110_111] = val;
         ehalls[stencil.ooo()][fsgrids::ehall::EZHALL_010_011] = val;
      });

      fsgrid.updateGhostCells(ehalls);
}

/** \brief Calculate the numerator of the Hall term on all given cells.
 *
 * \param perb fsGrid holding the perturbed B quantities
 * \param ehall fsGrid holding the Hall contributions to the electric field
 * \param moments fsGrid holding the moment quantities
 * \param dperb fsGrid holding the derivatives of perturbed B
 * \param bgb fsGrid holding the background B quantities
 * \param technical fsGrid holding technical information (such as boundary types)
 * \param stencil fsgrid stencil of current cell
 * \param sysBoundaries System boundary condition functions.
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 * \sa calculateHallTermSimple calculateEdgeHallTermComponents
 */
void calculateHallTerm(fsgrids::perbspan perb,
                       fsgrids::ehallspan ehall,
                       fsgrids::constmomentsspan moments,
                       fsgrids::constdperbspan dperb,
                       fsgrids::constbgbspan bgb,
                       fsgrids::consttechnicalspan technical, FieldSolverGrid &fsgrid, const fsgrid::FsStencil& stencil,
                       SysBoundary& sysBoundaries, const std::array<Real, 3>& gridSpacing) {
#ifdef DEBUG_FSOLVER
   if (!stencil.cellExists(0, 0, 0)) {
      cerr << "Out-of-bounds access in " << __FILE__ << ":" << __LINE__ << endl;
      exit(1);
   }
#endif

   const auto& tech = technical[stencil.ooo()];
   cuint cellSysBoundaryFlag = tech.sysBoundaryFlag;
   cuint cellSysBoundaryLayer = tech.sysBoundaryLayer;

   if (cellSysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE ||
       cellSysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) {
      return;
   }

   // Reconstruction order of the fields after Balsara 2009, 2 used for general B, 3 used
   // here for 2nd-order Hall term
   const std::array<Real, Rec::N_REC_COEFFICIENTS> perturbedCoefficients =
       reconstructionCoefficients(perb, dperb, stencil, 3);

   if ((cellSysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) && (cellSysBoundaryLayer != 1)) {
      auto* const sb = sysBoundaries.getSysBoundary(cellSysBoundaryFlag);
      sb->fieldSolverBoundaryCondHallElectricField(ehall, stencil, 0);
      sb->fieldSolverBoundaryCondHallElectricField(ehall, stencil, 1);
      sb->fieldSolverBoundaryCondHallElectricField(ehall, stencil, 2);
   } else {
      calculateEdgeHallTermComponents(perb, ehall, moments, dperb, bgb, technical, fsgrid, stencil, gridSpacing, perturbedCoefficients);
   }
}

/*! \brief High-level function computing the Hall term.
 *
 * Performs the communication before and after the computation as well as the computation of all Hall term numerator
 * components.
 *
 * \param perb fsGrid holding the perturbed B quantities
 * \param perbdt2 fsGrid holding the perturbed B quantities at runge-kutta half step
 * \param ehall fsGrid holding the Hall contributions to the electric field
 * \param moments fsGrid holding the moment quantities
 * \param momentsdt2 fsGrid holding the moment quantities at runge-kutta half step
 * \param dperb fsGrid holding the derivatives of perturbed B
 * \param dmoments fsGrid holding the derivatives of moments
 * \param dmomentsdt2 fsGrid holding the derivatives of moments at Runge-Kutta half step
 * \param bgb fsGrid holding the background B quantities
 * \param technical fsGrid holding technical information (such as boundary types)
 * \param fsgrid fsgrids container
 * \param sysBoundaries System boundary condition functions.
 * \param RKCase Element in the enum defining the Runge-Kutta method steps
 * \param communicateMomentsDerivatives Boolean flag whether to ghost update moments derivatives
 *
 * \sa calculateHallTerm
 */
void calculateHallTermSimple(fsgrids::perbspan perb,
                             fsgrids::perbspan perbdt2,
                             fsgrids::ehallspan ehall,
                             fsgrids::momentsspan moments,
                             fsgrids::momentsspan momentsdt2,
                             fsgrids::dperbspan dperb,
                             fsgrids::dmomentsspan dmoments,
                             fsgrids::dmomentsspan dmomentsdt2,
                             fsgrids::bgbspan bgb,
                             fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                             SysBoundary& sysBoundaries, int32_t RKCase, const bool communicateMomentsDerivatives) {


   if (not(RKCase == RK_ORDER1 || RKCase == RK_ORDER2_STEP2)) {
      perb = perbdt2;
      moments = momentsdt2;
      dmoments = dmomentsdt2;
   }
   phiprof::Timer hallTimer{"Calculate Hall term"};

   const size_t numCells = fsgrid.getNumCells();

   phiprof::Timer mpiTimer{"EHall ghost updates MPI", {"MPI"}};
   fsgrid.updateGhostCells(dperb);
   if (P::ohmGradPeTerm == 0 && communicateMomentsDerivatives) {
      fsgrid.updateGhostCells(dmoments);
   }
   mpiTimer.stop();

   fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                       phiprof::initializeTimer("EHall compute cells"), technical,
                       [=, &fsgrid, &sysBoundaries](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
                          calculateHallTerm(perb, ehall, moments, dperb, bgb, technical, fsgrid, stencil, sysBoundaries, coordinates.physicalGridSpacing);
                       });

   filterHallTerm(ehall, technical, fsgrid);

   hallTimer.stop(numCells, "Spatial Cells");
}
