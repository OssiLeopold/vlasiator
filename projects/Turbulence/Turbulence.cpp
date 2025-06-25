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

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "../../backgroundfield/backgroundfield.h"
#include "../../backgroundfield/constantfield.hpp"
#include "../../common.h"
#include "../../object_wrapper.h"
#include "../../readparameters.h"

#include "Turbulence.h"

using namespace spatial_cell;

namespace projects {
Turbulence::Turbulence() : Project() {}
Turbulence::~Turbulence() {}

void Turbulence::addParameters() {
   typedef Readparameters RP;
   
   RP::add("Turbulence.numberOfWaves", "Number of waves in the simulation", 1);

   RP::addComposing("Turbulence.wavelength", "Wavelength of wave (m)");
   RP::addComposing("Turbulence.amplitude", "Velocity amplitude (m/s)");
   RP::addComposing("Turbulence.phase", "Initial phase (rad)");
   
   
   RP::add("Turbulence.n0", "Background density (1/m^3)", 1e6);
   RP::add("Turbulence.B", "Background magnetic field strength (T)", 1e-8);
   RP::add("Turbulence.T", "Temperature (K)", 1e6);
   RP::add("Turbulence.spectralIndex", "Power law index for initial spectrum", -5.0/3.0);
   RP::add("Turbulence.randomSeed", "Seed for random phase generation", 12345);
   RP::add("Turbulence.verbose", "Verbose output", 1);
   RP::add("Turbulence.angle", "Wave angle (rad)",0.0);
}

void Turbulence::getParameters() {
   typedef Readparameters RP;
   Project::getParameters();

   RP::get("Turbulence.numberOfWaves", nWaves);

   RP::get("Turbulence.wavelength", wavelength);
   RP::get("Turbulence.amplitude", amplitude);
   RP::get("Turbulence.phase", phase);

   // We need the correct number of parameters for the waves
   if(   nWaves != (int)wavelength.size()
      || nWaves != (int)amplitude.size()
      || nWaves != (int)phase.size()
   ) {
      cerr << "Turbulence.numberOfWaves is set to " << nWaves << " so the same number of values is required for Turbulence.wavelength, Turbulence.amplitude, Turbulence.phase" << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
   }

   // Get scalar parameters
   RP::get("Turbulence.n0", n0);
   RP::get("Turbulence.B", B);
   RP::get("Turbulence.T", T);
   RP::get("Turbulence.spectralIndex", spectralIndex);
   RP::get("Turbulence.randomSeed", randomSeed);
   RP::get("Turbulence.verbose", verbose);
   RP::get("Turbulence.angle", angle);
}

bool Turbulence::initialize(void) {
   bool success = Project::initialize();

   creal m = physicalconstants::MASS_PROTON;
   creal e = physicalconstants::CHARGE;
   creal kB = physicalconstants::K_B;
   creal gamma = 5.0 / 3.0;
   creal mu0 = physicalconstants::MU_0;

   rho0 = m * n0; // Mass density
   p0 = n0 * kB * T; // pressure

   std::vector<WaveParameters> waves {};
   // Initialize waves based on parameters
   for (int idx = 0; idx < nWaves; idx++) {
       WaveParameters wave;
       wave.wavelength = wavelength.at(idx);
       wave.amplitude = amplitude.at(idx);
       wave.phase = phase.at(idx);
       waves.push_back(wave);
   }

   // Calculate Alfvén speed
   VA = B / sqrt(mu0 * rho0);

   if (verbose) {
      int myRank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
      if (myRank == MASTER_RANK) {
         std::cout << "Initialized multi-wave turbulence simulation\n";
         std::cout << "Number of waves: " << nWaves << "\n";
         std::cout << "Background field strength: " << B << " T\n";
         std::cout << "Alfvén speed: " << VA << " m/s\n";
         
         for (int idx = 0; idx < nWaves; idx++) {
             std::cout << "\nWave " << idx + 1 << ":\n";
             std::cout << "Wavelength: " << wavelength.at(idx) << " m\n";
             std::cout << "Amplitude: " << amplitude.at(idx) << " m/s\n";
             std::cout << "Phase: " << phase.at(idx) << " rad\n";
             std::cout << "Angle: " << angle * 180/M_PI << " degrees\n";
         }
      }
   }
   return success;
}

void Turbulence::calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) {}

Realf Turbulence::fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                       const uint popID,
                                       const uint nRequested
      ) const {
      // const AlfvenSpeciesParameters& sP = this->speciesParams[popID];

      // Fetch spatial cell center coordinates
      const Real x  = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];
      const Real y  = cell->parameters[CellParams::YCRD] + 0.5*cell->parameters[CellParams::DY];

      creal mass = physicalconstants::MASS_PROTON;
      creal mu0 = physicalconstants::MU_0;
      Real ux = 0.0, uy = 0.0, uz = 0.0;
      std::array<Real,4> kx {1,-1,-1,1};
      std::array<Real,4> ky {1,1,-1,-1};

      for (int idx = 0; idx < nWaves; idx++) {
         Real kwave = 2 * M_PI / wavelength.at(idx);
         Real kwave_x = kwave * kx.at(idx);
         Real kwave_y = kwave * ky.at(idx);
         
         ux += amplitude.at(idx) * cos(kwave_x * x + kwave_y * y + phase.at(idx));
         uy += - kwave_x / kwave_y * amplitude.at(idx) * cos(kwave_x * x + kwave_y * y + phase.at(idx));
         uz += 0;
      }
      creal initV0X = ux;
      creal initV0Y = uy;
      creal initV0Z = uz;

      Real initRho = n0;
      Real initT = T;

      #ifdef USE_GPU
      vmesh::VelocityMesh *vmesh = cell->dev_get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->dev_get_velocity_blocks(popID);
      #else
      vmesh::VelocityMesh *vmesh = cell->get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->get_velocity_blocks(popID);
      #endif
      // Loop over blocks
      Realf rhosum = 0;
      arch::parallel_reduce<arch::null>(
         {WID, WID, WID, nRequested},
         ARCH_LOOP_LAMBDA (const uint i, const uint j, const uint k, const uint initIndex, Realf *lsum ) {
            vmesh::GlobalID *GIDlist = vmesh->getGrid()->data();
            Realf* bufferData = VBC->getData();
            const vmesh::GlobalID blockGID = GIDlist[initIndex];
            // Calculate parameters for new block
            Real blockCoords[6];
            vmesh->getBlockInfo(blockGID,&blockCoords[0]);
            creal vxBlock = blockCoords[0];
            creal vyBlock = blockCoords[1];
            creal vzBlock = blockCoords[2];
            creal dvxCell = blockCoords[3];
            creal dvyCell = blockCoords[4];
            creal dvzCell = blockCoords[5];
            ARCH_INNER_BODY(i, j, k, initIndex, lsum) {
               creal vx = vxBlock + (i+0.5)*dvxCell - initV0X;
               creal vy = vyBlock + (j+0.5)*dvyCell - initV0Y;
               creal vz = vzBlock + (k+0.5)*dvzCell - initV0Z;
               const Realf value = MaxwellianPhaseSpaceDensity(vx,vy,vz,initT,initRho,mass);
               bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
               //lsum[0] += value;
            };
         }, rhosum);
      return rhosum;
   }

void Turbulence::setProjectBField(FsGrid<std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH>& perBGrid,
                                    FsGrid<std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH>& BgBGrid,
                                    FsGrid<fsgrids::technical, FS_STENCIL_WIDTH>& technicalGrid) {
   // Set background field
   ConstantField bgField;
   bgField.initialize(0.0, 0.0, B); // Background field according to angle
   setBackgroundField(bgField, BgBGrid);

   if (!P::isRestart) {
      auto localSize = perBGrid.getLocalSize().data();

   creal mu0 = physicalconstants::MU_0;

#pragma omp parallel for collapse(3)
      for (int i = 0; i < localSize[0]; ++i) {
         for (int j = 0; j < localSize[1]; ++j) {
            for (int k = 0; k < localSize[2]; ++k) {
               const std::array<Real, 3> x = perBGrid.getPhysicalCoords(i, j, k);
               std::array<Real, fsgrids::bfield::N_BFIELD>* cell = perBGrid.get(i, j, k);

               Real Bx = 0.0, By = 0.0, Bz = 0.0;
               std::array<Real,8> kx {1,-1,-1,1,0,-1,0,1};
               std::array<Real,8> ky {1,1,-1,-1,1,0,-1,0};

               // Sum contributions from all waves
               for (int idx = 0; idx < nWaves; idx++) {
                   Real kwave = 2 * M_PI / wavelength.at(idx);
                   Real kwave_x = kwave * kx.at(idx);
                   Real kwave_y = kwave * ky.at(idx);
                   Real B1 = std::pow(-1.0,idx) * amplitude.at(idx) * sqrt(mu0 * rho0);

                   Bx += B1 * cos(kwave_x * x[0] + kwave_y * x[1] + phase.at(idx));
                   By += - kwave_x / kwave_y * B1 * cos(kwave_x * x[0] + kwave_y * x[1] + phase.at(idx));
                   Bz += 0;
               }

               cell->at(fsgrids::bfield::PERBX) = Bx;
               cell->at(fsgrids::bfield::PERBY) = By;
               cell->at(fsgrids::bfield::PERBZ) = Bz;
            }
         }
      }
   }
}

} // namespace projects