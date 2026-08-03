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
Turbulence::Turbulence() : TriAxisSearch() {}
Turbulence::~Turbulence() {}

void Turbulence::addParameters() {
   typedef Readparameters RP;
   
   RP::add("Turbulence.numberOfWaves", "Number of waves in the simulation", 1);
   RP::add("Turbulence.amplitude", "Velocity amplitude (m/s)", 1e4);
   
   RP::add("Turbulence.n0", "Background density (1/m^3)", 1e6);
   RP::add("Turbulence.B", "Background magnetic field strength (T)", 1e-8);
   RP::add("Turbulence.T", "Temperature (K)", 1e6);
   RP::add("Turbulence.spectralIndex", "Power law index for initial spectrum", -5.0/3.0);
   RP::add("Turbulence.randomSeed", "Seed for random phase generation", 12345);
   RP::add("Turbulence.numberOfPossible", "How many possible wavelengths will be generated to pick from", 8);
   RP::add("Turbulence.verbose", "Verbose output", 1);
}

void Turbulence::getParameters() {
   typedef Readparameters RP;
   Project::getParameters();

   RP::get("Turbulence.numberOfWaves", nWaves);
   RP::get("Turbulence.amplitude", amplitude);
   RP::get("Turbulence.numberOfPossible", n_possible);

   // Get scalar parameters
   RP::get("Turbulence.n0", n0);
   RP::get("Turbulence.B", B);
   RP::get("Turbulence.T", T);
   RP::get("Turbulence.spectralIndex", spectralIndex);
   RP::get("Turbulence.randomSeed", randomSeed);
   RP::get("Turbulence.verbose", verbose);
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

   std::mt19937 gen1(randomSeed);
   std::uniform_int_distribution<> random_number(0, 1000000);

   std::mt19937 gen2(random_number(gen1));
   std::mt19937 gen3(random_number(gen1));
   std::mt19937 gen4(random_number(gen1));
   std::mt19937 gen5(random_number(gen1));

   // Construct random kx and ky
   std::vector<Real> allowed_k {0};
   Real k_min {2 * M_PI / Parameters::xmax};

   for (int idx = 1; idx <= n_possible; idx++){
      allowed_k.push_back(- k_min * idx);
      allowed_k.push_back(k_min * idx);
   }

   std::uniform_int_distribution<> random_index(0, 2 * n_possible);
   int index_x {};
   int index_y {};
   int index_z {};
   Real kx_temp {};
   Real ky_temp {};
   Real kz_temp {};
   Real k_mag {};
   
   for (int idx = 0; idx < nWaves;){
      index_x = random_index(gen2);
      index_y = random_index(gen3);
      index_z = random_index(gen5);

      kx_temp = allowed_k.at(index_x);
      ky_temp = allowed_k.at(index_y);
      kz_temp = allowed_k.at(index_z);
      k_mag = sqrt(std::pow(kx_temp, 2) + std::pow(ky_temp, 2));

      if (allowed_k.at(2 * n_possible) >= k_mag && k_mag > 0){
         kx.push_back(kx_temp);
         ky.push_back(ky_temp);
         kz.push_back(kz_temp);
         idx++;
      }
   }
   
   // Construct random phases
   std::uniform_real_distribution<> random_phase(0, 2.0 * M_PI);
   for (int idx = 0; idx < nWaves; idx++){
      phase_rand.push_back(random_phase(gen4));
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
         std::cout << "Amplitude: " << amplitude << " m/s\n";

         for (int idx = 0; idx < nWaves; idx++) {
             std::cout << "\nWave " << idx + 1 << ":\n";
             std::cout << "kx: " << kx.at(idx) << "\n";
             std::cout << "ky: " << ky.at(idx) << "\n";
             std::cout << "Phase: " << phase_rand.at(idx) << " rad\n";
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
      const Real z  = cell->parameters[CellParams::ZCRD] + 0.5*cell->parameters[CellParams::DZ];

      creal mass = physicalconstants::MASS_PROTON;
      creal mu0 = physicalconstants::MU_0;
      Real ux = 0.0, uy = 0.0, uz = 0.0;

      for (int idx = 0; idx < nWaves; idx++) {
         ux += ky.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * amplitude * cos(kx.at(idx) * x + ky.at(idx) * y + kz.at(idx) * z + phase_rand.at(idx));
         uy += kx.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * amplitude * cos(kx.at(idx) * x + ky.at(idx) * y + kz.at(idx) * z + phase_rand.at(idx));
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

               // Sum contributions from all waves
               for (int idx = 0; idx < nWaves; idx++) {
                   Real B1 = std::pow(-1.0,idx) * amplitude * sqrt(mu0 * rho0);

                   Bx += ky.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * B1 * cos(kx.at(idx) * x[0] + ky.at(idx) * x[1] + kz.at(idx) * x[2] + phase_rand.at(idx));
                   By += - kx.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * B1 * cos(kx.at(idx) * x[0] + ky.at(idx) * x[1] + kz.at(idx) * x[2] + phase_rand.at(idx));
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

 Realf Turbulence::probePhaseSpace(spatial_cell::SpatialCell *cell,
                                        const uint popID,
                                        Real vx_in, Real vy_in, Real vz_in
      ) const {
      
      const Real x  = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];
      const Real y  = cell->parameters[CellParams::YCRD] + 0.5*cell->parameters[CellParams::DY];

      creal mass = physicalconstants::MASS_PROTON;
      creal mu0 = physicalconstants::MU_0;
      Real ux = 0.0, uy = 0.0, uz = 0.0;

      for (int idx = 0; idx < nWaves; idx++) {
         ux += ky.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * amplitude * cos(kx.at(idx) * x + ky.at(idx) * y + phase_rand.at(idx));
         uy += - kx.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * amplitude * cos(kx.at(idx) * x + ky.at(idx) * y + phase_rand.at(idx));
         uz += 0;
      }
      creal initV0X = ux;
      creal initV0Y = uy;
      creal initV0Z = uz;

      Real initRho = n0;
      Real initT = T;

      creal vx = vx_in - initV0X;
      creal vy = vy_in - initV0Y;
      creal vz = vz_in - initV0Z;
      const Realf value = MaxwellianPhaseSpaceDensity(vx,vy,vz,initT,initRho,mass);
      return value;
   }

std::vector<std::array<Real, 3>> Turbulence::getV0(creal x, creal y, creal z, const uint popID) const{
   std::vector<std::array<Real, 3>> centerPoints;

   Real ux = 0.0, uy = 0.0, uz = 0.0;
   for (int idx = 0; idx < nWaves; idx++) {
         ux += ky.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * amplitude * cos(kx.at(idx) * x + ky.at(idx) * y + phase_rand.at(idx));
         uy += - kx.at(idx) / sqrt(std::pow(kx.at(idx),2) + std::pow(ky.at(idx),2)) * amplitude * cos(kx.at(idx) * x + ky.at(idx) * y + phase_rand.at(idx));
         uz += 0;
      }
   
   std::array<Real, 3> point {{ux, uy, uz}};
   centerPoints.push_back(point);
   return centerPoints;
}

} // namespace projects