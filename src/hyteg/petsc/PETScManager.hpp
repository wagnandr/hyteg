#pragma once

#include "PETScWrapper.hpp"

#ifdef HHG_BUILD_WITH_PETSC

class PETScManager
{
 public:
   PETScManager() { PetscInitializeNoArguments(); }
   PETScManager( int* argc, char*** argv ) { PetscInitialize( argc, argv, NULL, NULL ); }

   ~PETScManager() { PetscFinalize(); }
};

#endif