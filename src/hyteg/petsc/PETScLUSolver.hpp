/*
 * Copyright (c) 2017-2022 Boerge Struempfel, Daniel Drzisga, Dominik Thoennes, Nils Kohl.
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
#pragma once

#include <memory>

#include "hyteg/solvers/Solver.hpp"

#include "PETScSparseMatrix.hpp"
#include "PETScVector.hpp"

#ifdef HYTEG_BUILD_WITH_PETSC

#if ( PETSC_VERSION_MAJOR > 3 ) || ( PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 9 )
#define HYTEG_PCFactorSetMatSolverType PCFactorSetMatSolverType
#else
#define HYTEG_PCFactorSetMatSolverType PCFactorSetMatSolverPackage
#endif

namespace hyteg {

enum class PETScDirectSolverType
{
   MUMPS,
   SUPER_LU
};

template < class OperatorType >
class PETScLUSolver : public Solver< OperatorType >
{
 public:
   typedef typename OperatorType::srcType FunctionType;

   PETScLUSolver( const std::shared_ptr< PrimitiveStorage >& storage, const uint_t& level )
   : storage_( storage )
   , allocatedLevel_( level )
   , petscCommunicator_( storage->getSplitCommunicatorByPrimitiveDistribution() )
   , num( "numerator", storage, level, level )
   , Amat( "Amat", petscCommunicator_ )
   , AmatUnsymmetric( "AmatUnsymmetric", petscCommunicator_ )
   , AmatTmp( "AmatTmp", petscCommunicator_ )
   , xVec( "xVec", petscCommunicator_ )
   , bVec( "bVec", petscCommunicator_ )
#if 0
  , inKernel( numberOfLocalDoFs< typename FunctionType::Tag >( *storage, level ) )
#endif
   , flag_( hyteg::All )
   , verbose_( false )
   , manualAssemblyAndFactorization_( false )
   , reassembleMatrix_( false )
   , assumeSymmetry_( true )
   , solverType_( PETScDirectSolverType::MUMPS )
   {
      num.enumerate( level );
      KSPCreate( petscCommunicator_, &ksp );
      KSPSetType( ksp, KSPPREONLY );
      KSPSetFromOptions( ksp );
   }

   ~PETScLUSolver() { KSPDestroy( &ksp ); }

#if 0
  void setNullSpace( FunctionType & inKernel, const uint_t & level )
  {
    inKernel.createVectorFromFunction( inKernel, *num, level, All );
    VecNormalize(inKernel.get(), NULL);
    MatNullSpace nullspace;
    MatNullSpaceCreate( walberla::MPIManager::instance()->comm(), PETSC_FALSE, 1, &(inKernel.get()), &nullspace );
    MatSetNullSpace( Amat.get(), nullspace );
  }
#endif

   void setDirectSolverType( PETScDirectSolverType solverType ) { solverType_ = solverType; }

   void setConstantNullSpace()
   {
      MatNullSpace nullspace;
      MatNullSpaceCreate( petscCommunicator_, PETSC_TRUE, 0, NULL, &nullspace );
      MatSetNullSpace( Amat.get(), nullspace );
   }

   void setVerbose( bool verbose ) { verbose_ = verbose; }

   /// \brief If set to true, the symmetry of the operator is exploited by the solver.
   void assumeSymmetry( bool assumeSymmetry ) { assumeSymmetry_ = assumeSymmetry; }

   /// \brief If set to true, no assembly and no factorization will happen during the solve() call.
   ///        For successful solution of the system, assembleAndFactorize() has to be called before
   ///        the first solve() and after each modification of the operator.
   void setManualAssemblyAndFactorization( bool manualAssemblyAndFactorization )
   {
      manualAssemblyAndFactorization_ = manualAssemblyAndFactorization;
   }

   /// \brief If set to true, the operator is reassembled for every solve / manual assembly call.
   ///        Default is false.
   void reassembleMatrix( bool reassembleMatrix ) { reassembleMatrix_ = reassembleMatrix; }

   void setMUMPSIcntrl( uint_t key, int value ) { mumpsIcntrl_[key] = value; }
   void setMUMPSCntrl( uint_t key, real_t value ) { mumpsCntrl_[key] = value; }

   void assembleAndFactorize( const OperatorType& A )
   {
      storage_->getTimingTree()->start( "Matrix assembly" );

      bool matrixAssembledForTheFirstTime;
      if ( reassembleMatrix_ )
      {
         AmatUnsymmetric.zeroEntries();
         AmatUnsymmetric.createMatrixFromOperator( A, allocatedLevel_, num, All );
         matrixAssembledForTheFirstTime = true;
      }
      else
      {
         matrixAssembledForTheFirstTime = AmatUnsymmetric.createMatrixFromOperatorOnce( A, allocatedLevel_, num, All );
      }

      storage_->getTimingTree()->stop( "Matrix assembly" );

      if ( matrixAssembledForTheFirstTime )
      {
         Amat.zeroEntries();
         Amat.createMatrixFromOperator( A, allocatedLevel_, num, All );
         AmatTmp.zeroEntries();
         AmatTmp.createMatrixFromOperator( A, allocatedLevel_, num, All );

         MatCopy( AmatUnsymmetric.get(), Amat.get(), DIFFERENT_NONZERO_PATTERN );

         if ( assumeSymmetry_ )
         {
            Amat.applyDirichletBCSymmetrically( num, allocatedLevel_ );
         }
         else
         {
            Amat.applyDirichletBC( num, allocatedLevel_ );
         }

         KSPSetOperators( ksp, Amat.get(), Amat.get() );
         KSPGetPC( ksp, &pc );

         if ( assumeSymmetry_ )
         {
            PCSetType( pc, PCCHOLESKY );
         }
         else
         {
            PCSetType( pc, PCLU );
         }

         MatSolverType petscSolverType;
         switch ( solverType_ )
         {
         case PETScDirectSolverType::MUMPS:
#ifdef PETSC_HAVE_MUMPS
            petscSolverType = MATSOLVERMUMPS;
            break;
#else
            WALBERLA_ABORT( "PETSc is not build with MUMPS support." )
#endif
         case PETScDirectSolverType::SUPER_LU:
            petscSolverType = MATSOLVERSUPERLU_DIST;
            break;
         default:
            WALBERLA_ABORT( "Invalid PETSc solver type." )
         }
         HYTEG_PCFactorSetMatSolverType( pc, petscSolverType );

         if ( solverType_ == PETScDirectSolverType::MUMPS )
         {
            PCFactorSetUpMatSolverType( pc );
            PCFactorGetMatrix( pc, &F );
#ifdef PETSC_HAVE_MUMPS
            for ( auto it : mumpsIcntrl_ )
            {
               MatMumpsSetIcntl( F, it.first, it.second );
            }
            for ( auto it : mumpsCntrl_ )
            {
               MatMumpsSetCntl( F, it.first, it.second );
            }
#endif
         }
         storage_->getTimingTree()->start( "Factorization" );
         PCSetUp( pc );
         storage_->getTimingTree()->stop( "Factorization" );
      }
   }

   void solve( const OperatorType& A, const FunctionType& x, const FunctionType& b, const uint_t level )
   {
      WALBERLA_CHECK_EQUAL( level, allocatedLevel_ );

      walberla::WcTimer timer;

      storage_->getTimingTree()->start( "PETSc LU Solver" );
      storage_->getTimingTree()->start( "Setup" );

      timer.start();
      if ( !manualAssemblyAndFactorization_ )
      {
         assembleAndFactorize( A );
      }
      timer.end();
      const double matrixAssemblyAndFactorizationTime = timer.last();

      storage_->getTimingTree()->start( "RHS vector setup" );

      b.assign( { 1.0 }, { x }, level, DirichletBoundary );
      bVec.createVectorFromFunction( b, num, level, All );
      xVec.createVectorFromFunction( x, num, level, All );

      if ( assumeSymmetry_ )
      {
         AmatTmp.zeroEntries();
         MatCopy( AmatUnsymmetric.get(), AmatTmp.get(), DIFFERENT_NONZERO_PATTERN );
         AmatTmp.applyDirichletBCSymmetrically( x, num, bVec, allocatedLevel_ );
      }

      storage_->getTimingTree()->stop( "RHS vector setup" );

      storage_->getTimingTree()->stop( "Setup" );

      storage_->getTimingTree()->start( "Solver" );
      timer.start();
      KSPSolve( ksp, bVec.get(), xVec.get() );
      timer.end();
      const double petscKSPTimer = timer.last();
      storage_->getTimingTree()->stop( "Solver" );

      xVec.createFunctionFromVector( x, num, level, flag_ );

      if ( verbose_ )
      {
         WALBERLA_LOG_INFO_ON_ROOT( "[PETScLUSolver] "
                                    << "PETSc KSPSolver time: " << petscKSPTimer
                                    << ", assembly and fact time: " << matrixAssemblyAndFactorizationTime );
      }

      storage_->getTimingTree()->stop( "PETSc LU Solver" );
   }

 private:
   std::shared_ptr< PrimitiveStorage >                                                           storage_;
   uint_t                                                                                        allocatedLevel_;
   MPI_Comm                                                                                      petscCommunicator_;
   typename OperatorType::srcType::template FunctionType< idx_t >                                num;
   PETScSparseMatrix< OperatorType >                                                             Amat;
   PETScSparseMatrix< OperatorType >                                                             AmatUnsymmetric;
   PETScSparseMatrix< OperatorType >                                                             AmatTmp;
   PETScVector< typename FunctionType::valueType, OperatorType::srcType::template FunctionType > xVec;
   PETScVector< typename FunctionType::valueType, OperatorType::dstType::template FunctionType > bVec;
#if 0
  PETScVector<typename FunctionType::valueType, OperatorType::srcType::template FunctionType> inKernel;
#endif

   KSP                        ksp;
   PC                         pc;
   hyteg::DoFType             flag_;
   bool                       verbose_;
   Mat                        F; //factored Matrix
   bool                       manualAssemblyAndFactorization_;
   bool                       reassembleMatrix_;
   bool                       assumeSymmetry_;
   PETScDirectSolverType      solverType_;
   std::map< uint_t, int >    mumpsIcntrl_;
   std::map< uint_t, real_t > mumpsCntrl_;
};

} // namespace hyteg

#endif
