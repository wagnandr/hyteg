if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/output)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
endif()

waLBerla_link_files_to_builddir( *.prm )

waLBerla_add_executable( NAME MantleConvectionBlendedAnnulus
        FILES  MantleConvectionBlendedAnnulus.cpp
        DEPENDS hyteg convection_particles)

waLBerla_add_executable( NAME Benchmark_01_CircularAdvection
        FILES Benchmark_01_CircularAdvection.cpp Helpers.cpp
        DEPENDS hyteg convection_particles)

waLBerla_add_executable( NAME Benchmark_02_SwirlingAdvection
        FILES Benchmark_02_SwirlingAdvection.cpp Helpers.cpp
        DEPENDS hyteg convection_particles)

waLBerla_add_executable( NAME Benchmark_03_BlendedAdvection
        FILES Benchmark_03_BlendedAdvection.cpp Helpers.cpp
        DEPENDS hyteg convection_particles)

waLBerla_add_executable( NAME Benchmark_04_BlendedAdvectionDiffusion
        FILES Benchmark_04_BlendedAdvectionDiffusion.cpp Helpers.cpp
        DEPENDS hyteg convection_particles)