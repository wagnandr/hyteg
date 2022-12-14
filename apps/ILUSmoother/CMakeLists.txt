if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/output)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/csv)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/csv")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/scripts)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/scripts")
endif()

waLBerla_add_executable( NAME ILUSmoother3D
        FILES ILUSmoother3D.cpp
        DEPENDS hyteg core)

waLBerla_add_executable( NAME CompareSurrogateILUStencils
        FILES CompareSurrogateILUStencils.cpp
        DEPENDS hyteg core)

waLBerla_add_executable( NAME ILUSmootherProf
        FILES ILUSmootherProfiling.cpp
        DEPENDS hyteg core)

waLBerla_add_executable( NAME ILUForwardSubstitutionProfiling
        FILES ILUForwardSubstitutionProfiling.cpp
        DEPENDS hyteg core)

waLBerla_add_executable( NAME ILUSketchbook
        FILES ILUSketchbook.cpp
        DEPENDS hyteg core)

waLBerla_add_executable( NAME ILUSketchbook_bak
        FILES ILUSketchbook_bak.cpp
        DEPENDS hyteg core)

waLBerla_add_executable( NAME ILUSmootherMatrixBasedProfiling
        FILES ILUSmootherMatrixBasedProfiling.cpp
        DEPENDS hyteg core)

waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.py )
waLBerla_link_files_to_builddir( *.m )
waLBerla_link_files_to_builddir( *.sh )
waLBerla_link_files_to_builddir( scripts/*.py )
waLBerla_link_files_to_builddir( scripts/*.prm )
