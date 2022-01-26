if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/output)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/csv)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/csv")
endif()

waLBerla_add_executable( NAME ILUSmoother3D
        FILES ILUSmoother3D.cpp
        DEPENDS hyteg core)

waLBerla_link_files_to_builddir( *.prm )
waLBerla_link_files_to_builddir( *.py )
waLBerla_link_files_to_builddir( *.m )
waLBerla_link_files_to_builddir( *.sh )
