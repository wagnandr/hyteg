target_sources( hyteg
    PRIVATE
    PrimitiveStorage.hpp
    SetupPrimitiveStorage.cpp
    SetupPrimitiveStorage.hpp
    Visualization.hpp
    PrimitiveStorage.cpp
    StoragePermutator.hpp
    StoragePermutator.cpp
)
add_subdirectory( loadbalancing )
