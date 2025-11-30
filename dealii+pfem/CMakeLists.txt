# CMakeLists.txt for PFEM 2D Sloshing with deal.II
#
# Usage:
#   mkdir build && cd build
#   cmake -DDEAL_II_DIR=/path/to/dealii ..
#   make
#
# Or using CMake presets:
#   cmake --preset=default
#   cmake --build build

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

# Set the project name and version
PROJECT(PFEM2D VERSION 1.0.0 LANGUAGES CXX)

# Find deal.II package
FIND_PACKAGE(deal.II 9.4.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
ENDIF()

# Check for required deal.II configuration
IF(NOT DEAL_II_WITH_CGAL)
  MESSAGE(WARNING "\n"
    "*** deal.II was not configured with CGAL support. ***\n"
    "*** The alpha-shape functionality may be limited. ***\n"
    "*** A simplified triangulation will be used instead. ***\n"
  )
ENDIF()

# Initialize cached variables
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable for the main PFEM sloshing simulation
ADD_EXECUTABLE(pfem_sloshing pfem_sloshing.cc)
DEAL_II_SETUP_TARGET(pfem_sloshing)

# Add executable for the simplified version (no CGAL dependency)
ADD_EXECUTABLE(pfem_sloshing_simple pfem_sloshing_simple.cc)
DEAL_II_SETUP_TARGET(pfem_sloshing_simple)

# Find CGAL (optional, for full alpha-shape support)
FIND_PACKAGE(CGAL QUIET)
IF(CGAL_FOUND)
  MESSAGE(STATUS "CGAL found: ${CGAL_VERSION}")
  TARGET_LINK_LIBRARIES(pfem_sloshing CGAL::CGAL)
  TARGET_COMPILE_DEFINITIONS(pfem_sloshing PRIVATE USE_CGAL=1)
ELSE()
  MESSAGE(WARNING "CGAL not found. Using simplified triangulation in pfem_sloshing.")
  TARGET_COMPILE_DEFINITIONS(pfem_sloshing PRIVATE USE_CGAL=0)
ENDIF()

# Custom target for running the simulation
ADD_CUSTOM_TARGET(run
  COMMAND pfem_sloshing_simple
  DEPENDS pfem_sloshing_simple
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Running PFEM sloshing simulation..."
)

# Custom target for cleaning output files
ADD_CUSTOM_TARGET(clean-output
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/output
  COMMENT "Cleaning output directory..."
)

# Install targets
INSTALL(TARGETS pfem_sloshing pfem_sloshing_simple
  RUNTIME DESTINATION bin
)

# Print configuration summary
MESSAGE(STATUS "")
MESSAGE(STATUS "PFEM2D Configuration Summary")
MESSAGE(STATUS "============================")
MESSAGE(STATUS "deal.II version: ${DEAL_II_PACKAGE_VERSION}")
MESSAGE(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
IF(CGAL_FOUND)
  MESSAGE(STATUS "CGAL: ${CGAL_VERSION}")
ELSE()
  MESSAGE(STATUS "CGAL: Not found (using simplified version)")
ENDIF()
MESSAGE(STATUS "")
