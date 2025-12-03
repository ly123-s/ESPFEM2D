##
##  ESPFEM2D - deal.II Implementation
##  CMake configuration file
##
##  Based on: Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit 
##  smoothed particle finite element method code for geotechnical large deformation 
##  analysis[J]. Computational Mechanics, 2024, 74(2):467-484.
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(ESPFEM2D_DEALII CXX)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find deal.II
FIND_PACKAGE(deal.II 9.0 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)

# Initialize deal.II
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Create output directories
FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# Main executable
ADD_EXECUTABLE(espfem2d
  src/main.cc
)

DEAL_II_SETUP_TARGET(espfem2d)

# Installation
INSTALL(TARGETS espfem2d DESTINATION bin)

# Test target for running all examples
ADD_CUSTOM_TARGET(run_tests
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 1
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 2
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 3
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 4
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 5
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 6
  COMMAND ${CMAKE_BINARY_DIR}/espfem2d 7
  DEPENDS espfem2d
  COMMENT "Running all ESPFEM2D examples"
)

# Individual test targets
FOREACH(i RANGE 1 7)
  ADD_CUSTOM_TARGET(run_case${i}
    COMMAND ${CMAKE_BINARY_DIR}/espfem2d ${i}
    DEPENDS espfem2d
    COMMENT "Running ESPFEM2D example case ${i}"
  )
ENDFOREACH()
