##
#  CMake script for ESPFEM2D 
#  
#  Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
#  particle finite element method code for geotechnical large deformation analysis[J].
#  Computational Mechanics, 2024, 74(2):467-484.
#  
#  C++ implementation inspired by deal.II patterns
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(ESPFEM2D CXX)

# Set C++ standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Include directories
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# Source files
SET(SOURCES
  src/main.cpp
  src/spfem_solver.cpp
  src/constitutive_model.cpp
  src/mesh_handler.cpp
  src/vtk_output.cpp
  src/examples.cpp
)

# Create executable
ADD_EXECUTABLE(espfem2d ${SOURCES})

# Link math library on Unix
IF(UNIX)
  TARGET_LINK_LIBRARIES(espfem2d m)
ENDIF()

# Copy mesh data files to build directory
FILE(GLOB MESH_DATA_FILES "${CMAKE_SOURCE_DIR}/data/*.txt")
IF(MESH_DATA_FILES)
  FILE(COPY ${MESH_DATA_FILES} DESTINATION ${CMAKE_BINARY_DIR}/data)
ENDIF()

# Create output directory
FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# Enable testing
ENABLE_TESTING()

# Add test for bar vibration (quickest example)
ADD_TEST(NAME test_bar_vibration 
         COMMAND espfem2d 1
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
SET_TESTS_PROPERTIES(test_bar_vibration PROPERTIES TIMEOUT 300)
