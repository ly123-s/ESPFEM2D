# PFEM2D - Particle Finite Element Method for 2D Free Surface Flows

This folder contains a deal.II-based implementation of the Particle Finite Element Method (PFEM) for simulating 2D free surface sloshing problems. The implementation is based on the ESPFEM2D methodology developed by Zhang Wei, Liu Yihui, and Yuan Weihai.

## Overview

The Particle Finite Element Method (PFEM) is a Lagrangian approach for solving problems with large deformations and free surface evolution. Key features of this implementation include:

- **Lagrangian mesh formulation**: Nodes (particles) move with the material
- **Delaunay triangulation**: Automatic mesh generation
- **Alpha-shape boundary recognition**: Free surface detection
- **Explicit time integration**: Leapfrog/Verlet scheme
- **VTK output**: Visualization with ParaView or similar tools

## Files

- `pfem_sloshing_simple.cc` - Main PFEM implementation (no external dependencies besides deal.II)
- `pfem_sloshing.cc` - Extended version with optional CGAL support for improved alpha-shape
- `CMakeLists.txt` - Build configuration
- `sloshing.prm` - Parameter file for the sloshing example
- `README.md` - This file

## Dependencies

### Required
- **deal.II** (version 9.4 or later) - Finite element library
- **CMake** (version 3.13.4 or later) - Build system

### Optional
- **CGAL** - Computational geometry library (for improved alpha-shape computation)

## Building

### Standard build

```bash
mkdir build
cd build
cmake -DDEAL_II_DIR=/path/to/dealii ..
make
```

### With CGAL support

```bash
cmake -DDEAL_II_DIR=/path/to/dealii -DCGAL_DIR=/path/to/cgal ..
make
```

## Running

### Using default parameters

```bash
./pfem_sloshing_simple
```

### Using parameter file

```bash
./pfem_sloshing_simple ../sloshing.prm
```

### Custom parameters

You can modify the `sloshing.prm` file to change simulation parameters:

```
# Physical parameters
set Density = 1000.0      # Water density [kg/m³]
set Viscosity = 0.001     # Dynamic viscosity [Pa·s]

# Domain parameters
set TankLength = 1.0      # Tank length [m]
set TankHeight = 0.6      # Tank height [m]

# Initial water column
set InitialWaterLength = 0.4  # Initial width [m]
set InitialWaterHeight = 0.3  # Initial height [m]
```

## Output

Results are written to the `output/` directory as VTK files:
- `sloshing_000000.vtk`, `sloshing_000100.vtk`, etc.

### Visualization

Open the VTK files in ParaView:

1. Open ParaView
2. File → Open → select `sloshing_*.vtk`
3. Click "Apply"
4. Select "velocity" or "velocity_magnitude" to visualize the flow

To create an animation:
1. Select all VTK files
2. Play button to animate
3. File → Save Animation for video export

## Problem Description

### 2D Dam Break / Sloshing Test

The default example simulates a dam break problem in a rectangular tank:

```
+------------------------+
|                        |
|                        |
|  +--------+            |
|  |        |            |  Initial water column
|  | Water  |            |  on the left side
|  |        |            |
+--+--------+------------+
   0       0.4          1.0
```

Initial conditions:
- Tank dimensions: 1.0 m × 0.6 m
- Initial water column: 0.4 m × 0.3 m (left side)
- Gravity: -9.81 m/s² (downward)

The water column collapses under gravity and flows across the tank, demonstrating free surface evolution and sloshing behavior.

## Method Description

### PFEM Algorithm

The PFEM algorithm follows these steps at each time step:

1. **Remeshing check**: Evaluate mesh quality, remesh if needed
2. **Delaunay triangulation**: Generate triangular mesh from particles
3. **Alpha-shape**: Remove external triangles to identify fluid boundary
4. **Compute element data**: Calculate shape function derivatives
5. **Compute nodal data**: Smooth particle FEM approach for nodal quantities
6. **Force calculation**: Internal (viscous) and external (gravity) forces
7. **Time integration**: Update velocities and positions
8. **Boundary conditions**: Apply wall contacts

### Alpha-Shape Boundary Recognition

The alpha-shape algorithm filters triangles based on their circumradius:
- Triangles with circumradius > α × h are removed
- Where α is the alpha factor (default 1.2) and h is the mesh size
- This naturally identifies the free surface boundary

### Smoothed Particle FEM (SPFEM)

Following the ESPFEM2D approach:
- Quantities are computed at nodes rather than elements
- Shape function derivatives are averaged over connected elements
- Nodal area = sum of 1/3 of connected element areas
- This approach is more stable for large deformations

## References

1. Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed particle finite element method code for geotechnical large deformation analysis[J]. Computational Mechanics, 2024, 74(2):467-484.

2. Oñate E, Idelsohn SR, Del Pin F, Aubry R. The particle finite element method—an overview. International Journal of Computational Methods. 2004;1(02):267-307.

3. Idelsohn SR, Oñate E, Del Pin F. The particle finite element method: a powerful tool to solve incompressible flows with free-surfaces and breaking waves. International Journal for Numerical Methods in Engineering. 2004;61(7):964-989.

## License

This code is provided for educational and research purposes. Please cite the original ESPFEM2D paper when using this implementation.

## Contact

For questions about this implementation, please refer to the main ESPFEM2D repository.
