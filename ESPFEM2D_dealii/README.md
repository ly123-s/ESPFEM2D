# ESPFEM2D - deal.II-style C++ Implementation

A C++ implementation of the Smoothed Particle Finite Element Method (SPFEM) for 2D geotechnical large deformation analysis, following deal.II patterns and conventions.

## Original Work

Based on:
> Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed particle finite element method code for geotechnical large deformation analysis[J]. Computational Mechanics, 2024, 74(2):467-484.

All copyrights reserved Zhang Wei(1), Liu Yihui(1), Yuan Weihai(2)
1. South China Agricultural University, Guangzhou, China
2. Hohai University, Nanjing, China

## Features

- Explicit time integration using leapfrog scheme
- Node-based smoothed finite element formulation
- Adaptive remeshing with alpha-shape boundary detection
- Rank deficiency (hourglass) control
- Contact with rigid walls (frictional)
- VTK output for visualization with ParaView

### Constitutive Models
- Linear Elastic
- Drucker-Prager (with Mohr-Coulomb matching)

### Examples
1. Oscillation of an elastic cantilever beam
2. Non-cohesive soil collapse - Stage 1 (gravity loading)
3. Non-cohesive soil collapse - Stage 2 (collapse)
4. Cohesive soil collapse - Stage 1 (gravity loading)
5. Cohesive soil collapse - Stage 2 (collapse)
6. Slope failure - Stage 1 (gravity loading)
7. Slope failure - Stage 2 (failure)

## Requirements

- CMake 3.10 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+)

Note: This implementation is self-contained and does not require deal.II library, 
but follows deal.II design patterns for familiarity and future integration.

## Building

```bash
cd ESPFEM2D_dealii
mkdir build
cd build
cmake ..
make
```

## Running

```bash
# Run example 1 (cantilever beam)
./espfem2d 1

# Run example 2 (non-cohesive soil collapse stage 1)
./espfem2d 2

# Run any example (1-7)
./espfem2d <example_id>
```

## Output

VTK files are written to `output/<project_name>/<stage>/` directory and can be visualized with ParaView or similar software.

The output includes:
- Nodal displacements
- Nodal velocities  
- Stress tensor
- Equivalent plastic strain
- Material ID

## Code Structure

```
ESPFEM2D_dealii/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/
│   └── spfem_solver.h      # Main header with classes and structures
├── src/
│   ├── main.cpp            # Entry point
│   ├── spfem_solver.cpp    # Main solver implementation
│   ├── constitutive_model.cpp  # Material models
│   ├── mesh_handler.cpp    # Mesh utilities
│   ├── vtk_output.cpp      # VTK output
│   └── examples.cpp        # Example setup functions
└── data/                   # Mesh data files (if needed)
```

## Algorithm Overview

1. **Initialization**: Set up mesh, boundary conditions, and material properties
2. **Time Integration Loop**:
   - Output VTK at specified intervals
   - Remesh if mesh quality degrades (adaptive)
   - Compute element and nodal quantities (smoothed gradients)
   - Calculate internal forces
   - Apply rank deficiency control
   - Update velocities and positions (leapfrog)
   - Handle contact with rigid walls
3. **Post-processing**: Visualize results with ParaView

## Correspondence with MATLAB Code

| MATLAB File | C++ Implementation |
|-------------|-------------------|
| `SPFEM.m` | `spfem_solver.cpp` - `run()` |
| `initializing.m` | `spfem_solver.cpp` - `setup_mesh()` |
| `element_data_prepare.m` | `spfem_solver.cpp` - `compute_element_data()` |
| `node_data_prepare.m` | `spfem_solver.cpp` - `compute_nodal_data()` |
| `force_int.m` | `spfem_solver.cpp` - `compute_internal_forces()` |
| `force_rank_deficiency.m` | `spfem_solver.cpp` - `compute_rank_deficiency_forces()` |
| `time_integration.m` | `spfem_solver.cpp` - `time_integration()` |
| `contact_wall.m` | `spfem_solver.cpp` - `apply_contact_wall()` |
| `mesh_remesh.m` | `spfem_solver.cpp` - `remesh_if_needed()` |
| `mesh_alpha_shape.m` | `spfem_solver.cpp` - `apply_alpha_shape()` |
| `mesh_lap_smoothing.m` | `spfem_solver.cpp` - `laplacian_smoothing()` |
| `mesh_quality.m` | `spfem_solver.cpp` - `compute_mesh_quality()` |
| `constitutive_model.m` | `constitutive_model.cpp` |
| `mat_model_elas.m` | `constitutive_model.cpp` - `elastic_model()` |
| `mat_model_DP.m` | `constitutive_model.cpp` - `drucker_prager_model()` |
| `output_vtk.m` | `vtk_output.cpp` - `output_vtk()` |
| `ex_*.m` | `examples.cpp` |

## License

This implementation follows the original ESPFEM2D copyright terms. See the original paper for details.
