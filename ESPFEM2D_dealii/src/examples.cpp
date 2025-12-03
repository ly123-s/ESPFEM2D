/**
 * ESPFEM2D - Example Setup Functions
 * 
 * Based on:
 * Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
 * particle finite element method code for geotechnical large deformation analysis[J].
 * Computational Mechanics, 2024, 74(2):467-484.
 * 
 * Examples:
 * 1 - Oscillation of an elastic cantilever beam
 * 2 - Non-cohesive soil collapse stage1
 * 3 - Non-cohesive soil collapse stage2
 * 4 - Cohesive soil collapse stage1
 * 5 - Cohesive soil collapse stage2
 * 6 - Failure of a Mohr-Coulomb soil slope stage1
 * 7 - Failure of a Mohr-Coulomb soil slope stage2
 */

#include "spfem_solver.h"
#include <cmath>
#include <fstream>
#include <iostream>

namespace ESPFEM2D
{
    //==========================================================================
    // Example 1: Oscillation of an elastic cantilever beam
    //==========================================================================
    void setup_bar_gravity_vibration(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        // Parameters
        par.project = "bar_gravity_vibration";
        par.dir_plus = "initial";
        par.totaltime = 4.0;
        par.dtime = 1e-5;
        par.rank_deficiency = 0.1;
        par.MN = 16;
        par.vtk_output_interval = 4000;
        par.damping = 0.0;
        par.rigid_wall_count = 0;
        
        // Mesh parameters
        par.if_remesh = 0;
        par.alpha_shape = {{1, 1.5, -1000, 1000, -1000, 1000}};
        par.remesh_threshold = 20;
        par.max_remesh_threshold = 20;
        par.min_remesh_threshold = 5;
        par.remesh_quality_space = 5;
        par.Lap_threshold = 0.2;
        par.remesh_cnt = 0;
        
        // Material parameters
        par.mat_gravity = {{0.0, -9.8}};
        par.mat_props = {{1850, 80.0e6, 0.25, 0.0, 0.0, 0.0}};
        par.mat_model = {1};  // Elastic
        
        // Computational mesh
        double xlen = 2.0;
        double ylen = 0.2;
        par.space = 0.2 / std::round(0.2 / 0.02);
        
        // Create regular grid
        int nx = static_cast<int>(std::round(xlen / par.space)) + 1;
        int ny = static_cast<int>(std::round(ylen / par.space)) + 1;
        int pNN = nx * ny;
        par.node_cnt = pNN;
        
        nodal.coordinates.resize(pNN);
        nodal.displacements.resize(pNN, Point(0, 0));
        nodal.velocities.resize(pNN, Point(0, 0));
        nodal.accelerations.resize(pNN, Point(0, 0));
        nodal.stresses.resize(pNN, Tensor());
        nodal.plastic_strain_eq.resize(pNN, 0.0);
        nodal.material_id.resize(pNN, 1);
        nodal.boundary_type.resize(pNN);
        nodal.boundary_value.resize(pNN, Point(0, 0));
        nodal.deformation_gradient.resize(pNN, Tensor());
        
        int cnt = 0;
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                nodal.coordinates[cnt] = Point(j * par.space, i * par.space);
                cnt++;
            }
        }
        
        // Boundary conditions
        for (int in = 0; in < pNN; ++in)
        {
            nodal.boundary_type[in] = {BoundaryType::Free, BoundaryType::Free};
            double curx = nodal.coordinates[in][0];
            
            if (curx < 1e-6)  // Left edge - fixed
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
                nodal.boundary_type[in][1] = BoundaryType::Velocity;
                nodal.boundary_value[in][1] = 0.0;
            }
        }
        
        // Monitor setup
        int nstep = static_cast<int>(std::ceil(par.totaltime / par.dtime));
        mon.records.resize(nstep + 1);
        mon.items = {"nVel", "nVel", "nDisp", "nDisp", "nStress", "nStress", "nStress", "nStress"};
        mon.node_ids = {pNN, pNN, pNN, pNN, 1, 1, nx, nx};  // Monitor tip and root
        mon.columns = {1, 2, 1, 2, 1, 2, 1, 2};
    }

    //==========================================================================
    // Helper function to generate soil column nodes
    //==========================================================================
    void generate_non_cohesive_soil_nodes(std::vector<Point>& nodes, double space)
    {
        // Non-cohesive soil: column 0.2m wide, 0.2m high
        double xlen = 0.2;
        double ylen = 0.2;
        
        int nx = static_cast<int>(std::round(xlen / space)) + 1;
        int ny = static_cast<int>(std::round(ylen / space)) + 1;
        
        nodes.clear();
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                nodes.push_back(Point(j * space, i * space));
            }
        }
    }

    void generate_cohesive_soil_nodes(std::vector<Point>& nodes, double space)
    {
        // Cohesive soil: rectangle 4m wide, 1m high (approximate)
        double xlen = 4.0;
        double ylen = 1.0;
        
        int nx = static_cast<int>(std::round(xlen / space)) + 1;
        int ny = static_cast<int>(std::round(ylen / space)) + 1;
        
        nodes.clear();
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                nodes.push_back(Point(j * space, i * space));
            }
        }
    }

    void generate_slope_nodes(std::vector<Point>& nodes, double space)
    {
        // Slope geometry: trapezoidal shape
        // Base: 60m, height varies from 20m on left to 10m on right
        nodes.clear();
        
        double xlen = 60.0;
        
        // Create nodes row by row
        for (double y = 0.0; y <= 20.0; y += space)
        {
            // Calculate x range for this y level
            double x_start = 0.0;
            double x_end = xlen;
            
            // Slope on right side starts at x=30, from y=10 to y=20
            if (y > 10.0)
            {
                // Linear slope: at y=10, x_max=60; at y=20, x_max=45
                x_end = 45.0 + (60.0 - 45.0) * (20.0 - y) / 10.0;
            }
            
            for (double x = x_start; x <= x_end; x += space)
            {
                nodes.push_back(Point(x, y));
            }
        }
    }

    //==========================================================================
    // Example 2: Non-cohesive soil collapse stage 1 (gravity loading)
    //==========================================================================
    void setup_non_cohesive_soil_stage1(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        par.project = "non_cohesive_soil";
        par.dir_plus = "stage1";
        par.totaltime = 0.6;
        par.dtime = 1e-6;
        par.rank_deficiency = 0.1;
        par.MN = 16;
        par.vtk_output_interval = 1000;
        par.damping = 0.7;
        par.rigid_wall_count = 1;
        par.walls = {{0.2, 0.0, 1.0, 0.0, 10.0}};  // Ground wall with high friction
        
        // Mesh parameters
        par.if_remesh = 1;
        par.alpha_shape = {{1, 2.5, -1000, 1000, -1000, 1000}};
        par.remesh_threshold = 20;
        par.max_remesh_threshold = 20;
        par.min_remesh_threshold = 5;
        par.remesh_quality_space = 5;
        par.Lap_threshold = 0.2;
        par.remesh_cnt = 0;
        
        // Material parameters (sand)
        par.mat_gravity = {{0.0, -10.0}};
        par.mat_props = {{2650, 5.84e6, 0.3, 0.0, 19.8, 0.0}};  // Non-cohesive, phi=19.8°
        par.mat_model = {1};  // Elastic for stage 1
        
        par.space = 0.002;
        
        // Generate mesh nodes
        generate_non_cohesive_soil_nodes(nodal.coordinates, par.space);
        int pNN = nodal.coordinates.size();
        par.node_cnt = pNN;
        
        // Initialize arrays
        nodal.displacements.resize(pNN, Point(0, 0));
        nodal.velocities.resize(pNN, Point(0, 0));
        nodal.accelerations.resize(pNN, Point(0, 0));
        nodal.stresses.resize(pNN, Tensor());
        nodal.plastic_strain_eq.resize(pNN, 0.0);
        nodal.material_id.resize(pNN, 1);
        nodal.boundary_type.resize(pNN);
        nodal.boundary_value.resize(pNN, Point(0, 0));
        nodal.deformation_gradient.resize(pNN, Tensor());
        
        // Boundary conditions
        for (int in = 0; in < pNN; ++in)
        {
            nodal.boundary_type[in] = {BoundaryType::Free, BoundaryType::Free};
            double curx = nodal.coordinates[in][0];
            double cury = nodal.coordinates[in][1];
            
            if (cury < 1e-6)  // Bottom - fixed
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
                nodal.boundary_type[in][1] = BoundaryType::Velocity;
                nodal.boundary_value[in][1] = 0.0;
            }
            if (curx < 1e-6 || curx > 0.2 - 1e-6)  // Left/right walls
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
            }
        }
        
        // Monitor setup
        int nstep = static_cast<int>(std::ceil(par.totaltime / par.dtime));
        mon.records.resize(nstep + 1);
        mon.items = {"nVel", "nDisp", "nStress", "nStrain", "nVel", "nDisp", "nStress", "nStrain"};
        mon.node_ids = {102, 102, 102, 102, 102, 102, 102, 102};
        mon.columns = {1, 1, 1, 1, 2, 2, 2, 2};
    }

    //==========================================================================
    // Example 3: Non-cohesive soil collapse stage 2 (collapse)
    //==========================================================================
    void setup_non_cohesive_soil_stage2(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        // Start with stage1 setup
        setup_non_cohesive_soil_stage1(par, nodal, mon);
        
        // Modify for stage 2
        par.dir_plus = "stage2";
        par.damping = 0.0;  // No damping for collapse
        par.mat_model = {2};  // Drucker-Prager for plasticity
        
        // Remove right wall constraint for collapse
        for (int in = 0; in < par.node_cnt; ++in)
        {
            double curx = nodal.coordinates[in][0];
            double cury = nodal.coordinates[in][1];
            
            if (curx > 0.2 - 1e-6 && cury > 1e-6)  // Right wall, not bottom
            {
                nodal.boundary_type[in][0] = BoundaryType::Free;
            }
        }
        
        // Note: In real implementation, initial stress from stage1 should be loaded
        // For this implementation, we assume zero initial stress (simplified)
    }

    //==========================================================================
    // Example 4: Cohesive soil collapse stage 1 (gravity loading)
    //==========================================================================
    void setup_cohesive_soil_stage1(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        par.project = "cohesive_soil";
        par.dir_plus = "stage1";
        par.totaltime = 2.5;
        par.dtime = 5e-5;
        par.rank_deficiency = 0.1;
        par.MN = 16;
        par.vtk_output_interval = 200;
        par.damping = 0.7;
        par.rigid_wall_count = 1;
        par.walls = {{4.0, 0.0, 10.0, 0.0, 10.0}};  // Ground wall
        
        // Mesh parameters
        par.if_remesh = 1;
        par.alpha_shape = {{1, 1.5, -1000, 1000, -1000, 1000}};
        par.remesh_threshold = 20;
        par.max_remesh_threshold = 20;
        par.min_remesh_threshold = 5;
        par.remesh_quality_space = 5;
        par.Lap_threshold = 0.2;
        par.remesh_cnt = 0;
        
        // Material parameters (cohesive soil)
        par.mat_gravity = {{0.0, -10.0}};
        par.mat_props = {{1850, 1.8e6, 0.2, 5.0e3, 25.0, 0.0}};  // c=5kPa, phi=25°
        par.mat_model = {1};  // Elastic for stage 1
        
        par.space = 0.02;
        
        // Generate mesh nodes
        generate_cohesive_soil_nodes(nodal.coordinates, par.space);
        int pNN = nodal.coordinates.size();
        par.node_cnt = pNN;
        
        // Initialize arrays
        nodal.displacements.resize(pNN, Point(0, 0));
        nodal.velocities.resize(pNN, Point(0, 0));
        nodal.accelerations.resize(pNN, Point(0, 0));
        nodal.stresses.resize(pNN, Tensor());
        nodal.plastic_strain_eq.resize(pNN, 0.0);
        nodal.material_id.resize(pNN, 1);
        nodal.boundary_type.resize(pNN);
        nodal.boundary_value.resize(pNN, Point(0, 0));
        nodal.deformation_gradient.resize(pNN, Tensor());
        
        // Boundary conditions
        for (int in = 0; in < pNN; ++in)
        {
            nodal.boundary_type[in] = {BoundaryType::Free, BoundaryType::Free};
            double curx = nodal.coordinates[in][0];
            double cury = nodal.coordinates[in][1];
            
            if (cury < 1e-6)  // Bottom - fixed
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
                nodal.boundary_type[in][1] = BoundaryType::Velocity;
                nodal.boundary_value[in][1] = 0.0;
            }
            if (curx < 1e-6 || curx > 4.0 - 1e-6)  // Left/right walls
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
            }
        }
        
        // Monitor setup
        int nstep = static_cast<int>(std::ceil(par.totaltime / par.dtime));
        mon.records.resize(nstep + 1);
        mon.items = {"nVel", "nDisp", "nStress", "nStrain", "nVel", "nDisp", "nStress", "nStrain"};
        mon.node_ids = {202, 202, 202, 202, 202, 202, 202, 202};
        mon.columns = {1, 1, 1, 1, 2, 2, 2, 2};
    }

    //==========================================================================
    // Example 5: Cohesive soil collapse stage 2 (collapse)
    //==========================================================================
    void setup_cohesive_soil_stage2(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        // Start with stage1 setup
        setup_cohesive_soil_stage1(par, nodal, mon);
        
        // Modify for stage 2
        par.dir_plus = "stage2";
        par.damping = 0.0;
        par.mat_model = {2};  // Drucker-Prager
        
        // Remove constraints for collapse
        for (int in = 0; in < par.node_cnt; ++in)
        {
            double curx = nodal.coordinates[in][0];
            
            // Left edge fully fixed, right edge released for collapse
            if (curx > 4.0 - 1e-6)
            {
                nodal.boundary_type[in][0] = BoundaryType::Free;
            }
            
            // Keep left edge fully constrained
            if (curx < 1e-6)
            {
                nodal.boundary_type[in][1] = BoundaryType::Velocity;
                nodal.boundary_value[in][1] = 0.0;
            }
        }
    }

    //==========================================================================
    // Example 6: Slope failure stage 1 (gravity loading)
    //==========================================================================
    void setup_slope_stage1(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        par.project = "slope";
        par.dir_plus = "stage1";
        par.totaltime = 2.0;
        par.dtime = 1e-4;
        par.rank_deficiency = 0.1;
        par.MN = 16;
        par.vtk_output_interval = 200;
        par.damping = 0.7;
        par.rigid_wall_count = 0;  // No rigid walls for slope
        
        // Mesh parameters
        par.if_remesh = 1;
        par.alpha_shape = {{1, 1.5, -1000, 1000, -1000, 1000}};
        par.remesh_threshold = 20;
        par.max_remesh_threshold = 20;
        par.min_remesh_threshold = 5;
        par.remesh_quality_space = 5;
        par.Lap_threshold = 0.2;
        par.remesh_cnt = 0;
        
        // Material parameters
        par.mat_gravity = {{0.0, -10.0}};
        par.mat_props = {{2000, 1.0e8, 0.3, 10.0e3, 20.0, 0.0}};  // c=10kPa, phi=20°
        par.mat_model = {1};  // Elastic for stage 1
        
        par.space = 0.3;
        
        // Generate slope mesh nodes
        generate_slope_nodes(nodal.coordinates, par.space);
        int pNN = nodal.coordinates.size();
        par.node_cnt = pNN;
        
        // Initialize arrays
        nodal.displacements.resize(pNN, Point(0, 0));
        nodal.velocities.resize(pNN, Point(0, 0));
        nodal.accelerations.resize(pNN, Point(0, 0));
        nodal.stresses.resize(pNN, Tensor());
        nodal.plastic_strain_eq.resize(pNN, 0.0);
        nodal.material_id.resize(pNN, 1);
        nodal.boundary_type.resize(pNN);
        nodal.boundary_value.resize(pNN, Point(0, 0));
        nodal.deformation_gradient.resize(pNN, Tensor());
        
        // Boundary conditions
        for (int in = 0; in < pNN; ++in)
        {
            nodal.boundary_type[in] = {BoundaryType::Free, BoundaryType::Free};
            double curx = nodal.coordinates[in][0];
            double cury = nodal.coordinates[in][1];
            
            if (cury < 1e-5)  // Bottom - fixed
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
                nodal.boundary_type[in][1] = BoundaryType::Velocity;
                nodal.boundary_value[in][1] = 0.0;
            }
            if (curx < 1e-6 || curx > 60.0 - 1e-6)  // Left/right boundaries
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
            }
        }
        
        // Monitor setup
        int nstep = static_cast<int>(std::ceil(par.totaltime / par.dtime));
        mon.records.resize(nstep + 1);
        mon.items = {"nVel", "nDisp", "nStress", "nStrain", "nVel", "nDisp", "nStress", "nStrain"};
        mon.node_ids = {286, 286, 286, 286, 286, 286, 286, 286};
        mon.columns = {1, 1, 1, 1, 2, 2, 2, 2};
    }

    //==========================================================================
    // Example 7: Slope failure stage 2 (failure)
    //==========================================================================
    void setup_slope_stage2(Parameters& par, NodalData& nodal, Monitor& mon)
    {
        // Start with stage1 setup  
        setup_slope_stage1(par, nodal, mon);
        
        // Modify for stage 2
        par.dir_plus = "stage2";
        par.totaltime = 5.0;
        par.vtk_output_interval = 100;
        par.damping = 0.0;
        par.mat_model = {2};  // Drucker-Prager
        
        // Adjust right boundary for failure analysis
        for (int in = 0; in < par.node_cnt; ++in)
        {
            double curx = nodal.coordinates[in][0];
            
            // Move right boundary constraint from 60m to 45m
            if (curx > 45.0 - 1e-6 && curx < 60.0)
            {
                nodal.boundary_type[in][0] = BoundaryType::Velocity;
                nodal.boundary_value[in][0] = 0.0;
            }
        }
        
        // Update monitor
        mon.node_ids = {213, 213, 213, 213, 213, 213, 213, 213};
    }

} // namespace ESPFEM2D
