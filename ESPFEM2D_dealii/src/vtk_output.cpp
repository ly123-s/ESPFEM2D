/**
 * ESPFEM2D - VTK Output Implementation
 * 
 * Based on:
 * Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
 * particle finite element method code for geotechnical large deformation analysis[J].
 * Computational Mechanics, 2024, 74(2):467-484.
 */

#include "spfem_solver.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cerrno>

// Filesystem support - use experimental if standard not available
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
// Fallback for systems without filesystem - use POSIX mkdir
#include <sys/stat.h>
#include <sys/types.h>
namespace fs {
    inline void create_directories(const std::string& path) {
        // Simple recursive directory creation using mkdir
        std::string current;
        for (size_t i = 0; i < path.size(); ++i) {
            current += path[i];
            if (path[i] == '/' || i == path.size() - 1) {
                mkdir(current.c_str(), 0755);
            }
        }
    }
}
#endif

namespace ESPFEM2D
{
    void SPFEMSolver::output_vtk() const
    {
        // Create output directory
        std::string folder = "output/" + parameters.project + "/" + parameters.dir_plus;
        fs::create_directories(folder);
        
        // Generate filename
        std::ostringstream filename;
        filename << folder << "/" << parameters.project << std::setfill('0') << std::setw(6) << parameters.istep << ".vtk";
        
        std::ofstream fid(filename.str());
        if (!fid.is_open())
        {
            std::cerr << "Error: Could not open file " << filename.str() << " for writing.\n";
            return;
        }
        
        // VTK header
        fid << "# vtk DataFile Version 3.0\n";
        fid << "ESPFEM2D output - displacement\n";
        fid << "ASCII\n\n";
        
        // Points
        fid << "DATASET UNSTRUCTURED_GRID\n";
        fid << "POINTS " << parameters.node_cnt << " float\n";
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            fid << std::fixed << std::setprecision(6)
                << nodal_data.coordinates[in][0] << " "
                << nodal_data.coordinates[in][1] << " "
                << 0.0 << "\n";
        }
        
        // Cells
        fid << "\nCELLS " << parameters.element_cnt << " " << parameters.element_cnt * 4 << "\n";
        for (int ie = 0; ie < parameters.element_cnt; ++ie)
        {
            fid << "3 "
                << element_data.connectivity[ie][0] << " "
                << element_data.connectivity[ie][1] << " "
                << element_data.connectivity[ie][2] << "\n";
        }
        
        // Cell types (VTK_TRIANGLE = 5)
        fid << "\nCELL_TYPES " << parameters.element_cnt << "\n";
        for (int ie = 0; ie < parameters.element_cnt; ++ie)
        {
            fid << "5\n";
        }
        
        // Point data
        fid << "\nPOINT_DATA " << parameters.node_cnt << "\n";
        
        // Displacement vectors
        fid << "VECTORS nDisp float\n";
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            fid << std::fixed << std::setprecision(6)
                << nodal_data.displacements[in][0] << " "
                << nodal_data.displacements[in][1] << " "
                << 0.0 << "\n";
        }
        
        // Velocity vectors
        fid << "\nVECTORS nVel float\n";
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            fid << std::fixed << std::setprecision(6)
                << nodal_data.velocities[in][0] << " "
                << nodal_data.velocities[in][1] << " "
                << 0.0 << "\n";
        }
        
        // Stress tensor
        fid << "\nTENSORS nStress float\n";
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            // Write as 3x3 symmetric tensor
            // First row: sigma_xx, sigma_xy, 0
            // Second row: sigma_xy, sigma_yy, 0  
            // Third row: 0, 0, sigma_zz
            fid << std::fixed << std::setprecision(6)
                << nodal_data.stresses[in][0] << " "
                << nodal_data.stresses[in][2] << " "
                << 0.0 << "\n"
                << nodal_data.stresses[in][2] << " "
                << nodal_data.stresses[in][1] << " "
                << 0.0 << "\n"
                << 0.0 << " "
                << 0.0 << " "
                << nodal_data.stresses[in][3] << "\n\n";
        }
        
        // Equivalent plastic strain
        fid << "SCALARS npStrain_eq float\n";
        fid << "LOOKUP_TABLE default\n";
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            fid << std::fixed << std::setprecision(6)
                << nodal_data.plastic_strain_eq[in] << "\n";
        }
        
        // Material ID
        fid << "\nSCALARS nMat int\n";
        fid << "LOOKUP_TABLE default\n";
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            fid << nodal_data.material_id[in] << "\n";
        }
        
        fid.close();
    }

    void SPFEMSolver::save_monitor_data()
    {
        // Resize if necessary
        if (monitor.records.size() <= static_cast<size_t>(parameters.istep))
        {
            monitor.records.resize(parameters.istep + 1);
        }
        
        monitor.records[parameters.istep].resize(monitor.node_ids.size());
        
        for (size_t t = 0; t < monitor.items.size(); ++t)
        {
            int cur_id = monitor.node_ids[t] - 1;  // Convert to 0-based
            int cur_col = monitor.columns[t] - 1;
            
            if (cur_id < 0 || cur_id >= parameters.node_cnt)
            {
                continue;
            }
            
            if (monitor.items[t] == "nVel")
            {
                monitor.records[parameters.istep][t] = nodal_data.velocities[cur_id][cur_col];
            }
            else if (monitor.items[t] == "nDisp")
            {
                monitor.records[parameters.istep][t] = nodal_data.displacements[cur_id][cur_col];
            }
            else if (monitor.items[t] == "nStress")
            {
                monitor.records[parameters.istep][t] = nodal_data.stresses[cur_id][cur_col];
            }
        }
    }

} // namespace ESPFEM2D
