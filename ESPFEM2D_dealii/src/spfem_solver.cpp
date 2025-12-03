/**
 * ESPFEM2D - Main SPFEM Solver Implementation
 * 
 * Based on:
 * Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
 * particle finite element method code for geotechnical large deformation analysis[J].
 * Computational Mechanics, 2024, 74(2):467-484.
 */

#include "spfem_solver.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>

// Filesystem support - use experimental if standard not available
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
// Fallback for systems without filesystem
#include <sys/stat.h>
#include <sys/types.h>
namespace fs {
    inline bool create_directories(const std::string& path) {
        return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
    }
}
#endif

namespace ESPFEM2D
{
    SPFEMSolver::SPFEMSolver()
    {
    }

    void SPFEMSolver::initialize(int example_id)
    {
        std::cout << "Initializing example " << example_id << "...\n";

        switch (example_id)
        {
        case 1:
            setup_bar_gravity_vibration(parameters, nodal_data, monitor);
            break;
        case 2:
            setup_non_cohesive_soil_stage1(parameters, nodal_data, monitor);
            break;
        case 3:
            setup_non_cohesive_soil_stage2(parameters, nodal_data, monitor);
            break;
        case 4:
            setup_cohesive_soil_stage1(parameters, nodal_data, monitor);
            break;
        case 5:
            setup_cohesive_soil_stage2(parameters, nodal_data, monitor);
            break;
        case 6:
            setup_slope_stage1(parameters, nodal_data, monitor);
            break;
        case 7:
            setup_slope_stage2(parameters, nodal_data, monitor);
            break;
        default:
            throw std::runtime_error("Invalid example ID");
        }

        // Setup mesh and related data
        setup_mesh();
        
        std::cout << "Initialization complete.\n";
        std::cout << "  Nodes: " << parameters.node_cnt << "\n";
        std::cout << "  Elements: " << parameters.element_cnt << "\n";
        std::cout << "  Total time: " << parameters.totaltime << " s\n";
        std::cout << "  Time step: " << parameters.dtime << " s\n";
    }

    void SPFEMSolver::setup_mesh()
    {
        std::cout << "Setting up mesh...\n";
        
        // Initialize triangulation using Delaunay
        initialize_triangulation();
        
        // Apply alpha shape
        apply_alpha_shape();
        
        // Compute related elements and nodes
        compute_related_elements_and_nodes();
        
        // Allocate force vectors
        internal_forces.resize(parameters.node_cnt, Point(0, 0));
        rank_deficiency_forces.resize(parameters.node_cnt, Point(0, 0));
        
        // Initialize deformation gradient to identity
        for (int i = 0; i < parameters.node_cnt; ++i)
        {
            nodal_data.deformation_gradient[i][0] = 1.0;  // F11 = 1
            nodal_data.deformation_gradient[i][1] = 0.0;  // F12 = 0
            nodal_data.deformation_gradient[i][2] = 0.0;  // F21 = 0
            nodal_data.deformation_gradient[i][3] = 1.0;  // F22 = 1
        }
        
        std::cout << "Mesh setup complete.\n";
    }

    void SPFEMSolver::initialize_triangulation()
    {
        // Perform Delaunay triangulation
        // Implementation using simple incremental algorithm for 2D
        
        const int n = parameters.node_cnt;
        std::vector<std::array<int, 3>> triangles;
        
        // Simple Delaunay triangulation implementation
        // First, create a super-triangle that contains all points
        double xmin = 1e20, xmax = -1e20, ymin = 1e20, ymax = -1e20;
        for (int i = 0; i < n; ++i)
        {
            xmin = std::min(xmin, nodal_data.coordinates[i][0]);
            xmax = std::max(xmax, nodal_data.coordinates[i][0]);
            ymin = std::min(ymin, nodal_data.coordinates[i][1]);
            ymax = std::max(ymax, nodal_data.coordinates[i][1]);
        }
        
        double dx = xmax - xmin;
        double dy = ymax - ymin;
        double dmax = std::max(dx, dy);
        double xmid = (xmin + xmax) / 2.0;
        double ymid = (ymin + ymax) / 2.0;
        
        // Create super triangle vertices (virtual points)
        Point p0(xmid - 20 * dmax, ymid - dmax);
        Point p1(xmid, ymid + 20 * dmax);
        Point p2(xmid + 20 * dmax, ymid - dmax);
        
        // Temporary storage including super triangle vertices
        std::vector<Point> all_points = nodal_data.coordinates;
        all_points.push_back(p0);
        all_points.push_back(p1);
        all_points.push_back(p2);
        
        int n_super = n;  // Index of first super-triangle vertex
        
        // Start with super-triangle
        triangles.push_back({n_super, n_super + 1, n_super + 2});
        
        // Insert points one by one using Bowyer-Watson algorithm
        for (int i = 0; i < n; ++i)
        {
            const Point& pt = all_points[i];
            
            // Find all triangles whose circumcircle contains the point
            std::vector<int> bad_triangles;
            for (size_t t = 0; t < triangles.size(); ++t)
            {
                const auto& tri = triangles[t];
                
                // Compute circumcircle
                const Point& pa = all_points[tri[0]];
                const Point& pb = all_points[tri[1]];
                const Point& pc = all_points[tri[2]];
                
                double ax = pa[0], ay = pa[1];
                double bx = pb[0], by = pb[1];
                double cx = pc[0], cy = pc[1];
                
                double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
                if (std::abs(d) < 1e-16) continue;
                
                double ux = ((ax * ax + ay * ay) * (by - cy) + 
                             (bx * bx + by * by) * (cy - ay) + 
                             (cx * cx + cy * cy) * (ay - by)) / d;
                double uy = ((ax * ax + ay * ay) * (cx - bx) + 
                             (bx * bx + by * by) * (ax - cx) + 
                             (cx * cx + cy * cy) * (bx - ax)) / d;
                
                double r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
                double dist2 = (pt[0] - ux) * (pt[0] - ux) + (pt[1] - uy) * (pt[1] - uy);
                
                if (dist2 < r2)
                {
                    bad_triangles.push_back(t);
                }
            }
            
            // Find boundary edges of the polygon hole
            std::vector<std::pair<int, int>> polygon;
            for (int bt : bad_triangles)
            {
                const auto& tri = triangles[bt];
                for (int j = 0; j < 3; ++j)
                {
                    int e0 = tri[j];
                    int e1 = tri[(j + 1) % 3];
                    
                    // Check if this edge is shared with another bad triangle
                    bool shared = false;
                    for (int other : bad_triangles)
                    {
                        if (other == bt) continue;
                        const auto& other_tri = triangles[other];
                        int count = 0;
                        for (int k = 0; k < 3; ++k)
                        {
                            if (other_tri[k] == e0 || other_tri[k] == e1)
                                count++;
                        }
                        if (count == 2)
                        {
                            shared = true;
                            break;
                        }
                    }
                    
                    if (!shared)
                    {
                        polygon.push_back({e0, e1});
                    }
                }
            }
            
            // Remove bad triangles (in reverse order to maintain indices)
            std::sort(bad_triangles.rbegin(), bad_triangles.rend());
            for (int bt : bad_triangles)
            {
                triangles.erase(triangles.begin() + bt);
            }
            
            // Create new triangles from polygon edges to new point
            for (const auto& edge : polygon)
            {
                triangles.push_back({edge.first, edge.second, i});
            }
        }
        
        // Remove triangles that share vertices with super-triangle
        std::vector<std::array<int, 3>> final_triangles;
        for (const auto& tri : triangles)
        {
            bool valid = true;
            for (int j = 0; j < 3; ++j)
            {
                if (tri[j] >= n_super)
                {
                    valid = false;
                    break;
                }
            }
            if (valid)
            {
                final_triangles.push_back(tri);
            }
        }
        
        // Ensure consistent orientation (counter-clockwise)
        for (auto& tri : final_triangles)
        {
            const Point& pa = nodal_data.coordinates[tri[0]];
            const Point& pb = nodal_data.coordinates[tri[1]];
            const Point& pc = nodal_data.coordinates[tri[2]];
            
            double area = (pa[0] * (pb[1] - pc[1]) + pb[0] * (pc[1] - pa[1]) + pc[0] * (pa[1] - pb[1])) / 2.0;
            if (area < 0)
            {
                std::swap(tri[1], tri[2]);
            }
        }
        
        element_data.connectivity = final_triangles;
        parameters.element_cnt = final_triangles.size();
        
        std::cout << "Delaunay triangulation complete: " << parameters.element_cnt << " elements\n";
    }

    void SPFEMSolver::apply_alpha_shape()
    {
        if (parameters.alpha_shape.empty()) return;
        
        std::vector<std::array<int, 3>> filtered_triangles;
        
        for (const auto& tri : element_data.connectivity)
        {
            bool keep = true;
            
            const Point& p0 = nodal_data.coordinates[tri[0]];
            const Point& p1 = nodal_data.coordinates[tri[1]];
            const Point& p2 = nodal_data.coordinates[tri[2]];
            
            // Compute area
            double area = std::abs((p0[0] * (p1[1] - p2[1]) + 
                                   p1[0] * (p2[1] - p0[1]) + 
                                   p2[0] * (p0[1] - p1[1])) / 2.0);
            
            if (area < 1e-12)
            {
                keep = false;
            }
            else
            {
                // Compute circumradius
                double a = std::sqrt((p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]));
                double b = std::sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]));
                double c = std::sqrt((p0[0] - p2[0]) * (p0[0] - p2[0]) + (p0[1] - p2[1]) * (p0[1] - p2[1]));
                double radius = (a * b * c) / (4 * area);
                
                // Compute centroid
                double xcen = (p0[0] + p1[0] + p2[0]) / 3.0;
                double ycen = (p0[1] + p1[1] + p2[1]) / 3.0;
                
                // Check against all alpha shape constraints
                for (const auto& alpha : parameters.alpha_shape)
                {
                    int ref_node = static_cast<int>(alpha[0]) - 1;  // Convert to 0-based
                    double ref_radius = alpha[1] * parameters.space;
                    double x_minus = alpha[2];
                    double x_add = alpha[3];
                    double y_minus = alpha[4];
                    double y_add = alpha[5];
                    
                    if (ref_node >= 0 && ref_node < parameters.node_cnt)
                    {
                        double x_ref = nodal_data.coordinates[ref_node][0];
                        double y_ref = nodal_data.coordinates[ref_node][1];
                        
                        if (xcen < x_ref + x_add && ycen < y_ref + y_add &&
                            xcen > x_ref + x_minus && ycen > y_ref + y_minus &&
                            radius > ref_radius)
                        {
                            keep = false;
                            break;
                        }
                    }
                }
            }
            
            if (keep)
            {
                filtered_triangles.push_back(tri);
            }
        }
        
        int before = element_data.connectivity.size();
        element_data.connectivity = filtered_triangles;
        parameters.element_cnt = filtered_triangles.size();
        
        std::cout << "Alpha shape: Before=" << before << ", After=" << parameters.element_cnt << "\n";
    }

    void SPFEMSolver::compute_related_elements_and_nodes()
    {
        const int n = parameters.node_cnt;
        const int ne = parameters.element_cnt;
        
        // Initialize related element data
        nodal_data.related_element_count.resize(n, 0);
        nodal_data.related_elements.resize(n);
        nodal_data.related_node_count.resize(n, 0);
        nodal_data.related_nodes.resize(n);
        
        // Count related elements per node
        for (int ie = 0; ie < ne; ++ie)
        {
            for (int j = 0; j < 3; ++j)
            {
                int node_id = element_data.connectivity[ie][j];
                nodal_data.related_element_count[node_id]++;
                nodal_data.related_elements[node_id].push_back(ie);
            }
        }
        
        // Find related nodes per node (all nodes sharing an element)
        for (int in = 0; in < n; ++in)
        {
            std::set<int> related;
            for (int ie : nodal_data.related_elements[in])
            {
                for (int j = 0; j < 3; ++j)
                {
                    related.insert(element_data.connectivity[ie][j]);
                }
            }
            nodal_data.related_nodes[in] = std::vector<int>(related.begin(), related.end());
            nodal_data.related_node_count[in] = nodal_data.related_nodes[in].size();
        }
        
        // Initialize shape function derivative storage
        nodal_data.dNdx.resize(n);
        nodal_data.dNdy.resize(n);
        for (int i = 0; i < n; ++i)
        {
            int max_related = parameters.MN * 3;
            nodal_data.dNdx[i].resize(max_related, 0.0);
            nodal_data.dNdy[i].resize(max_related, 0.0);
        }
    }

    void SPFEMSolver::run()
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        parameters.ctime = 0.0;
        parameters.istep = 0;
        
        // Prepare for leapfrog time integration
        update_half_velocity();
        
        std::cout << "\nStarting time integration...\n";
        
        while (parameters.ctime < parameters.totaltime + 1e-9)
        {
            // Output VTK at intervals
            if (parameters.istep % parameters.vtk_output_interval == 0)
            {
                output_vtk();
                std::cout << "Step=" << parameters.istep 
                          << ", Time=" << parameters.ctime << "s\n";
            }
            
            // Remeshing if enabled
            if (parameters.if_remesh == 1)
            {
                remesh_if_needed();
            }
            
            // Compute element data (areas, shape function derivatives)
            compute_element_data();
            
            // Compute nodal data (smoothed derivatives, nodal areas)
            compute_nodal_data();
            
            // Compute internal forces and update stress
            compute_internal_forces();
            
            // Rank deficiency treatment (hourglass control)
            if (parameters.rank_deficiency > 1e-6)
            {
                compute_rank_deficiency_forces();
            }
            
            // Time integration
            time_integration();
            
            // Contact with rigid walls
            if (parameters.rigid_wall_count > 0)
            {
                apply_contact_wall();
            }
            
            // Update time
            parameters.ctime += parameters.dtime;
            parameters.istep++;
            
            // Save monitor data
            save_monitor_data();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "\nSimulation finished in " << duration.count() << " seconds.\n";
    }

    void SPFEMSolver::update_half_velocity()
    {
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            for (int i = 0; i < 2; ++i)
            {
                BoundaryType type = nodal_data.boundary_type[in][i];
                double value = nodal_data.boundary_value[in][i];
                
                if (type == BoundaryType::Velocity)
                {
                    nodal_data.velocities[in][i] = value;
                }
                else
                {
                    nodal_data.velocities[in][i] += 
                        nodal_data.accelerations[in][i] * parameters.dtime * 0.5;
                }
            }
        }
        
        // Update positions
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            nodal_data.displacements[in] += nodal_data.velocities[in] * parameters.dtime;
            nodal_data.coordinates[in] += nodal_data.velocities[in] * parameters.dtime;
        }
    }

    void SPFEMSolver::compute_element_data()
    {
        const int ne = parameters.element_cnt;
        
        element_data.areas.resize(ne);
        element_data.dNdx.resize(ne);
        element_data.dNdy.resize(ne);
        
        for (int ie = 0; ie < ne; ++ie)
        {
            const auto& tri = element_data.connectivity[ie];
            
            std::array<Point, 3> x;
            for (int t = 0; t < 3; ++t)
            {
                x[t] = nodal_data.coordinates[tri[t]];
            }
            
            // Compute area
            double area = (x[0][0] * (x[1][1] - x[2][1]) +
                          x[1][0] * (x[2][1] - x[0][1]) +
                          x[2][0] * (x[0][1] - x[1][1])) * 0.5;
            
            element_data.areas[ie] = area;
            
            // Compute shape function derivatives
            for (int j = 0; j < 3; ++j)
            {
                double x1, y1, x2, y2;
                switch (j)
                {
                case 0:
                    x1 = x[1][0]; y1 = x[1][1];
                    x2 = x[2][0]; y2 = x[2][1];
                    break;
                case 1:
                    x1 = x[2][0]; y1 = x[2][1];
                    x2 = x[0][0]; y2 = x[0][1];
                    break;
                case 2:
                    x1 = x[0][0]; y1 = x[0][1];
                    x2 = x[1][0]; y2 = x[1][1];
                    break;
                default:
                    x1 = 0; y1 = 0; x2 = 0; y2 = 0;
                }
                
                double b = y1 - y2;
                double c = x2 - x1;
                
                element_data.dNdx[ie][j] = b / (2 * area);
                element_data.dNdy[ie][j] = c / (2 * area);
            }
        }
    }

    void SPFEMSolver::compute_nodal_data()
    {
        const int n = parameters.node_cnt;
        
        nodal_data.nodal_area.resize(n, 0.0);
        
        for (int in = 0; in < n; ++in)
        {
            double cur_node_area = 0.0;
            int related_element_cnt = nodal_data.related_element_count[in];
            int related_node_cnt = nodal_data.related_node_count[in];
            
            if (related_element_cnt == 0)
            {
                // Isolated node
                nodal_data.nodal_area[in] = 1e-9;
                std::fill(nodal_data.dNdx[in].begin(), nodal_data.dNdx[in].end(), 0.0);
                std::fill(nodal_data.dNdy[in].begin(), nodal_data.dNdy[in].end(), 0.0);
                continue;
            }
            
            std::vector<double> dNdx_accum(related_node_cnt, 0.0);
            std::vector<double> dNdy_accum(related_node_cnt, 0.0);
            
            const auto& related_nodes = nodal_data.related_nodes[in];
            
            for (int i = 0; i < related_element_cnt; ++i)
            {
                int cur_elem = nodal_data.related_elements[in][i];
                double cur_area = element_data.areas[cur_elem] / 3.0;
                cur_node_area += cur_area;
                
                for (int j = 0; j < 3; ++j)
                {
                    int cur_node = element_data.connectivity[cur_elem][j];
                    double cur_dNdx = element_data.dNdx[cur_elem][j];
                    double cur_dNdy = element_data.dNdy[cur_elem][j];
                    
                    // Find index in related_nodes
                    auto it = std::find(related_nodes.begin(), related_nodes.end(), cur_node);
                    if (it != related_nodes.end())
                    {
                        int idx = std::distance(related_nodes.begin(), it);
                        dNdx_accum[idx] += cur_dNdx * cur_area;
                        dNdy_accum[idx] += cur_dNdy * cur_area;
                    }
                }
            }
            
            nodal_data.nodal_area[in] = cur_node_area;
            
            // Normalize by nodal area
            std::fill(nodal_data.dNdx[in].begin(), nodal_data.dNdx[in].end(), 0.0);
            std::fill(nodal_data.dNdy[in].begin(), nodal_data.dNdy[in].end(), 0.0);
            
            for (int i = 0; i < related_node_cnt; ++i)
            {
                nodal_data.dNdx[in][i] = dNdx_accum[i] / cur_node_area;
                nodal_data.dNdy[in][i] = dNdy_accum[i] / cur_node_area;
            }
        }
    }

    void SPFEMSolver::compute_internal_forces()
    {
        const int n = parameters.node_cnt;
        
        // Reset internal forces
        std::fill(internal_forces.begin(), internal_forces.end(), Point(0, 0));
        
        for (int in = 0; in < n; ++in)
        {
            int nnode = nodal_data.related_node_count[in];
            if (nnode == 0) continue;
            
            int mat = nodal_data.material_id[in] - 1;  // Convert to 0-based
            int mat_model = parameters.mat_model[mat];
            const auto& mat_props = parameters.mat_props[mat];
            
            // Compute velocity gradient
            Tensor dudx;  // dudx, dudy, dvdx, dvdy
            for (int i = 0; i < nnode; ++i)
            {
                int cur_node = nodal_data.related_nodes[in][i];
                Point ddisp = nodal_data.velocities[cur_node] * parameters.dtime;
                
                dudx[0] += nodal_data.dNdx[in][i] * ddisp[0];
                dudx[1] += nodal_data.dNdy[in][i] * ddisp[0];
                dudx[2] += nodal_data.dNdx[in][i] * ddisp[1];
                dudx[3] += nodal_data.dNdy[in][i] * ddisp[1];
            }
            
            // Compute strain increment
            Tensor dstrain;
            dstrain[0] = dudx[0];              // eps_xx
            dstrain[1] = dudx[3];              // eps_yy
            dstrain[2] = dudx[1] + dudx[2];    // 2*eps_xy
            dstrain[3] = 0.0;                  // eps_zz (plane strain)
            
            // Update deformation gradient for rank deficiency treatment
            if (parameters.rank_deficiency > 1e-6)
            {
                nodal_data.deformation_gradient[in][0] = dudx[0] + 1.0;
                nodal_data.deformation_gradient[in][1] = dudx[1];
                nodal_data.deformation_gradient[in][2] = dudx[2];
                nodal_data.deformation_gradient[in][3] = dudx[3] + 1.0;
            }
            
            // Apply constitutive model
            Tensor stress = nodal_data.stresses[in];
            double pstrain_eq = nodal_data.plastic_strain_eq[in];
            
            apply_constitutive_model(mat_model, mat_props, stress, dstrain, pstrain_eq);
            
            nodal_data.stresses[in] = stress;
            nodal_data.plastic_strain_eq[in] = pstrain_eq;
            
            // Compute internal force contributions
            for (int i = 0; i < nnode; ++i)
            {
                // B matrix (strain-displacement)
                double B11 = nodal_data.dNdx[in][i];  // N_x
                double B22 = nodal_data.dNdy[in][i];  // N_y
                double B31 = B22;                      // N_y (for shear)
                double B32 = B11;                      // N_x (for shear)
                
                // f_int = B^T * sigma * area
                double fx = (B11 * stress[0] + B31 * stress[2]) * nodal_data.nodal_area[in];
                double fy = (B22 * stress[1] + B32 * stress[2]) * nodal_data.nodal_area[in];
                
                int cur_node = nodal_data.related_nodes[in][i];
                internal_forces[cur_node][0] += fx;
                internal_forces[cur_node][1] += fy;
            }
        }
    }

    void SPFEMSolver::compute_rank_deficiency_forces()
    {
        const int n = parameters.node_cnt;
        
        std::fill(rank_deficiency_forces.begin(), rank_deficiency_forces.end(), Point(0, 0));
        
        for (int in = 0; in < n; ++in)
        {
            int nnode = nodal_data.related_node_count[in];
            if (nnode == 0) continue;
            
            Point f_hg(0, 0);
            int mat = nodal_data.material_id[in] - 1;
            double Ei = parameters.mat_props[mat][1];  // Young's modulus
            
            // Deformation gradient of current node
            auto& Fi = nodal_data.deformation_gradient[in];
            
            for (int i = 0; i < nnode; ++i)
            {
                int cur_node = nodal_data.related_nodes[in][i];
                if (cur_node == in) continue;
                
                int mat_j = nodal_data.material_id[cur_node] - 1;
                double Ej = parameters.mat_props[mat_j][1];
                
                // Current and previous positions
                Point xi_xj = nodal_data.coordinates[cur_node] - nodal_data.coordinates[in];
                Point xi_xj_0 = xi_xj - (nodal_data.velocities[cur_node] - 
                                            nodal_data.velocities[in]) * parameters.dtime;
                
                double lpij = xi_xj.norm();
                double lpij0 = xi_xj_0.norm();
                
                if (lpij > 1e-12 && lpij0 > 1e-12)
                {
                    // Apply deformation gradient Fi
                    Point xij_i;
                    xij_i[0] = Fi[0] * xi_xj_0[0] + Fi[1] * xi_xj_0[1];
                    xij_i[1] = Fi[2] * xi_xj_0[0] + Fi[3] * xi_xj_0[1];
                    Point epsilon_ij_i = xij_i - xi_xj;
                    
                    // Apply deformation gradient Fj
                    auto& Fj = nodal_data.deformation_gradient[cur_node];
                    Point xij_j;
                    xij_j[0] = Fj[0] * xi_xj_0[0] + Fj[1] * xi_xj_0[1];
                    xij_j[1] = Fj[2] * xi_xj_0[0] + Fj[3] * xi_xj_0[1];
                    Point epsilon_ij_j = xij_j - xi_xj;
                    
                    double dot_i = epsilon_ij_i[0] * xi_xj[0] + epsilon_ij_i[1] * xi_xj[1];
                    double dot_j = epsilon_ij_j[0] * xi_xj[0] + epsilon_ij_j[1] * xi_xj[1];
                    
                    double factor = -0.5 * parameters.rank_deficiency / 
                                   (lpij * lpij * lpij0 * lpij0);
                    
                    f_hg += factor * xi_xj * 
                           (Ei * nodal_data.nodal_area[in] * dot_i +
                            Ej * nodal_data.nodal_area[cur_node] * dot_j);
                }
            }
            
            rank_deficiency_forces[in] = f_hg;
        }
    }

    void SPFEMSolver::time_integration()
    {
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            int mat = nodal_data.material_id[in] - 1;
            double rho = parameters.mat_props[mat][0];
            double mass = nodal_data.nodal_area[in] * rho;
            
            for (int i = 0; i < 2; ++i)
            {
                double gravity = parameters.mat_gravity[mat][i];
                BoundaryType type = nodal_data.boundary_type[in][i];
                double value = nodal_data.boundary_value[in][i];
                
                if (type == BoundaryType::Velocity)
                {
                    nodal_data.velocities[in][i] = value;
                }
                else
                {
                    double force = -internal_forces[in][i];
                    
                    // Gravity
                    force += gravity * mass;
                    
                    // Applied force boundary
                    if (type == BoundaryType::Force)
                    {
                        force += value;
                    }
                    
                    // Hourglass control force
                    if (parameters.rank_deficiency > 1e-6)
                    {
                        force += rank_deficiency_forces[in][i];
                    }
                    
                    // Damping
                    if (nodal_data.velocities[in][i] >= 0)
                    {
                        force -= std::abs(force) * parameters.damping;
                    }
                    else
                    {
                        force += std::abs(force) * parameters.damping;
                    }
                    
                    nodal_data.velocities[in][i] += force / mass * parameters.dtime;
                }
            }
        }
        
        // Update displacements and coordinates
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            nodal_data.displacements[in] += nodal_data.velocities[in] * parameters.dtime;
            nodal_data.coordinates[in] += nodal_data.velocities[in] * parameters.dtime;
        }
    }

    void SPFEMSolver::apply_contact_wall()
    {
        for (int iwall = 0; iwall < parameters.rigid_wall_count; ++iwall)
        {
            const auto& wall = parameters.walls[iwall];
            Point coor1(wall[0], wall[1]);
            Point coor2(wall[2], wall[3]);
            double mu = wall[4];
            
            Point vector12 = coor2 - coor1;
            double mag = vector12.norm();
            vector12 /= mag;
            
            Point vector_n(-vector12[1], vector12[0]);
            Point wall_vel(0, 0);
            
            for (int in = 0; in < parameters.node_cnt; ++in)
            {
                Point vector1s = nodal_data.coordinates[in] - coor1;
                double gap = vector1s[0] * vector_n[0] + vector1s[1] * vector_n[1];
                
                if (gap > -1e-6) continue;
                
                int mat = nodal_data.material_id[in] - 1;
                double rho = parameters.mat_props[mat][0];
                double mp = nodal_data.nodal_area[in] * rho;
                
                Point vp = nodal_data.velocities[in] - wall_vel;
                Point fc = -mp * vp / parameters.dtime;
                
                if (mu < 1.0)  // Frictional contact
                {
                    double fn_dot = fc[0] * vector_n[0] + fc[1] * vector_n[1];
                    Point fn = fn_dot * vector_n;
                    Point ft = fc - fn;
                    double ft_mag = ft.norm();
                    double fn_mag = fn.norm();
                    
                    if (ft_mag > mu * fn_mag)
                    {
                        ft *= mu * fn_mag / ft_mag;
                    }
                    fc = ft + fn;
                }
                
                Point vel_correction = fc * parameters.dtime / mp;
                nodal_data.velocities[in] += vel_correction;
                nodal_data.displacements[in] += vel_correction * parameters.dtime;
                nodal_data.coordinates[in] += vel_correction * parameters.dtime;
            }
        }
    }

    void SPFEMSolver::remesh_if_needed()
    {
        parameters.mesh_quality = compute_mesh_quality();
        
        if (parameters.mesh_quality >= parameters.remesh_threshold)
        {
            return;  // Mesh quality is acceptable
        }
        
        std::cout << "Remeshing at step " << parameters.istep 
                  << " (quality=" << parameters.mesh_quality << ")\n";
        
        // Re-triangulation
        initialize_triangulation();
        apply_alpha_shape();
        compute_related_elements_and_nodes();
        
        // Laplacian smoothing
        laplacian_smoothing();
        
        // Adjust remeshing threshold
        parameters.mesh_quality = compute_mesh_quality();
        parameters.remesh_threshold = parameters.mesh_quality - parameters.remesh_quality_space;
        
        parameters.remesh_threshold = std::min(parameters.remesh_threshold, 
                                               parameters.max_remesh_threshold);
        parameters.remesh_threshold = std::max(parameters.remesh_threshold,
                                               parameters.min_remesh_threshold);
        
        parameters.remesh_cnt++;
    }

    void SPFEMSolver::laplacian_smoothing()
    {
        double threshold = parameters.Lap_threshold * parameters.space;
        
        for (int in = 0; in < parameters.node_cnt; ++in)
        {
            int nnode = nodal_data.related_node_count[in];
            if (nnode == 0) continue;
            
            const Point& cur = nodal_data.coordinates[in];
            
            // Find minimum distance to neighbors
            double min_dis = 1e6;
            for (int t = 0; t < nnode; ++t)
            {
                int tnode = nodal_data.related_nodes[in][t];
                if (tnode == in) continue;
                
                double dist = (cur - nodal_data.coordinates[tnode]).norm();
                min_dis = std::min(min_dis, dist);
            }
            
            if (min_dis > threshold)
            {
                continue;  // No smoothing needed
            }
            
            // Compute average position of neighbors
            Point avg(0, 0);
            int cnt = 0;
            for (int t = 0; t < nnode; ++t)
            {
                int tnode = nodal_data.related_nodes[in][t];
                if (tnode == in) continue;
                avg += nodal_data.coordinates[tnode];
                cnt++;
            }
            avg /= cnt;
            
            // Move towards average, respecting boundary conditions
            Point diff = avg - nodal_data.coordinates[in];
            for (int i = 0; i < 2; ++i)
            {
                if (nodal_data.boundary_type[in][i] == BoundaryType::Velocity)
                {
                    diff[i] = 0;  // Don't move fixed boundaries
                }
            }
            
            nodal_data.coordinates[in] += diff;
        }
    }

    double SPFEMSolver::compute_mesh_quality()
    {
        double min_angle = 180.0;
        
        for (int ie = 0; ie < parameters.element_cnt; ++ie)
        {
            const auto& tri = element_data.connectivity[ie];
            std::array<Point, 3> x;
            for (int i = 0; i < 3; ++i)
            {
                x[i] = nodal_data.coordinates[tri[i]];
            }
            
            double area = (x[0][0] * (x[1][1] - x[2][1]) +
                          x[1][0] * (x[2][1] - x[0][1]) +
                          x[2][0] * (x[0][1] - x[1][1])) * 0.5;
            
            if (area <= 0)
            {
                return -1.0;  // Inverted element
            }
            
            // Compute edge lengths
            double a = (x[1] - x[2]).norm();
            double b = (x[0] - x[2]).norm();
            double c = (x[0] - x[1]).norm();
            
            // Compute angles using law of sines
            double angle0 = std::asin(2 * area / (c * b)) * 180.0 / M_PI;
            double angle1 = std::asin(2 * area / (a * c)) * 180.0 / M_PI;
            double angle2 = std::asin(2 * area / (b * a)) * 180.0 / M_PI;
            
            min_angle = std::min(min_angle, std::min(angle0, std::min(angle1, angle2)));
        }
        
        return min_angle;
    }

} // namespace ESPFEM2D
