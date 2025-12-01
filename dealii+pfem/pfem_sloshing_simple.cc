/* -----------------------------------------------------------------------------
 * PFEM2D Simple - Particle Finite Element Method for 2D Free Surface Sloshing
 * 
 * Simplified version without CGAL dependency.
 * Uses a simple alpha-shape implementation based on circumradius filtering.
 *
 * This code implements the Particle Finite Element Method (PFEM) using the 
 * deal.II library for solving free surface sloshing problems. The implementation
 * is based on the ESPFEM2D methodology.
 *
 * Key features:
 * - Lagrangian mesh formulation  
 * - Delaunay triangulation (using Bowyer-Watson algorithm)
 * - Alpha-shape boundary recognition via circumradius filtering
 * - Explicit time integration
 * - Free surface tracking
 * - VTK output for visualization
 *
 * Author: Based on ESPFEM2D by Zhang Wei, Liu Yihui, Yuan Weihai
 * Adapted for deal.II library
 * -----------------------------------------------------------------------------
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <filesystem>
#include <random>

namespace PFEM2D
{
  using namespace dealii;

  // Parameters for the sloshing problem
  struct PFEMParameters
  {
    // Physical parameters
    double density;
    double viscosity;
    double gravity_x;
    double gravity_y;
    
    // Time stepping
    double dt;
    double total_time;
    int output_interval;
    
    // Domain parameters  
    double tank_length;
    double tank_height;
    double initial_water_length;
    double initial_water_height;
    double initial_tilt_angle; // in degrees
    
    // Mesh parameters
    double mesh_size;
    double alpha_shape_radius_factor;
    double remesh_threshold;
    
    // Boundary conditions
    double wall_friction;
    double damping;
    
    PFEMParameters()
      : density(1000.0)
      , viscosity(0.001)
      , gravity_x(0.0)
      , gravity_y(-9.81)
      , dt(0.001)
      , total_time(0.1)
      , output_interval(50)
      , tank_length(1.0)
      , tank_height(0.6)
      , initial_water_length(0.4)
      , initial_water_height(0.3)
      , initial_tilt_angle(0.0)
      , mesh_size(0.04)
      , alpha_shape_radius_factor(1.2)
      , remesh_threshold(10.0)
      , wall_friction(0.0)
      , damping(0.0)
    {}
    
    void declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry("Density", "1000.0", Patterns::Double(0.0));
      prm.declare_entry("Viscosity", "0.001", Patterns::Double(0.0));
      prm.declare_entry("GravityX", "0.0", Patterns::Double());
      prm.declare_entry("GravityY", "-9.81", Patterns::Double());
      prm.declare_entry("TimeStep", "0.0005", Patterns::Double(0.0));
      prm.declare_entry("TotalTime", "2.0", Patterns::Double(0.0));
      prm.declare_entry("OutputInterval", "100", Patterns::Integer(1));
      prm.declare_entry("TankLength", "1.0", Patterns::Double(0.0));
      prm.declare_entry("TankHeight", "0.6", Patterns::Double(0.0));
      prm.declare_entry("InitialWaterLength", "0.4", Patterns::Double(0.0));
      prm.declare_entry("InitialWaterHeight", "0.3", Patterns::Double(0.0));
      prm.declare_entry("InitialTiltAngle", "0.0", Patterns::Double());
      prm.declare_entry("MeshSize", "0.02", Patterns::Double(0.0));
      prm.declare_entry("AlphaShapeRadiusFactor", "1.2", Patterns::Double(0.5));
      prm.declare_entry("RemeshThreshold", "10.0", Patterns::Double(0.0));
      prm.declare_entry("WallFriction", "0.0", Patterns::Double(0.0));
      prm.declare_entry("Damping", "0.0", Patterns::Double(0.0));
    }
    
    void parse_parameters(ParameterHandler &prm)
    {
      density = prm.get_double("Density");
      viscosity = prm.get_double("Viscosity");
      gravity_x = prm.get_double("GravityX");
      gravity_y = prm.get_double("GravityY");
      dt = prm.get_double("TimeStep");
      total_time = prm.get_double("TotalTime");
      output_interval = prm.get_integer("OutputInterval");
      tank_length = prm.get_double("TankLength");
      tank_height = prm.get_double("TankHeight");
      initial_water_length = prm.get_double("InitialWaterLength");
      initial_water_height = prm.get_double("InitialWaterHeight");
      initial_tilt_angle = prm.get_double("InitialTiltAngle");
      mesh_size = prm.get_double("MeshSize");
      alpha_shape_radius_factor = prm.get_double("AlphaShapeRadiusFactor");
      remesh_threshold = prm.get_double("RemeshThreshold");
      wall_friction = prm.get_double("WallFriction");
      damping = prm.get_double("Damping");
    }
  };

  // Simple 2D particle structure for PFEM
  struct Particle
  {
    Point<2> position;
    Point<2> velocity;
    Point<2> acceleration;
    double pressure;
    int boundary_type; // 0: interior, 1: wall, 2: free surface
    
    Particle()
      : position()
      , velocity()
      , acceleration()
      , pressure(0.0)
      , boundary_type(0)
    {}
    
    Particle(const Point<2> &pos)
      : position(pos)
      , velocity()
      , acceleration()
      , pressure(0.0)
      , boundary_type(0)
    {}
  };

  // Triangle element structure
  struct Triangle
  {
    std::array<unsigned int, 3> vertices;
    double area;
    std::array<double, 3> dN_dx;
    std::array<double, 3> dN_dy;
    bool is_valid;
    
    Triangle() : area(0.0), is_valid(true) 
    {
      vertices.fill(0);
      dN_dx.fill(0.0);
      dN_dy.fill(0.0);
    }
  };

  // Simple Delaunay triangulation using incremental insertion
  class DelaunayTriangulation2D
  {
  public:
    DelaunayTriangulation2D() {}
    
    void triangulate(const std::vector<Point<2>> &points, 
                     std::vector<std::array<unsigned int, 3>> &triangles)
    {
      triangles.clear();
      
      if (points.size() < 3)
        return;
      
      // Find bounding box
      double min_x = std::numeric_limits<double>::max();
      double max_x = std::numeric_limits<double>::lowest();
      double min_y = std::numeric_limits<double>::max();
      double max_y = std::numeric_limits<double>::lowest();
      
      for (const auto &p : points)
      {
        min_x = std::min(min_x, p[0]);
        max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]);
        max_y = std::max(max_y, p[1]);
      }
      
      double dx = max_x - min_x;
      double dy = max_y - min_y;
      double d_max = std::max(dx, dy);
      double mid_x = (min_x + max_x) / 2.0;
      double mid_y = (min_y + max_y) / 2.0;
      
      // Create super triangle
      std::vector<Point<2>> all_points = points;
      all_points.emplace_back(mid_x - 2.0 * d_max, mid_y - d_max);
      all_points.emplace_back(mid_x + 2.0 * d_max, mid_y - d_max);
      all_points.emplace_back(mid_x, mid_y + 2.0 * d_max);
      
      unsigned int st0 = points.size();
      unsigned int st1 = points.size() + 1;
      unsigned int st2 = points.size() + 2;
      
      // Initial triangle (super triangle)
      std::vector<std::array<unsigned int, 3>> tri_list;
      tri_list.push_back({st0, st1, st2});
      
      // Insert points one by one (Bowyer-Watson algorithm)
      for (unsigned int i = 0; i < points.size(); ++i)
      {
        const Point<2> &p = all_points[i];
        
        // Find triangles whose circumcircle contains p
        std::vector<std::array<unsigned int, 3>> bad_triangles;
        std::vector<std::array<unsigned int, 3>> good_triangles;
        
        for (const auto &tri : tri_list)
        {
          if (in_circumcircle(all_points[tri[0]], all_points[tri[1]], 
                             all_points[tri[2]], p))
          {
            bad_triangles.push_back(tri);
          }
          else
          {
            good_triangles.push_back(tri);
          }
        }
        
        // Find boundary edges of the polygon hole
        std::vector<std::array<unsigned int, 2>> polygon;
        for (const auto &tri : bad_triangles)
        {
          for (int j = 0; j < 3; ++j)
          {
            unsigned int e0 = tri[j];
            unsigned int e1 = tri[(j + 1) % 3];
            
            // Check if this edge is shared with another bad triangle
            bool shared = false;
            for (const auto &other : bad_triangles)
            {
              if (&tri == &other)
                continue;
              
              for (int k = 0; k < 3; ++k)
              {
                unsigned int oe0 = other[k];
                unsigned int oe1 = other[(k + 1) % 3];
                if ((e0 == oe0 && e1 == oe1) || (e0 == oe1 && e1 == oe0))
                {
                  shared = true;
                  break;
                }
              }
              if (shared)
                break;
            }
            
            if (!shared)
            {
              polygon.push_back({e0, e1});
            }
          }
        }
        
        // Create new triangles
        tri_list = good_triangles;
        for (const auto &edge : polygon)
        {
          tri_list.push_back({edge[0], edge[1], i});
        }
      }
      
      // Remove triangles that contain super triangle vertices
      for (const auto &tri : tri_list)
      {
        if (tri[0] < points.size() && tri[1] < points.size() && tri[2] < points.size())
        {
          triangles.push_back(tri);
        }
      }
    }
    
  private:
    bool in_circumcircle(const Point<2> &a, const Point<2> &b, 
                        const Point<2> &c, const Point<2> &p) const
    {
      double ax = a[0] - p[0];
      double ay = a[1] - p[1];
      double bx = b[0] - p[0];
      double by = b[1] - p[1];
      double cx = c[0] - p[0];
      double cy = c[1] - p[1];
      
      double det = (ax * ax + ay * ay) * (bx * cy - cx * by) -
                   (bx * bx + by * by) * (ax * cy - cx * ay) +
                   (cx * cx + cy * cy) * (ax * by - bx * ay);
      
      // Check orientation
      double orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
      
      if (orient > 0)
        return det > 0;
      else
        return det < 0;
    }
  };

  // Main PFEM Sloshing class
  class PFEMSloshing
  {
  public:
    PFEMSloshing(const PFEMParameters &params);
    void run();
    
  private:
    void initialize_particles();
    void generate_mesh();
    void apply_alpha_shape();
    void compute_element_data();
    void compute_nodal_data();
    void compute_forces();
    void time_integration();
    void apply_boundary_conditions();
    void update_boundary_types();
    double compute_mesh_quality();
    bool check_remesh_needed();
    void output_results(unsigned int step);
    void output_vtk(const std::string &filename);
    
    // Helper functions
    double compute_triangle_area(unsigned int i0, unsigned int i1, unsigned int i2) const;
    double compute_circumradius(unsigned int i0, unsigned int i1, unsigned int i2) const;
    void compute_shape_derivatives(unsigned int elem_idx);
    
    PFEMParameters parameters;
    
    // Particle data
    std::vector<Particle> particles;
    unsigned int n_particles;
    
    // Mesh data
    std::vector<Triangle> elements;
    std::vector<std::vector<unsigned int>> node_to_elements;
    std::vector<std::vector<unsigned int>> node_to_nodes;
    
    // Nodal quantities (SPFEM approach)
    std::vector<double> nodal_area;
    std::vector<std::array<double, 2>> nodal_force_int;
    std::vector<std::array<double, 2>> nodal_force_ext;
    std::vector<std::vector<double>> nodal_dNdx;
    std::vector<std::vector<double>> nodal_dNdy;
    
    // Simulation state
    double current_time;
    unsigned int time_step;
    double remesh_threshold;
    unsigned int remesh_count;
    
    // Triangulation
    DelaunayTriangulation2D triangulator;
    
    // Output
    std::string output_directory;
  };

  // Constructor
  PFEMSloshing::PFEMSloshing(const PFEMParameters &params)
    : parameters(params)
    , n_particles(0)
    , current_time(0.0)
    , time_step(0)
    , remesh_threshold(params.remesh_threshold)
    , remesh_count(0)
    , output_directory("output")
  {
    // Create output directory using C++17 filesystem
    std::filesystem::create_directories(output_directory);
  }

  // Initialize particles based on initial water configuration
  void PFEMSloshing::initialize_particles()
  {
    particles.clear();
    
    const double h = parameters.mesh_size;
    const double water_L = parameters.initial_water_length;
    const double water_H = parameters.initial_water_height;
    const double tilt = parameters.initial_tilt_angle * M_PI / 180.0;
    
    // Generate particles in a grid pattern
    int nx = static_cast<int>(std::ceil(water_L / h));
    int ny = static_cast<int>(std::ceil(water_H / h));
    
    // Add small perturbation to avoid degenerate configurations
    // Using C++11 random number facilities
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    
    for (int j = 0; j <= ny; ++j)
    {
      for (int i = 0; i <= nx; ++i)
      {
        double x = i * h;
        double y = j * h;
        
        // Small random perturbation (0.1% of mesh size) using C++11 random
        double px = dist(rng) * h * 0.001;
        double py = dist(rng) * h * 0.001;
        x += px;
        y += py;
        
        // Apply initial tilt if any
        if (std::abs(tilt) > 1e-10)
        {
          double x_new = x * std::cos(tilt) - y * std::sin(tilt);
          double y_new = x * std::sin(tilt) + y * std::cos(tilt);
          x = x_new;
          y = y_new;
        }
        
        // Offset from walls
        x += h * 0.5;
        y += h * 0.5;
        
        Particle p(Point<2>(x, y));
        particles.push_back(p);
      }
    }
    
    n_particles = particles.size();
    
    // Initialize data structures
    nodal_area.resize(n_particles, 0.0);
    nodal_force_int.resize(n_particles, {0.0, 0.0});
    nodal_force_ext.resize(n_particles, {0.0, 0.0});
    node_to_elements.resize(n_particles);
    node_to_nodes.resize(n_particles);
    nodal_dNdx.resize(n_particles);
    nodal_dNdy.resize(n_particles);
    
    std::cout << "Initialized " << n_particles << " particles" << std::endl;
  }

  // Generate mesh using Delaunay triangulation
  void PFEMSloshing::generate_mesh()
  {
    elements.clear();
    
    // Clear connectivity
    for (auto &conn : node_to_elements)
      conn.clear();
    for (auto &conn : node_to_nodes)
      conn.clear();
    
    // Get point positions
    std::vector<Point<2>> points;
    points.reserve(n_particles);
    for (const auto &p : particles)
    {
      points.push_back(p.position);
    }
    
    // Triangulate
    std::vector<std::array<unsigned int, 3>> tri_indices;
    triangulator.triangulate(points, tri_indices);
    
    // Convert to Triangle structures
    elements.reserve(tri_indices.size());
    for (const auto &tri : tri_indices)
    {
      Triangle t;
      t.vertices = tri;
      t.area = compute_triangle_area(tri[0], tri[1], tri[2]);
      t.is_valid = (t.area > 1e-14);
      
      if (t.is_valid)
      {
        elements.push_back(t);
      }
    }
    
    std::cout << "Generated " << elements.size() << " elements" << std::endl;
  }

  // Apply alpha shape to remove external elements
  void PFEMSloshing::apply_alpha_shape()
  {
    const double alpha_radius = parameters.alpha_shape_radius_factor * parameters.mesh_size;
    
    std::vector<Triangle> filtered_elements;
    filtered_elements.reserve(elements.size());
    
    for (auto &elem : elements)
    {
      if (!elem.is_valid)
        continue;
        
      double r = compute_circumradius(elem.vertices[0], elem.vertices[1], elem.vertices[2]);
      
      // Keep element if circumradius is less than alpha radius
      if (r <= alpha_radius)
      {
        filtered_elements.push_back(elem);
      }
    }
    
    elements = std::move(filtered_elements);
    
    // Rebuild node-element connectivity
    for (auto &conn : node_to_elements)
      conn.clear();
    
    for (unsigned int e = 0; e < elements.size(); ++e)
    {
      for (int i = 0; i < 3; ++i)
      {
        node_to_elements[elements[e].vertices[i]].push_back(e);
      }
    }
    
    // Build node-node connectivity
    for (auto &conn : node_to_nodes)
      conn.clear();
    
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      std::set<unsigned int> neighbors;
      for (unsigned int e : node_to_elements[n])
      {
        for (int i = 0; i < 3; ++i)
        {
          if (elements[e].vertices[i] != n)
          {
            neighbors.insert(elements[e].vertices[i]);
          }
        }
      }
      node_to_nodes[n].assign(neighbors.begin(), neighbors.end());
    }
    
    // Update boundary types
    update_boundary_types();
    
    std::cout << "After alpha shape: " << elements.size() << " elements" << std::endl;
  }

  // Update boundary type for each particle
  void PFEMSloshing::update_boundary_types()
  {
    const double epsilon = parameters.mesh_size * 0.5;
    
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      const Point<2> &pos = particles[n].position;
      
      // Check wall boundaries
      bool on_bottom = (pos[1] < epsilon);
      bool on_left = (pos[0] < epsilon);
      bool on_right = (pos[0] > parameters.tank_length - epsilon);
      
      if (on_bottom || on_left || on_right)
      {
        particles[n].boundary_type = 1; // Wall
      }
      else
      {
        // Check if on free surface (has fewer connected elements than interior nodes)
        size_t n_connected = node_to_elements[n].size();
        if (n_connected < 4) // Simplified criterion for free surface
        {
          particles[n].boundary_type = 2; // Free surface
        }
        else
        {
          particles[n].boundary_type = 0; // Interior
        }
      }
    }
  }

  // Compute element shape function derivatives
  void PFEMSloshing::compute_element_data()
  {
    for (unsigned int e = 0; e < elements.size(); ++e)
    {
      compute_shape_derivatives(e);
    }
  }

  // Compute shape function derivatives for a single element
  void PFEMSloshing::compute_shape_derivatives(unsigned int elem_idx)
  {
    Triangle &elem = elements[elem_idx];
    
    const Point<2> &p0 = particles[elem.vertices[0]].position;
    const Point<2> &p1 = particles[elem.vertices[1]].position;
    const Point<2> &p2 = particles[elem.vertices[2]].position;
    
    double x0 = p0[0], y0 = p0[1];
    double x1 = p1[0], y1 = p1[1];
    double x2 = p2[0], y2 = p2[1];
    
    // Area using cross product (signed)
    double area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    elem.area = std::abs(area);
    
    if (elem.area < 1e-14)
    {
      elem.is_valid = false;
      return;
    }
    
    double inv_2A = 1.0 / (2.0 * area);
    
    // Shape function derivatives for linear triangle
    elem.dN_dx[0] = (y1 - y2) * inv_2A;
    elem.dN_dy[0] = (x2 - x1) * inv_2A;
    
    elem.dN_dx[1] = (y2 - y0) * inv_2A;
    elem.dN_dy[1] = (x0 - x2) * inv_2A;
    
    elem.dN_dx[2] = (y0 - y1) * inv_2A;
    elem.dN_dy[2] = (x1 - x0) * inv_2A;
  }

  // Compute nodal areas and averaged shape function derivatives (SPFEM approach)
  void PFEMSloshing::compute_nodal_data()
  {
    // Reset nodal data
    std::fill(nodal_area.begin(), nodal_area.end(), 0.0);
    
    for (auto &dNdx : nodal_dNdx)
      dNdx.clear();
    for (auto &dNdy : nodal_dNdy)
      dNdy.clear();
    
    // Compute nodal areas and averaged derivatives
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      const auto &connected_elements = node_to_elements[n];
      const auto &connected_nodes = node_to_nodes[n];
      
      if (connected_elements.empty())
      {
        nodal_area[n] = 1e-12; // Small value for isolated nodes
        continue;
      }
      
      double area_sum = 0.0;
      std::map<unsigned int, double> dNdx_sum, dNdy_sum;
      
      // Initialize sums for all connected nodes
      for (unsigned int neighbor : connected_nodes)
      {
        dNdx_sum[neighbor] = 0.0;
        dNdy_sum[neighbor] = 0.0;
      }
      dNdx_sum[n] = 0.0;
      dNdy_sum[n] = 0.0;
      
      // Accumulate contributions from connected elements
      for (unsigned int e : connected_elements)
      {
        const Triangle &elem = elements[e];
        if (!elem.is_valid)
          continue;
          
        double elem_area = elem.area / 3.0; // Contribution to node
        area_sum += elem_area;
        
        // Add shape derivative contributions
        for (int i = 0; i < 3; ++i)
        {
          unsigned int node_i = elem.vertices[i];
          dNdx_sum[node_i] += elem.dN_dx[i] * elem_area;
          dNdy_sum[node_i] += elem.dN_dy[i] * elem_area;
        }
      }
      
      nodal_area[n] = std::max(area_sum, 1e-12);
      
      // Normalize and store derivatives
      nodal_dNdx[n].resize(connected_nodes.size() + 1);
      nodal_dNdy[n].resize(connected_nodes.size() + 1);
      
      // Self derivative first
      nodal_dNdx[n][0] = dNdx_sum[n] / nodal_area[n];
      nodal_dNdy[n][0] = dNdy_sum[n] / nodal_area[n];
      
      // Neighbor derivatives
      for (size_t i = 0; i < connected_nodes.size(); ++i)
      {
        unsigned int neighbor = connected_nodes[i];
        nodal_dNdx[n][i + 1] = dNdx_sum[neighbor] / nodal_area[n];
        nodal_dNdy[n][i + 1] = dNdy_sum[neighbor] / nodal_area[n];
      }
    }
  }

  // Compute internal and external forces
  void PFEMSloshing::compute_forces()
  {
    // Reset forces
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      nodal_force_int[n] = {0.0, 0.0};
      nodal_force_ext[n] = {0.0, 0.0};
    }
    
    const double mu = parameters.viscosity;
    const double rho = parameters.density;
    
    // Compute forces using SPFEM (smoothed particle finite element) approach
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      const auto &connected_nodes = node_to_nodes[n];
      
      if (connected_nodes.empty() || nodal_dNdx[n].empty())
        continue;
      
      // Compute velocity gradients at node n
      double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;
      
      // Self contribution
      dudx += nodal_dNdx[n][0] * particles[n].velocity[0];
      dudy += nodal_dNdy[n][0] * particles[n].velocity[0];
      dvdx += nodal_dNdx[n][0] * particles[n].velocity[1];
      dvdy += nodal_dNdy[n][0] * particles[n].velocity[1];
      
      // Neighbor contributions
      for (size_t i = 0; i < connected_nodes.size(); ++i)
      {
        unsigned int neighbor = connected_nodes[i];
        dudx += nodal_dNdx[n][i + 1] * particles[neighbor].velocity[0];
        dudy += nodal_dNdy[n][i + 1] * particles[neighbor].velocity[0];
        dvdx += nodal_dNdx[n][i + 1] * particles[neighbor].velocity[1];
        dvdy += nodal_dNdy[n][i + 1] * particles[neighbor].velocity[1];
      }
      
      // Viscous stress tensor (Newtonian fluid)
      double sigma_xx = 2.0 * mu * dudx;
      double sigma_yy = 2.0 * mu * dvdy;
      double sigma_xy = mu * (dudy + dvdx);
      
      // Internal force (divergence of stress, distributed to connected nodes)
      double fx_int = 0.0, fy_int = 0.0;
      
      fx_int = (nodal_dNdx[n][0] * sigma_xx + nodal_dNdy[n][0] * sigma_xy) * nodal_area[n];
      fy_int = (nodal_dNdx[n][0] * sigma_xy + nodal_dNdy[n][0] * sigma_yy) * nodal_area[n];
      
      nodal_force_int[n][0] = fx_int;
      nodal_force_int[n][1] = fy_int;
      
      // External forces (gravity)
      double mass = rho * nodal_area[n];
      nodal_force_ext[n][0] = mass * parameters.gravity_x;
      nodal_force_ext[n][1] = mass * parameters.gravity_y;
    }
  }

  // Explicit time integration (leapfrog/Verlet scheme)
  void PFEMSloshing::time_integration()
  {
    const double dt = parameters.dt;
    const double rho = parameters.density;
    const double damping = parameters.damping;
    
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      double mass = rho * nodal_area[n];
      if (mass < 1e-14)
        continue;
      
      // Total force
      double fx = nodal_force_ext[n][0] - nodal_force_int[n][0];
      double fy = nodal_force_ext[n][1] - nodal_force_int[n][1];
      
      // Apply damping
      if (damping > 0)
      {
        if (particles[n].velocity[0] >= 0)
          fx -= std::abs(fx) * damping;
        else
          fx += std::abs(fx) * damping;
          
        if (particles[n].velocity[1] >= 0)
          fy -= std::abs(fy) * damping;
        else
          fy += std::abs(fy) * damping;
      }
      
      // Update acceleration
      particles[n].acceleration = Point<2>(fx / mass, fy / mass);
      
      // Update velocity (half-step forward)
      particles[n].velocity[0] += particles[n].acceleration[0] * dt;
      particles[n].velocity[1] += particles[n].acceleration[1] * dt;
      
      // Update position
      particles[n].position[0] += particles[n].velocity[0] * dt;
      particles[n].position[1] += particles[n].velocity[1] * dt;
    }
  }

  // Apply boundary conditions (wall contacts)
  void PFEMSloshing::apply_boundary_conditions()
  {
    const double tank_L = parameters.tank_length;
    const double epsilon = parameters.mesh_size * 0.1;
    const double friction = parameters.wall_friction;
    
    for (auto &p : particles)
    {
      // Bottom wall (y = 0)
      if (p.position[1] < epsilon)
      {
        p.position[1] = epsilon;
        if (p.velocity[1] < 0.0)
        {
          p.velocity[1] = 0.0;
          p.velocity[0] *= (1.0 - friction);
        }
      }
      
      // Left wall (x = 0)
      if (p.position[0] < epsilon)
      {
        p.position[0] = epsilon;
        if (p.velocity[0] < 0.0)
        {
          p.velocity[0] = 0.0;
          p.velocity[1] *= (1.0 - friction);
        }
      }
      
      // Right wall (x = tank_L)
      if (p.position[0] > tank_L - epsilon)
      {
        p.position[0] = tank_L - epsilon;
        if (p.velocity[0] > 0.0)
        {
          p.velocity[0] = 0.0;
          p.velocity[1] *= (1.0 - friction);
        }
      }
    }
  }

  // Compute mesh quality (minimum angle)
  double PFEMSloshing::compute_mesh_quality()
  {
    double min_angle = 180.0;
    
    for (const auto &elem : elements)
    {
      if (!elem.is_valid)
        continue;
      
      const Point<2> &p0 = particles[elem.vertices[0]].position;
      const Point<2> &p1 = particles[elem.vertices[1]].position;
      const Point<2> &p2 = particles[elem.vertices[2]].position;
      
      // Compute edge lengths
      double a = p0.distance(p1);
      double b = p1.distance(p2);
      double c = p2.distance(p0);
      
      if (a < 1e-12 || b < 1e-12 || c < 1e-12)
        continue;
      
      // Compute angles using law of cosines
      auto compute_angle = [](double opp, double adj1, double adj2) {
        double cos_val = (adj1 * adj1 + adj2 * adj2 - opp * opp) / (2.0 * adj1 * adj2);
        cos_val = std::max(-1.0, std::min(1.0, cos_val));
        return std::acos(cos_val) * 180.0 / M_PI;
      };
      
      double angle0 = compute_angle(b, c, a);
      double angle1 = compute_angle(c, a, b);
      double angle2 = compute_angle(a, b, c);
      
      min_angle = std::min(min_angle, std::min(angle0, std::min(angle1, angle2)));
    }
    
    return min_angle;
  }

  // Check if remeshing is needed
  bool PFEMSloshing::check_remesh_needed()
  {
    double quality = compute_mesh_quality();
    return quality < remesh_threshold;
  }

  // Compute triangle area
  double PFEMSloshing::compute_triangle_area(unsigned int i0, unsigned int i1, unsigned int i2) const
  {
    const Point<2> &p0 = particles[i0].position;
    const Point<2> &p1 = particles[i1].position;
    const Point<2> &p2 = particles[i2].position;
    
    return 0.5 * std::abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - 
                          (p2[0] - p0[0]) * (p1[1] - p0[1]));
  }

  // Compute circumradius of triangle
  double PFEMSloshing::compute_circumradius(unsigned int i0, unsigned int i1, unsigned int i2) const
  {
    const Point<2> &p0 = particles[i0].position;
    const Point<2> &p1 = particles[i1].position;
    const Point<2> &p2 = particles[i2].position;
    
    double a = p0.distance(p1);
    double b = p1.distance(p2);
    double c = p2.distance(p0);
    
    double area = compute_triangle_area(i0, i1, i2);
    
    if (area < 1e-14)
      return 1e10;
    
    return (a * b * c) / (4.0 * area);
  }

  // Output results to VTK file
  void PFEMSloshing::output_vtk(const std::string &filename)
  {
    std::ofstream out(filename);
    
    out << "# vtk DataFile Version 3.0\n";
    out << "PFEM 2D Sloshing Simulation\n";
    out << "ASCII\n\n";
    
    out << "DATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << n_particles << " float\n";
    
    for (const auto &p : particles)
    {
      out << std::fixed << std::setprecision(8);
      out << p.position[0] << " " << p.position[1] << " 0.0\n";
    }
    
    // Count valid elements
    size_t valid_count = 0;
    for (const auto &elem : elements)
    {
      if (elem.is_valid)
        valid_count++;
    }
    
    out << "\nCELLS " << valid_count << " " << valid_count * 4 << "\n";
    for (const auto &elem : elements)
    {
      if (elem.is_valid)
      {
        out << "3 " << elem.vertices[0] << " " << elem.vertices[1] << " " << elem.vertices[2] << "\n";
      }
    }
    
    out << "\nCELL_TYPES " << valid_count << "\n";
    for (const auto &elem : elements)
    {
      if (elem.is_valid)
      {
        out << "5\n"; // VTK_TRIANGLE
      }
    }
    
    out << "\nPOINT_DATA " << n_particles << "\n";
    
    out << "VECTORS velocity float\n";
    for (const auto &p : particles)
    {
      out << std::scientific << std::setprecision(6);
      out << p.velocity[0] << " " << p.velocity[1] << " 0.0\n";
    }
    
    out << "\nSCALARS pressure float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto &p : particles)
    {
      out << p.pressure << "\n";
    }
    
    out << "\nSCALARS boundary_type int 1\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto &p : particles)
    {
      out << p.boundary_type << "\n";
    }
    
    out << "\nSCALARS nodal_area float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      out << nodal_area[n] << "\n";
    }
    
    out << "\nSCALARS velocity_magnitude float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto &p : particles)
    {
      double vmag = std::sqrt(p.velocity[0] * p.velocity[0] + p.velocity[1] * p.velocity[1]);
      out << vmag << "\n";
    }
    
    out.close();
  }

  // Output results
  void PFEMSloshing::output_results(unsigned int step)
  {
    std::ostringstream filename;
    filename << output_directory << "/sloshing_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    output_vtk(filename.str());
    std::cout << "  Output: " << filename.str() << std::endl;
  }

  // Main simulation loop
  void PFEMSloshing::run()
  {
    std::cout << "\n=== PFEM 2D Free Surface Sloshing Simulation ===" << std::endl;
    std::cout << "Based on ESPFEM2D methodology" << std::endl;
    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Tank: " << parameters.tank_length << " x " << parameters.tank_height << " m" << std::endl;
    std::cout << "  Initial water: " << parameters.initial_water_length << " x " 
              << parameters.initial_water_height << " m" << std::endl;
    std::cout << "  Mesh size: " << parameters.mesh_size << " m" << std::endl;
    std::cout << "  Time step: " << parameters.dt << " s" << std::endl;
    std::cout << "  Total time: " << parameters.total_time << " s" << std::endl;
    std::cout << "  Density: " << parameters.density << " kg/m³" << std::endl;
    std::cout << "  Viscosity: " << parameters.viscosity << " Pa·s" << std::endl;
    std::cout << "  Gravity: (" << parameters.gravity_x << ", " << parameters.gravity_y << ") m/s²" << std::endl;
    std::cout << std::endl;
    
    // Initialize
    Timer timer;
    timer.start();
    
    std::cout << "Initializing..." << std::endl;
    initialize_particles();
    
    std::cout << "Generating initial mesh..." << std::endl;
    generate_mesh();
    apply_alpha_shape();
    compute_element_data();
    compute_nodal_data();
    
    // Output initial state
    output_results(0);
    
    std::cout << "\nStarting time integration..." << std::endl;
    
    // Time stepping loop
    while (current_time < parameters.total_time)
    {
      // Check for remeshing
      if (time_step > 0 && time_step % 100 == 0)
      {
        if (check_remesh_needed())
        {
          std::cout << "\n  Remeshing at step " << time_step << ", time = " << current_time << " s" << std::endl;
          generate_mesh();
          apply_alpha_shape();
          remesh_count++;
          std::cout << "  After remesh: " << elements.size() << " elements" << std::endl;
        }
      }
      
      // Compute forces
      compute_forces();
      
      // Time integration
      time_integration();
      
      // Apply boundary conditions
      apply_boundary_conditions();
      
      // Update mesh data
      compute_element_data();
      compute_nodal_data();
      
      // Update time
      current_time += parameters.dt;
      time_step++;
      
      // Output
      if (time_step % parameters.output_interval == 0)
      {
        std::cout << "\nStep " << time_step << ", Time = " << std::fixed << std::setprecision(4) 
                  << current_time << " s, Elements = " << elements.size();
        
        // Compute and print some statistics
        double max_vel = 0.0;
        double max_y = 0.0;
        for (const auto &p : particles)
        {
          double v = std::sqrt(p.velocity[0] * p.velocity[0] + p.velocity[1] * p.velocity[1]);
          max_vel = std::max(max_vel, v);
          max_y = std::max(max_y, p.position[1]);
        }
        std::cout << ", Max velocity = " << std::scientific << std::setprecision(2) << max_vel 
                  << " m/s, Max height = " << std::fixed << std::setprecision(4) << max_y << " m" << std::endl;
        
        output_results(time_step);
      }
    }
    
    timer.stop();
    
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "Total steps: " << time_step << std::endl;
    std::cout << "Remesh count: " << remesh_count << std::endl;
    std::cout << "Wall clock time: " << std::fixed << std::setprecision(2) << timer.wall_time() << " seconds" << std::endl;
    std::cout << "Output files written to: " << output_directory << "/" << std::endl;
  }

} // namespace PFEM2D


// Main function
int main(int argc, char *argv[])
{
  try
  {
    using namespace PFEM2D;
    
    PFEMParameters params;
    
    // Check for parameter file
    if (argc > 1)
    {
      std::string param_file = argv[1];
      dealii::ParameterHandler prm;
      params.declare_parameters(prm);
      prm.parse_input(param_file);
      params.parse_parameters(prm);
      std::cout << "Loaded parameters from " << param_file << std::endl;
    }
    else
    {
      std::cout << "Using default parameters (dam break / sloshing test)" << std::endl;
    }
    
    PFEMSloshing problem(params);
    problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  
  return 0;
}
