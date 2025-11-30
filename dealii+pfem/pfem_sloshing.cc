/* -----------------------------------------------------------------------------
 * PFEM2D - Particle Finite Element Method for 2D Free Surface Sloshing
 * 
 * This code implements the Particle Finite Element Method (PFEM) using the 
 * deal.II library for solving free surface sloshing problems. The implementation
 * is based on the ESPFEM2D methodology but adapted for incompressible Navier-Stokes
 * equations with free surface tracking.
 *
 * Key features:
 * - Lagrangian mesh formulation
 * - Delaunay triangulation with alpha-shape boundary recognition
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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>

namespace PFEM2D
{
  using namespace dealii;

  // CGAL typedefs for alpha shape computation
  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef K::FT FT;
  typedef K::Point_2 CGAL_Point;
  typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
  typedef CGAL::Alpha_shape_face_base_2<K> Fb;
  typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
  typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
  typedef CGAL::Alpha_shape_2<Delaunay> Alpha_shape;

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
    
    PFEMParameters()
      : density(1000.0)
      , viscosity(0.001)
      , gravity_x(0.0)
      , gravity_y(-9.81)
      , dt(0.001)
      , total_time(2.0)
      , output_interval(50)
      , tank_length(1.0)
      , tank_height(0.5)
      , initial_water_length(0.4)
      , initial_water_height(0.3)
      , initial_tilt_angle(0.0)
      , mesh_size(0.02)
      , alpha_shape_radius_factor(1.5)
      , remesh_threshold(15.0)
      , wall_friction(0.0)
    {}
    
    void declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry("Density", "1000.0", Patterns::Double(0.0));
      prm.declare_entry("Viscosity", "0.001", Patterns::Double(0.0));
      prm.declare_entry("GravityX", "0.0", Patterns::Double());
      prm.declare_entry("GravityY", "-9.81", Patterns::Double());
      prm.declare_entry("TimeStep", "0.001", Patterns::Double(0.0));
      prm.declare_entry("TotalTime", "2.0", Patterns::Double(0.0));
      prm.declare_entry("OutputInterval", "50", Patterns::Integer(1));
      prm.declare_entry("TankLength", "1.0", Patterns::Double(0.0));
      prm.declare_entry("TankHeight", "0.5", Patterns::Double(0.0));
      prm.declare_entry("InitialWaterLength", "0.4", Patterns::Double(0.0));
      prm.declare_entry("InitialWaterHeight", "0.3", Patterns::Double(0.0));
      prm.declare_entry("InitialTiltAngle", "0.0", Patterns::Double());
      prm.declare_entry("MeshSize", "0.02", Patterns::Double(0.0));
      prm.declare_entry("AlphaShapeRadiusFactor", "1.5", Patterns::Double(1.0));
      prm.declare_entry("RemeshThreshold", "15.0", Patterns::Double(0.0));
      prm.declare_entry("WallFriction", "0.0", Patterns::Double(0.0));
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
    }
  };

  // Simple 2D particle structure for PFEM
  struct Particle
  {
    Point<2> position;
    Point<2> velocity;
    Point<2> acceleration;
    double pressure;
    int boundary_type; // 0: free, 1: wall, 2: free surface
    
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
    
    Triangle() : area(0.0), is_valid(true) {}
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
    double compute_mesh_quality();
    bool check_remesh_needed();
    void output_results(const unsigned int step);
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
    
    // Nodal quantities
    std::vector<double> nodal_area;
    std::vector<std::array<double, 2>> nodal_force_int;
    std::vector<std::array<double, 2>> nodal_force_ext;
    
    // Simulation state
    double current_time;
    unsigned int time_step;
    double remesh_threshold;
    unsigned int remesh_count;
    
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
    // Create output directory
    std::string cmd = "mkdir -p " + output_directory;
    int result = system(cmd.c_str());
    (void)result; // Suppress unused result warning
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
    int nx = static_cast<int>(water_L / h) + 1;
    int ny = static_cast<int>(water_H / h) + 1;
    
    for (int j = 0; j <= ny; ++j)
    {
      for (int i = 0; i <= nx; ++i)
      {
        double x = i * h;
        double y = j * h;
        
        // Apply initial tilt if any
        if (std::abs(tilt) > 1e-10)
        {
          double x_new = x * std::cos(tilt) - y * std::sin(tilt);
          double y_new = x * std::sin(tilt) + y * std::cos(tilt);
          x = x_new;
          y = y_new;
        }
        
        // Add some offset from walls
        x += h * 0.5;
        y += h * 0.5;
        
        Particle p(Point<2>(x, y));
        
        // Set boundary type
        if (j == 0)
          p.boundary_type = 1; // Bottom wall contact
        else if (std::abs(x - h * 0.5) < 1e-10)
          p.boundary_type = 1; // Left wall contact
        else if (j == ny)
          p.boundary_type = 2; // Free surface
        else if (i == nx)
          p.boundary_type = 2; // Free surface (right side of water)
        
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
    
    std::cout << "Initialized " << n_particles << " particles" << std::endl;
  }

  // Generate mesh using Delaunay triangulation
  void PFEMSloshing::generate_mesh()
  {
    elements.clear();
    
    // Clear node-element connectivity
    for (auto &conn : node_to_elements)
      conn.clear();
    for (auto &conn : node_to_nodes)
      conn.clear();
    
    // Create CGAL points
    std::vector<CGAL_Point> cgal_points;
    cgal_points.reserve(n_particles);
    
    for (const auto &p : particles)
    {
      cgal_points.push_back(CGAL_Point(p.position[0], p.position[1]));
    }
    
    // Create Delaunay triangulation
    Delaunay dt(cgal_points.begin(), cgal_points.end());
    
    // Create a map from CGAL points to particle indices
    std::map<CGAL_Point, unsigned int> point_to_index;
    for (unsigned int i = 0; i < n_particles; ++i)
    {
      point_to_index[cgal_points[i]] = i;
    }
    
    // Extract triangles
    for (auto face = dt.finite_faces_begin(); face != dt.finite_faces_end(); ++face)
    {
      Triangle tri;
      
      for (int i = 0; i < 3; ++i)
      {
        CGAL_Point pt = face->vertex(i)->point();
        auto it = point_to_index.find(pt);
        if (it != point_to_index.end())
        {
          tri.vertices[i] = it->second;
        }
        else
        {
          tri.is_valid = false;
          break;
        }
      }
      
      if (tri.is_valid)
      {
        tri.area = compute_triangle_area(tri.vertices[0], tri.vertices[1], tri.vertices[2]);
        if (tri.area > 1e-12)
        {
          elements.push_back(tri);
        }
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
    
    std::cout << "After alpha shape: " << elements.size() << " elements" << std::endl;
  }

  // Compute element shape function derivatives
  void PFEMSloshing::compute_element_data()
  {
    for (auto &elem : elements)
    {
      compute_shape_derivatives(static_cast<unsigned int>(&elem - &elements[0]));
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
    
    // Area using cross product
    double area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    elem.area = std::abs(area);
    
    if (elem.area < 1e-12)
    {
      elem.is_valid = false;
      return;
    }
    
    double inv_2A = 1.0 / (2.0 * area);
    
    // Shape function derivatives (constant for linear triangles)
    // dN0/dx = (y1 - y2) / (2*A), dN0/dy = (x2 - x1) / (2*A)
    // dN1/dx = (y2 - y0) / (2*A), dN1/dy = (x0 - x2) / (2*A)
    // dN2/dx = (y0 - y1) / (2*A), dN2/dy = (x1 - x0) / (2*A)
    
    elem.dN_dx[0] = (y1 - y2) * inv_2A;
    elem.dN_dy[0] = (x2 - x1) * inv_2A;
    
    elem.dN_dx[1] = (y2 - y0) * inv_2A;
    elem.dN_dy[1] = (x0 - x2) * inv_2A;
    
    elem.dN_dx[2] = (y0 - y1) * inv_2A;
    elem.dN_dy[2] = (x1 - x0) * inv_2A;
  }

  // Compute nodal areas and averaged gradients (SPFEM approach)
  void PFEMSloshing::compute_nodal_data()
  {
    // Reset nodal areas
    std::fill(nodal_area.begin(), nodal_area.end(), 0.0);
    
    // Compute nodal areas by summing 1/3 of connected element areas
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      double area = 0.0;
      for (unsigned int e : node_to_elements[n])
      {
        area += elements[e].area / 3.0;
      }
      nodal_area[n] = std::max(area, 1e-12);
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
    
    // Compute viscous forces (simplified form for incompressible flow)
    const double mu = parameters.viscosity;
    
    for (const auto &elem : elements)
    {
      if (!elem.is_valid || elem.area < 1e-12)
        continue;
      
      // Compute strain rate tensor at element centroid
      double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;
      
      for (int i = 0; i < 3; ++i)
      {
        unsigned int node = elem.vertices[i];
        const Point<2> &vel = particles[node].velocity;
        
        dudx += elem.dN_dx[i] * vel[0];
        dudy += elem.dN_dy[i] * vel[0];
        dvdx += elem.dN_dx[i] * vel[1];
        dvdy += elem.dN_dy[i] * vel[1];
      }
      
      // Stress tensor (viscous part only, incompressible)
      // sigma_xx = 2*mu*du/dx, sigma_yy = 2*mu*dv/dy
      // sigma_xy = mu*(du/dy + dv/dx)
      double sigma_xx = 2.0 * mu * dudx;
      double sigma_yy = 2.0 * mu * dvdy;
      double sigma_xy = mu * (dudy + dvdx);
      
      // Distribute forces to nodes
      for (int i = 0; i < 3; ++i)
      {
        unsigned int node = elem.vertices[i];
        double area_contrib = elem.area / 3.0;
        
        // Internal force contribution (negative divergence of stress)
        nodal_force_int[node][0] += (elem.dN_dx[i] * sigma_xx + elem.dN_dy[i] * sigma_xy) * elem.area;
        nodal_force_int[node][1] += (elem.dN_dx[i] * sigma_xy + elem.dN_dy[i] * sigma_yy) * elem.area;
        
        // External forces (gravity)
        double mass = parameters.density * area_contrib;
        nodal_force_ext[node][0] += mass * parameters.gravity_x;
        nodal_force_ext[node][1] += mass * parameters.gravity_y;
      }
    }
  }

  // Explicit time integration (leapfrog scheme)
  void PFEMSloshing::time_integration()
  {
    const double dt = parameters.dt;
    const double rho = parameters.density;
    
    for (unsigned int n = 0; n < n_particles; ++n)
    {
      double mass = rho * nodal_area[n];
      if (mass < 1e-12)
        continue;
      
      // Total force
      double fx = nodal_force_ext[n][0] - nodal_force_int[n][0];
      double fy = nodal_force_ext[n][1] - nodal_force_int[n][1];
      
      // Update acceleration
      particles[n].acceleration = Point<2>(fx / mass, fy / mass);
      
      // Update velocity
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
    const double tank_H = parameters.tank_height;
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
          // Apply friction to horizontal velocity
          p.velocity[0] *= (1.0 - friction);
        }
        p.boundary_type = 1;
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
        p.boundary_type = 1;
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
        p.boundary_type = 1;
      }
      
      // Note: Top is open (free surface)
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
    
    if (area < 1e-12)
      return 1e10; // Very large radius for degenerate triangles
    
    return (a * b * c) / (4.0 * area);
  }

  // Output results to VTK file
  void PFEMSloshing::output_vtk(const std::string &filename)
  {
    std::ofstream out(filename);
    
    out << "# vtk DataFile Version 3.0\n";
    out << "PFEM Sloshing Output\n";
    out << "ASCII\n\n";
    
    out << "DATASET UNSTRUCTURED_GRID\n";
    out << "POINTS " << n_particles << " float\n";
    
    for (const auto &p : particles)
    {
      out << p.position[0] << " " << p.position[1] << " 0.0\n";
    }
    
    out << "\nCELLS " << elements.size() << " " << elements.size() * 4 << "\n";
    for (const auto &elem : elements)
    {
      out << "3 " << elem.vertices[0] << " " << elem.vertices[1] << " " << elem.vertices[2] << "\n";
    }
    
    out << "\nCELL_TYPES " << elements.size() << "\n";
    for (size_t i = 0; i < elements.size(); ++i)
    {
      out << "5\n"; // VTK_TRIANGLE
    }
    
    out << "\nPOINT_DATA " << n_particles << "\n";
    
    out << "VECTORS velocity float\n";
    for (const auto &p : particles)
    {
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
    
    out.close();
    std::cout << "Output written to " << filename << std::endl;
  }

  // Output results
  void PFEMSloshing::output_results(const unsigned int step)
  {
    std::ostringstream filename;
    filename << output_directory << "/sloshing_" << std::setfill('0') << std::setw(6) << step << ".vtk";
    output_vtk(filename.str());
  }

  // Main simulation loop
  void PFEMSloshing::run()
  {
    std::cout << "=== PFEM 2D Free Surface Sloshing Simulation ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Tank: " << parameters.tank_length << " x " << parameters.tank_height << " m" << std::endl;
    std::cout << "  Initial water: " << parameters.initial_water_length << " x " << parameters.initial_water_height << " m" << std::endl;
    std::cout << "  Mesh size: " << parameters.mesh_size << " m" << std::endl;
    std::cout << "  Time step: " << parameters.dt << " s" << std::endl;
    std::cout << "  Total time: " << parameters.total_time << " s" << std::endl;
    std::cout << std::endl;
    
    // Initialize
    Timer timer;
    timer.start();
    
    initialize_particles();
    generate_mesh();
    apply_alpha_shape();
    compute_element_data();
    compute_nodal_data();
    
    // Output initial state
    output_results(0);
    
    // Time stepping loop
    while (current_time < parameters.total_time)
    {
      // Check for remeshing
      if (time_step > 0 && check_remesh_needed())
      {
        std::cout << "Remeshing at step " << time_step << ", time = " << current_time << std::endl;
        generate_mesh();
        apply_alpha_shape();
        compute_element_data();
        compute_nodal_data();
        remesh_count++;
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
        std::cout << "Step " << time_step << ", Time = " << current_time 
                  << " s, Elements = " << elements.size() << std::endl;
        output_results(time_step);
      }
    }
    
    timer.stop();
    
    std::cout << std::endl;
    std::cout << "=== Simulation Complete ===" << std::endl;
    std::cout << "Total steps: " << time_step << std::endl;
    std::cout << "Remesh count: " << remesh_count << std::endl;
    std::cout << "Wall clock time: " << timer.wall_time() << " seconds" << std::endl;
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
      ParameterHandler prm;
      params.declare_parameters(prm);
      prm.parse_input(param_file);
      params.parse_parameters(prm);
      std::cout << "Loaded parameters from " << param_file << std::endl;
    }
    else
    {
      // Use default parameters for dam break / sloshing
      std::cout << "Using default parameters (dam break sloshing test)" << std::endl;
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
