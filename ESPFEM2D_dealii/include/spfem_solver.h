/**
 * ESPFEM2D - Smoothed Particle Finite Element Method 2D Solver
 * 
 * Based on:
 * Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
 * particle finite element method code for geotechnical large deformation analysis[J].
 * Computational Mechanics, 2024, 74(2):467-484.
 * 
 * All copyrights reserved Zhang Wei(1), Liu Yihui(1), Yuan Weihai(2)
 * 1: South China Agricultural University, Guangzhou, China
 * 2: Hohai University, Nanjing, China
 * 
 * C++ implementation using deal.II-style patterns
 */

#ifndef SPFEM_SOLVER_H
#define SPFEM_SOLVER_H

#include <vector>
#include <string>
#include <memory>
#include <array>
#include <map>
#include <set>
#include <cmath>
#include <stdexcept>

namespace ESPFEM2D
{
    /**
     * Simple 2D Point class (mimics deal.II Point<2>)
     */
    class Point2D
    {
    public:
        Point2D() : x(0.0), y(0.0) {}
        Point2D(double x_, double y_) : x(x_), y(y_) {}
        
        double& operator[](int i) { return (i == 0) ? x : y; }
        double operator[](int i) const { return (i == 0) ? x : y; }
        
        Point2D operator+(const Point2D& other) const { return Point2D(x + other.x, y + other.y); }
        Point2D operator-(const Point2D& other) const { return Point2D(x - other.x, y - other.y); }
        Point2D operator*(double s) const { return Point2D(x * s, y * s); }
        Point2D operator/(double s) const { return Point2D(x / s, y / s); }
        
        Point2D& operator+=(const Point2D& other) { x += other.x; y += other.y; return *this; }
        Point2D& operator-=(const Point2D& other) { x -= other.x; y -= other.y; return *this; }
        Point2D& operator*=(double s) { x *= s; y *= s; return *this; }
        Point2D& operator/=(double s) { x /= s; y /= s; return *this; }
        
        double norm() const { return std::sqrt(x * x + y * y); }
        double norm_square() const { return x * x + y * y; }
        
        double x, y;
    };
    
    inline Point2D operator*(double s, const Point2D& p) { return Point2D(s * p.x, s * p.y); }
    
    // Alias for compatibility
    using Point = Point2D;
    
    /**
     * Simple 4-component Tensor (for stress/strain)
     */
    class Tensor4
    {
    public:
        Tensor4() : data{0.0, 0.0, 0.0, 0.0} {}
        
        double& operator[](int i) { return data[i]; }
        double operator[](int i) const { return data[i]; }
        
        std::array<double, 4> data;
    };
    
    // Alias for compatibility
    using Tensor = Tensor4;

    /**
     * Structure to hold simulation parameters
     */
    struct Parameters
    {
        std::string project;
        std::string dir_plus;
        double totaltime;
        double dtime;
        double rank_deficiency;
        int MN;  // Maximum number of related elements
        int vtk_output_interval;
        double damping;
        int rigid_wall_count;
        double space;  // Mesh spacing
        
        // Mesh parameters
        int if_remesh;
        std::vector<std::array<double, 6>> alpha_shape;
        double remesh_threshold;
        double max_remesh_threshold;
        double min_remesh_threshold;
        double remesh_quality_space;
        double Lap_threshold;
        int remesh_cnt;
        double mesh_quality;
        
        // Material parameters
        std::vector<std::array<double, 2>> mat_gravity;  // gx, gy per material
        std::vector<std::array<double, 6>> mat_props;    // rho, E, v, c, phi, psi
        std::vector<int> mat_model;                      // 1=elastic, 2=DP
        
        // Rigid wall parameters (x0, y0, x1, y1, mu)
        std::vector<std::array<double, 5>> walls;
        
        // Current state
        double ctime;
        int istep;
        int node_cnt;
        int element_cnt;
    };

    /**
     * Boundary condition type
     */
    enum class BoundaryType
    {
        Free = -1,      // Free (natural) boundary
        Velocity = 1,   // Fixed velocity boundary
        Force = 3       // Applied force boundary
    };

    /**
     * Structure to hold nodal data
     */
    struct NodalData
    {
        std::vector<Point> coordinates;
        std::vector<Point> displacements;
        std::vector<Point> velocities;
        std::vector<Point> accelerations;
        std::vector<Tensor> stresses;  // sigma_xx, sigma_yy, sigma_xy, sigma_zz
        std::vector<double> plastic_strain_eq;
        std::vector<int> material_id;
        std::vector<std::array<BoundaryType, 2>> boundary_type;
        std::vector<Point> boundary_value;
        std::vector<double> nodal_area;
        std::vector<Tensor> deformation_gradient;  // F11, F12, F21, F22
        
        // Nodal shape function derivatives (smoothed)
        std::vector<std::vector<double>> dNdx;
        std::vector<std::vector<double>> dNdy;
        
        // Related elements and nodes
        std::vector<int> related_element_count;
        std::vector<std::vector<int>> related_elements;
        std::vector<int> related_node_count;
        std::vector<std::vector<int>> related_nodes;
    };

    /**
     * Structure to hold element data
     */
    struct ElementData
    {
        std::vector<std::array<int, 3>> connectivity;  // Node IDs for each triangle
        std::vector<double> areas;
        std::vector<std::array<double, 3>> dNdx;  // Shape function x-derivatives
        std::vector<std::array<double, 3>> dNdy;  // Shape function y-derivatives
    };

    /**
     * Structure for monitoring output
     */
    struct Monitor
    {
        std::vector<std::vector<double>> records;
        std::vector<std::string> items;
        std::vector<int> node_ids;
        std::vector<int> columns;
    };

    /**
     * Main SPFEM Solver class
     */
    class SPFEMSolver
    {
    public:
        SPFEMSolver();
        ~SPFEMSolver() = default;

        /**
         * Initialize the solver with a specific example
         * @param example_id Example number (1-7)
         */
        void initialize(int example_id);

        /**
         * Run the simulation
         */
        void run();

        /**
         * Get current time
         */
        double get_current_time() const { return parameters.ctime; }

    private:
        // Setup functions
        void setup_mesh();
        void initialize_triangulation();
        void apply_alpha_shape();
        void compute_related_elements_and_nodes();

        // Computation functions
        void update_half_velocity();
        void compute_element_data();
        void compute_nodal_data();
        void compute_internal_forces();
        void compute_rank_deficiency_forces();
        void time_integration();
        void apply_contact_wall();
        void remesh_if_needed();
        void laplacian_smoothing();
        double compute_mesh_quality();

        // Output functions
        void output_vtk() const;
        void save_monitor_data();

        // Constitutive model
        void apply_constitutive_model(int material_model,
                                     const std::array<double, 6>& mat_props,
                                     Tensor& stress,
                                     const Tensor& dstrain,
                                     double& plastic_strain_eq) const;
        
        void elastic_model(const std::array<double, 6>& mat_props,
                          Tensor& stress,
                          const Tensor& dstrain) const;
        
        void drucker_prager_model(const std::array<double, 6>& mat_props,
                                  Tensor& stress,
                                  const Tensor& dstrain,
                                  double& plastic_strain_eq) const;

        // Member variables
        Parameters parameters;
        NodalData nodal_data;
        ElementData element_data;
        Monitor monitor;

        // Force vectors
        std::vector<Point> internal_forces;
        std::vector<Point> rank_deficiency_forces;
    };

    /**
     * Example setup functions - each returns initialized parameters and nodal data
     */
    void setup_bar_gravity_vibration(Parameters& par, NodalData& nodal, Monitor& mon);
    void setup_non_cohesive_soil_stage1(Parameters& par, NodalData& nodal, Monitor& mon);
    void setup_non_cohesive_soil_stage2(Parameters& par, NodalData& nodal, Monitor& mon);
    void setup_cohesive_soil_stage1(Parameters& par, NodalData& nodal, Monitor& mon);
    void setup_cohesive_soil_stage2(Parameters& par, NodalData& nodal, Monitor& mon);
    void setup_slope_stage1(Parameters& par, NodalData& nodal, Monitor& mon);
    void setup_slope_stage2(Parameters& par, NodalData& nodal, Monitor& mon);

    /**
     * Load mesh data from file
     */
    void load_mesh_data(const std::string& filename, std::vector<Point>& nodes);

} // namespace ESPFEM2D

#endif // SPFEM_SOLVER_H
