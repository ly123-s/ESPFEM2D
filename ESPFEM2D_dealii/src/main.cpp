/**
 * ESPFEM2D - Main entry point
 * 
 * Based on:
 * Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
 * particle finite element method code for geotechnical large deformation analysis[J].
 * Computational Mechanics, 2024, 74(2):467-484.
 */

#include "spfem_solver.h"
#include <iostream>
#include <cstdlib>
#include <stdexcept>

int main(int argc, char *argv[])
{
    try
    {
        // Default example ID
        int example_id = 1;

        // Parse command line arguments
        if (argc > 1)
        {
            example_id = std::atoi(argv[1]);
        }

        // Validate example ID
        if (example_id < 1 || example_id > 7)
        {
            std::cerr << "Usage: " << argv[0] << " [example_id]\n"
                      << "  example_id: 1 - Oscillation of an elastic cantilever beam\n"
                      << "              2 - Non-cohesive soil collapse stage1\n"
                      << "              3 - Non-cohesive soil collapse stage2\n"
                      << "              4 - Cohesive soil collapse stage1\n"
                      << "              5 - Cohesive soil collapse stage2\n"
                      << "              6 - Failure of a Mohr-Coulomb soil slope stage1\n"
                      << "              7 - Failure of a Mohr-Coulomb soil slope stage2\n";
            return 1;
        }

        std::cout << "=========================================\n"
                  << "ESPFEM2D - Smoothed Particle FEM 2D Solver\n"
                  << "=========================================\n"
                  << "Running example " << example_id << "\n\n";

        // Create solver and run
        ESPFEM2D::SPFEMSolver solver;
        solver.initialize(example_id);
        solver.run();

        std::cout << "\nSimulation completed successfully!\n";
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "Exception caught:\n" << exc.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception caught!\n";
        return 1;
    }
}
