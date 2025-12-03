/**
 * ESPFEM2D - Constitutive Model Implementation
 * 
 * Based on:
 * Zhang W, Liu Y H, Li J H, Yuan W H. ESPFEM2D: A MATLAB 2D explicit smoothed 
 * particle finite element method code for geotechnical large deformation analysis[J].
 * Computational Mechanics, 2024, 74(2):467-484.
 */

#include "spfem_solver.h"
#include <cmath>
#include <array>

namespace ESPFEM2D
{
    void SPFEMSolver::apply_constitutive_model(int material_model,
                                               const std::array<double, 6>& mat_props,
                                               Tensor& stress,
                                               const Tensor& dstrain,
                                               double& plastic_strain_eq) const
    {
        if (material_model == 1)  // Elastic
        {
            elastic_model(mat_props, stress, dstrain);
        }
        else if (material_model == 2)  // Drucker-Prager
        {
            drucker_prager_model(mat_props, stress, dstrain, plastic_strain_eq);
        }
    }

    void SPFEMSolver::elastic_model(const std::array<double, 6>& mat_props,
                                    Tensor& stress,
                                    const Tensor& dstrain) const
    {
        // Material properties
        double E = mat_props[1];   // Young's modulus
        double v = mat_props[2];   // Poisson's ratio
        
        // Elastic stiffness matrix coefficient
        double a = E / ((1 + v) * (1 - 2 * v));
        
        // Elasticity matrix for plane strain (4x4)
        // dee = a * [1-v,   v,   0,        v;
        //             v,  1-v,   0,        v;
        //             0,    0,  (1-2v)/2,  0;
        //             v,    v,   0,       1-v]
        
        // Stress increment: d_sigma = D * d_epsilon
        // sigma_xx, sigma_yy, sigma_xy, sigma_zz
        
        double d11 = a * (1 - v);
        double d12 = a * v;
        double d33 = a * (1 - 2 * v) / 2;
        
        // Update stress
        // stress = [sigma_xx, sigma_yy, sigma_xy, sigma_zz]
        // dstrain = [eps_xx, eps_yy, 2*eps_xy, eps_zz]
        
        double dsig_xx = d11 * dstrain[0] + d12 * dstrain[1] + d12 * dstrain[3];
        double dsig_yy = d12 * dstrain[0] + d11 * dstrain[1] + d12 * dstrain[3];
        double dsig_xy = d33 * dstrain[2];
        double dsig_zz = d12 * dstrain[0] + d12 * dstrain[1] + d11 * dstrain[3];
        
        stress[0] += dsig_xx;
        stress[1] += dsig_yy;
        stress[2] += dsig_xy;
        stress[3] += dsig_zz;
    }

    void SPFEMSolver::drucker_prager_model(const std::array<double, 6>& mat_props,
                                           Tensor& stress,
                                           const Tensor& dstrain,
                                           double& plastic_strain_eq) const
    {
        // Material properties
        double E = mat_props[1];    // Young's modulus
        double v = mat_props[2];    // Poisson's ratio
        double c = mat_props[3];    // Cohesion
        double phi = mat_props[4];  // Friction angle (degrees)
        double psi = mat_props[5];  // Dilation angle (degrees)
        
        // Bulk and shear moduli
        double K = E / (3 * (1 - 2 * v));
        double G = E / (2 * (1 + v));
        
        // Drucker-Prager parameters (outer cone approximation to Mohr-Coulomb)
        double sin_phi = std::sin(phi * M_PI / 180.0);
        double cos_phi = std::cos(phi * M_PI / 180.0);
        double sin_psi = std::sin(psi * M_PI / 180.0);
        
        double alpha = (2 * sin_phi) / (std::sqrt(3.0) * (3 + sin_phi));
        double k_dp = (6 * c * cos_phi) / (std::sqrt(3.0) * (3 + sin_phi));
        double alpha_psi = (2 * sin_psi) / (std::sqrt(3.0) * (3 + sin_psi));
        
        double tenf = 0.0;
        if (alpha > 1e-6)
        {
            tenf = k_dp / (3 * alpha);
        }
        
        // Elastic stiffness matrix for 3D
        double a = E / ((1 + v) * (1 - 2 * v));
        double d11 = a * (1 - v);
        double d12 = a * v;
        double d33 = a * (1 - 2 * v) / 2;
        
        // Convert 2D stress/strain to 3D format for DP model
        // 3D: [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz]
        std::array<double, 6> sigma3d = {stress[0], stress[1], stress[3], stress[2], 0.0, 0.0};
        std::array<double, 6> dstrain3d = {dstrain[0], dstrain[1], dstrain[3], dstrain[2], 0.0, 0.0};
        
        // Trial elastic stress
        sigma3d[0] += d11 * dstrain3d[0] + d12 * dstrain3d[1] + d12 * dstrain3d[2];
        sigma3d[1] += d12 * dstrain3d[0] + d11 * dstrain3d[1] + d12 * dstrain3d[2];
        sigma3d[2] += d12 * dstrain3d[0] + d12 * dstrain3d[1] + d11 * dstrain3d[2];
        sigma3d[3] += d33 * dstrain3d[3];
        
        // First invariant and deviatoric stress
        double I1 = sigma3d[0] + sigma3d[1] + sigma3d[2];
        double p = I1 / 3.0;
        
        // Deviatoric stress
        std::array<double, 6> s = {
            sigma3d[0] - p,
            sigma3d[1] - p,
            sigma3d[2] - p,
            sigma3d[3],
            sigma3d[4],
            sigma3d[5]
        };
        
        // Tension cut-off check
        double dpTi = p - tenf;
        double dlamda = 0.0;
        
        if (dpTi >= 1e-6)
        {
            dlamda = dpTi / K;
            I1 = 3 * tenf;
            p = tenf;
        }
        
        // Second invariant of deviatoric stress
        double J2 = 0.5 * (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] +
                          2 * s[3] * s[3] + 2 * s[4] * s[4] + 2 * s[5] * s[5]);
        double sqrt_J2 = std::sqrt(J2);
        
        // Drucker-Prager yield function
        double dpFi = sqrt_J2 + alpha * I1 - k_dp;
        
        // Plastic strain storage
        std::array<double, 6> dpstrain3d = {0, 0, 0, 0, 0, 0};
        
        if (dpFi >= 1e-6)
        {
            // Plastic return mapping
            dlamda = dpFi / (G + 9 * K * alpha * alpha_psi);
            I1 -= 9 * K * alpha_psi * dlamda;
            p = I1 / 3.0;
            
            double tau = k_dp - alpha * I1 + G * dlamda;
            double ratio = 1.0;
            if (tau > 1e-10)
            {
                ratio = (k_dp - alpha * I1) / (k_dp - alpha * I1 + G * dlamda);
            }
            
            // Scale deviatoric stress
            for (int i = 0; i < 6; ++i)
            {
                s[i] *= ratio;
            }
            
            // Compute plastic strain
            // Elastic strain = C * (sigma_new - sigma_old)
            double sigma3d_old[6] = {stress[0], stress[1], stress[3], stress[2], 0.0, 0.0};
            sigma3d[0] = s[0] + p;
            sigma3d[1] = s[1] + p;
            sigma3d[2] = s[2] + p;
            sigma3d[3] = s[3];
            sigma3d[4] = s[4];
            sigma3d[5] = s[5];
            
            // Compliance matrix C = 1/E * [...]
            double C11 = 1.0 / E;
            double C12 = -v / E;
            double C44 = 2 * (1 + v) / E;
            
            for (int i = 0; i < 3; ++i)
            {
                double dsig = sigma3d[i] - sigma3d_old[i];
                dpstrain3d[i] = dstrain3d[i] - (C11 * dsig);
                for (int j = 0; j < 3; ++j)
                {
                    if (i != j)
                    {
                        dpstrain3d[i] -= C12 * (sigma3d[j] - sigma3d_old[j]);
                    }
                }
            }
            dpstrain3d[3] = dstrain3d[3] - C44 * (sigma3d[3] - sigma3d_old[3]);
        }
        else
        {
            // Elastic - update full stress
            sigma3d[0] = s[0] + p;
            sigma3d[1] = s[1] + p;
            sigma3d[2] = s[2] + p;
            sigma3d[3] = s[3];
        }
        
        // Convert back to 2D
        stress[0] = sigma3d[0];
        stress[1] = sigma3d[1];
        stress[2] = sigma3d[3];
        stress[3] = sigma3d[2];
        
        // Compute equivalent plastic strain increment
        std::array<double, 4> dpstrain = {dpstrain3d[0], dpstrain3d[1], dpstrain3d[3], dpstrain3d[2]};
        double mean_dpstrain = (dpstrain[0] + dpstrain[1] + dpstrain[3]) / 3.0;
        
        std::array<double, 4> dpstrain_dev;
        dpstrain_dev[0] = dpstrain[0] - mean_dpstrain;
        dpstrain_dev[1] = dpstrain[1] - mean_dpstrain;
        dpstrain_dev[3] = dpstrain[3] - mean_dpstrain;
        dpstrain_dev[2] = dpstrain[2];
        
        double dpstrain_eq_inc = std::sqrt(2.0 / 3.0 * (
            dpstrain_dev[0] * dpstrain_dev[0] +
            dpstrain_dev[1] * dpstrain_dev[1] +
            dpstrain_dev[3] * dpstrain_dev[3] +
            0.5 * dpstrain_dev[2] * dpstrain_dev[2]));
        
        plastic_strain_eq += dpstrain_eq_inc;
    }

} // namespace ESPFEM2D
