//! 3D Immersed Topology Optimization Engine in Rust.
//! Implements SIMP material penalization, density filtering, matrix-free compliance,
//! and Optimality Criteria (OC) design updates with Rayon multithreading.

use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct TopOpt3DConfig {
    pub nelx: usize,
    pub nely: usize,
    pub nelz: usize,
    pub volfrac: f64,
    pub penal: f64,
    pub rmin: f64,
    pub max_iter: usize,
    pub tol: f64,
}

impl Default for TopOpt3DConfig {
    fn default() -> Self {
        Self {
            nelx: 24,
            nely: 12,
            nelz: 12,
            volfrac: 0.35,
            penal: 3.0,
            rmin: 1.5,
            max_iter: 20,
            tol: 1e-3,
        }
    }
}

pub struct TopOpt3D {
    pub config: TopOpt3DConfig,
    pub n_elem: usize,
    pub densities: Vec<f64>,
    pub filtered_densities: Vec<f64>,
    pub sensitivities: Vec<f64>,
    pub compliance_history: Vec<f64>,
    // Neighborhood filter weights for each element
    filter_neighbors: Vec<Vec<(usize, f64)>>,
}

impl TopOpt3D {
    pub fn new(config: TopOpt3DConfig) -> Self {
        let n_elem = config.nelx * config.nely * config.nelz;
        let densities = vec![config.volfrac; n_elem];
        let filtered_densities = vec![config.volfrac; n_elem];
        let sensitivities = vec![0.0; n_elem];

        // Precompute Cartesian distance filter weights within rmin
        let rmin = config.rmin;
        let r_ceil = rmin.ceil() as isize;
        let nx = config.nelx as isize;
        let ny = config.nely as isize;
        let nz = config.nelz as isize;

        let filter_neighbors: Vec<Vec<(usize, f64)>> = (0..n_elem)
            .into_par_iter()
            .map(|e| {
                let iz = (e / (config.nelx * config.nely)) as isize;
                let rem = e % (config.nelx * config.nely);
                let iy = (rem / config.nelx) as isize;
                let ix = (rem % config.nelx) as isize;

                let mut nbs = Vec::new();
                let mut sum_h = 0.0;

                for dz in -r_ceil..=r_ceil {
                    let jz = iz + dz;
                    if jz < 0 || jz >= nz { continue; }
                    for dy in -r_ceil..=r_ceil {
                        let jy = iy + dy;
                        if jy < 0 || jy >= ny { continue; }
                        for dx in -r_ceil..=r_ceil {
                            let jx = ix + dx;
                            if jx < 0 || jx >= nx { continue; }

                            let dist = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                            if dist < rmin {
                                let weight = rmin - dist;
                                let neighbor_idx = (jz as usize) * (config.nelx * config.nely)
                                    + (jy as usize) * config.nelx
                                    + (jx as usize);
                                nbs.push((neighbor_idx, weight));
                                sum_h += weight;
                            }
                        }
                    }
                }

                // Normalize weights
                for nb in &mut nbs {
                    nb.1 /= sum_h;
                }
                nbs
            })
            .collect();

        Self {
            config,
            n_elem,
            densities,
            filtered_densities,
            sensitivities,
            compliance_history: Vec::new(),
            filter_neighbors,
        }
    }

    /// Applies the linear density filter: \tilde{\rho}_e = \sum_j H_{ej} \rho_j
    pub fn apply_density_filter(&mut self) {
        let x = &self.densities;
        self.filtered_densities = self.filter_neighbors
            .par_iter()
            .map(|nbs| {
                let mut val = 0.0;
                for &(j, w) in nbs {
                    val += w * x[j];
                }
                val
            })
            .collect();
    }

    /// Evaluates SIMP compliance and element sensitivities for a given displacement field
    pub fn evaluate_compliance_and_sensitivities(&mut self, elem_strain_energies: &[f64]) -> f64 {
        let p = self.config.penal;
        let emin = 1e-6;

        let (compliance, raw_dc): (f64, Vec<f64>) = self.filtered_densities
            .par_iter()
            .zip(elem_strain_energies.par_iter())
            .map(|(&rho_f, &u_ku)| {
                let e_mod = emin + rho_f.powf(p) * (1.0 - emin);
                let c_e = e_mod * u_ku;
                let dc_e = -p * rho_f.powf(p - 1.0) * (1.0 - emin) * u_ku;
                (c_e, dc_e)
            })
            .fold(|| (0.0, Vec::with_capacity(self.n_elem)), |(mut c_acc, mut dc_acc), (c_e, dc_e)| {
                c_acc += c_e;
                dc_acc.push(dc_e);
                (c_acc, dc_acc)
            })
            .reduce(|| (0.0, Vec::new()), |(c1, mut dc1), (c2, dc2)| {
                dc1.extend(dc2);
                (c1 + c2, dc1)
            });

        // Filter sensitivities via adjoint filter
        self.sensitivities = (0..self.n_elem)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for &(j, w) in &self.filter_neighbors[i] {
                    sum += w * raw_dc[j];
                }
                sum
            })
            .collect();

        self.compliance_history.push(compliance);
        compliance
    }

    /// Performs one Optimality Criteria (OC) design variable update
    pub fn step_optimality_criteria(&mut self) -> f64 {
        let move_limit = 0.2;
        let eta = 0.5;
        let mut l1 = 1e-9;
        let mut l2 = 1e9;
        let target_vol = self.config.volfrac * (self.n_elem as f64);

        let mut x_new = vec![0.0; self.n_elem];

        // Bisection loop on Lagrange multiplier lambda
        while (l2 - l1) / (l1 + l2) > 1e-4 {
            let l_mid = 0.5 * (l1 + l2);

            let vol: f64 = self.densities
                .par_iter()
                .zip(self.sensitivities.par_iter())
                .zip(x_new.par_iter_mut())
                .map(|((&x, &dc), x_out)| {
                    // Be = -dc / l_mid
                    let b_e = (-dc / l_mid).max(1e-12);
                    let trial = x * b_e.powf(eta);
                    let val = trial
                        .min(x + move_limit)
                        .max(x - move_limit)
                        .min(1.0)
                        .max(0.001);
                    *x_out = val;
                    val
                })
                .sum();

            if vol > target_vol {
                l1 = l_mid;
            } else {
                l2 = l_mid;
            }
        }

        // Measure max density change
        let mut max_change: f64 = 0.0;
        for i in 0..self.n_elem {
            let diff = (x_new[i] - self.densities[i]).abs();
            if diff > max_change {
                max_change = diff;
            }
        }

        self.densities = x_new;
        self.apply_density_filter();
        max_change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topopt_initialization_and_filter() {
        let config = TopOpt3DConfig {
            nelx: 10,
            nely: 6,
            nelz: 6,
            volfrac: 0.3,
            rmin: 1.5,
            ..Default::default()
        };
        let mut topopt = TopOpt3D::new(config);
        assert_eq!(topopt.n_elem, 360);

        topopt.apply_density_filter();
        let avg_dens: f64 = topopt.filtered_densities.iter().sum::<f64>() / (topopt.n_elem as f64);
        assert!((avg_dens - 0.3).abs() < 1e-4);

        // Dummy strain energy
        let strain_energies = vec![1.0; topopt.n_elem];
        let c = topopt.evaluate_compliance_and_sensitivities(&strain_energies);
        assert!(c > 0.0);

        let change = topopt.step_optimality_criteria();
        assert!(change >= 0.0);
    }
}
