//! Labeled CAD Geometry, Bodies, Faces, and Boundary Condition Dispatcher.
//! Enables named tagging of CAD surfaces/domains for material assignment and BC imposition.

use std::collections::HashMap;
use crate::cut_cell::TriangleMesh3D;
use crate::structural_mesh::StructuralMesh3D;

/// Material properties associated with a CAD body / domain
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialProperties {
    pub name: String,
    pub youngs_modulus: f64, // E [MPa or Pa]
    pub poissons_ratio: f64, // \nu
    pub density: f64,        // \rho [kg/m^3]
}

impl Default for MaterialProperties {
    fn default() -> Self {
        Self {
            name: "StructuralSteel".to_string(),
            youngs_modulus: 2.1e5, // 210 GPa in MPa
            poissons_ratio: 0.30,
            density: 7850.0,
        }
    }
}

/// A labeled CAD boundary face or surface region
#[derive(Clone, Debug)]
pub struct CadFace {
    pub name: String,
    pub triangle_indices: Vec<usize>,
    pub total_area: f64,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
}

/// Boundary condition specification tied to a named CAD face
#[derive(Clone, Debug)]
pub enum BoundaryConditionType {
    /// Strong Dirichlet clamp: u = 0 (or prescribed value) on specified components [x, y, z]
    Dirichlet {
        components: [bool; 3],
        values: [f64; 3],
    },
    /// Surface traction vector [t_x, t_y, t_z] integrated over face area
    NeumannTraction {
        traction: [f64; 3],
    },
    /// Normal pressure p * n (positive = compression inward, negative = tension outward)
    NeumannPressure {
        pressure: f64,
    },
    /// Elastic foundation impedance / spring stiffness k_s [N/m^3 or N/mm^3]
    RobinFoundation {
        normal_stiffness: f64,
        tangential_stiffness: f64,
    },
}

#[derive(Clone, Debug)]
pub struct LabeledBoundaryCondition {
    pub target_face_name: String,
    pub bc_type: BoundaryConditionType,
}

/// A named CAD solid body containing geometry, material properties, and labeled boundary faces
#[derive(Clone, Debug)]
pub struct CadBody {
    pub name: String,
    pub mesh: TriangleMesh3D,
    pub material: MaterialProperties,
    pub faces: HashMap<String, CadFace>,
}

impl CadBody {
    pub fn new(name: impl Into<String>, mesh: TriangleMesh3D, material: MaterialProperties) -> Self {
        let mut body = Self {
            name: name.into(),
            mesh,
            material,
            faces: HashMap::new(),
        };
        body.compute_default_faces();
        body
    }

    /// Automatically computes face normal and area for each triangle
    pub fn triangle_normal_and_area(&self, tri_idx: usize) -> ([f64; 3], f64) {
        let tri = self.mesh.triangles[tri_idx];
        let v0 = self.mesh.vertices[tri[0]];
        let v1 = self.mesh.vertices[tri[1]];
        let v2 = self.mesh.vertices[tri[2]];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let area = 0.5 * len;

        let normal = if len > 1e-14 {
            [cross[0] / len, cross[1] / len, cross[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        (normal, area)
    }

    /// Initial default group containing all triangles as "AllSurfaces"
    fn compute_default_faces(&mut self) {
        let n_tri = self.mesh.triangles.len();
        let mut total_area = 0.0;
        let mut centroid_sum = [0.0; 3];
        let mut normal_sum = [0.0; 3];

        for i in 0..n_tri {
            let (n, a) = self.triangle_normal_and_area(i);
            total_area += a;
            let tri = self.mesh.triangles[i];
            for k in 0..3 {
                centroid_sum[k] += a * (self.mesh.vertices[tri[0]][k] + self.mesh.vertices[tri[1]][k] + self.mesh.vertices[tri[2]][k]) / 3.0;
                normal_sum[k] += a * n[k];
            }
        }

        if total_area > 0.0 {
            centroid_sum[0] /= total_area;
            centroid_sum[1] /= total_area;
            centroid_sum[2] /= total_area;
            let nlen = (normal_sum[0].powi(2) + normal_sum[1].powi(2) + normal_sum[2].powi(2)).sqrt();
            if nlen > 0.0 {
                normal_sum[0] /= nlen; normal_sum[1] /= nlen; normal_sum[2] /= nlen;
            }
        }

        self.faces.insert("AllSurfaces".to_string(), CadFace {
            name: "AllSurfaces".to_string(),
            triangle_indices: (0..n_tri).collect(),
            total_area,
            centroid: centroid_sum,
            normal: normal_sum,
        });
    }

    /// Labels a subset of triangles by a geometric predicate closure:
    /// `predicate(centroid: [f64; 3], normal: [f64; 3]) -> bool`
    pub fn label_face_by_predicate<F>(&mut self, face_name: impl Into<String>, predicate: F)
    where
        F: Fn([f64; 3], [f64; 3]) -> bool,
    {
        let name = face_name.into();
        let mut matched_indices = Vec::new();
        let mut total_area = 0.0;
        let mut centroid_sum = [0.0; 3];
        let mut normal_sum = [0.0; 3];

        for i in 0..self.mesh.triangles.len() {
            let (n, a) = self.triangle_normal_and_area(i);
            let tri = self.mesh.triangles[i];
            let c = [
                (self.mesh.vertices[tri[0]][0] + self.mesh.vertices[tri[1]][0] + self.mesh.vertices[tri[2]][0]) / 3.0,
                (self.mesh.vertices[tri[0]][1] + self.mesh.vertices[tri[1]][1] + self.mesh.vertices[tri[2]][1]) / 3.0,
                (self.mesh.vertices[tri[0]][2] + self.mesh.vertices[tri[1]][2] + self.mesh.vertices[tri[2]][2]) / 3.0,
            ];

            if predicate(c, n) {
                matched_indices.push(i);
                total_area += a;
                for k in 0..3 {
                    centroid_sum[k] += a * c[k];
                    normal_sum[k] += a * n[k];
                }
            }
        }

        if total_area > 0.0 {
            centroid_sum[0] /= total_area;
            centroid_sum[1] /= total_area;
            centroid_sum[2] /= total_area;
            let nlen = (normal_sum[0].powi(2) + normal_sum[1].powi(2) + normal_sum[2].powi(2)).sqrt();
            if nlen > 0.0 {
                normal_sum[0] /= nlen; normal_sum[1] /= nlen; normal_sum[2] /= nlen;
            }
        }

        self.faces.insert(name.clone(), CadFace {
            name,
            triangle_indices: matched_indices,
            total_area,
            centroid: centroid_sum,
            normal: normal_sum,
        });
    }

    /// Parses a Wavefront OBJ string with named objects (`o <body_name>`) and groups (`g <face_name>`)
    pub fn parse_obj(obj_str: &str, default_material: MaterialProperties) -> Result<Self, String> {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let mut body_name = "CadObject".to_string();
        let mut current_group = "Default".to_string();
        let mut group_triangles: HashMap<String, Vec<usize>> = HashMap::new();

        for line in obj_str.lines() {
            let line = line.trim();
            if line.starts_with("o ") {
                body_name = line[2..].trim().to_string();
            } else if line.starts_with("g ") || line.starts_with("usemtl ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() > 1 {
                    current_group = parts[1].to_string();
                }
            } else if line.starts_with("v ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let x: f64 = parts[1].parse().map_err(|e| format!("{:?}", e))?;
                    let y: f64 = parts[2].parse().map_err(|e| format!("{:?}", e))?;
                    let z: f64 = parts[3].parse().map_err(|e| format!("{:?}", e))?;
                    vertices.push([x, y, z]);
                }
            } else if line.starts_with("f ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let mut idxs = Vec::new();
                    for &p in &parts[1..=3] {
                        let token = p.split('/').next().unwrap();
                        let idx: usize = token.parse().map_err(|e| format!("{:?}", e))?;
                        idxs.push(idx.saturating_sub(1)); // OBJ 1-based indexing
                    }
                    let tri_idx = triangles.len();
                    triangles.push([idxs[0], idxs[1], idxs[2]]);
                    group_triangles.entry(current_group.clone()).or_default().push(tri_idx);
                }
            }
        }

        let mesh = TriangleMesh3D::new(vertices, triangles);
        let mut body = Self::new(body_name, mesh, default_material);

        // Populate parsed groups as labeled faces
        for (g_name, t_indices) in group_triangles {
            let mut area = 0.0;
            let mut c_sum = [0.0; 3];
            let mut n_sum = [0.0; 3];
            for &ti in &t_indices {
                let (n, a) = body.triangle_normal_and_area(ti);
                area += a;
                let tri = body.mesh.triangles[ti];
                for k in 0..3 {
                    c_sum[k] += a * (body.mesh.vertices[tri[0]][k] + body.mesh.vertices[tri[1]][k] + body.mesh.vertices[tri[2]][k]) / 3.0;
                    n_sum[k] += a * n[k];
                }
            }
            if area > 0.0 {
                c_sum[0] /= area; c_sum[1] /= area; c_sum[2] /= area;
                let nl = (n_sum[0].powi(2) + n_sum[1].powi(2) + n_sum[2].powi(2)).sqrt();
                if nl > 0.0 { n_sum[0] /= nl; n_sum[1] /= nl; n_sum[2] /= nl; }
            }
            body.faces.insert(g_name.clone(), CadFace {
                name: g_name,
                triangle_indices: t_indices,
                total_area: area,
                centroid: c_sum,
                normal: n_sum,
            });
        }

        Ok(body)
    }
}

/// CAD Multi-Body Assembly Container
#[derive(Clone, Debug)]
pub struct CadAssembly {
    pub bodies: Vec<CadBody>,
    pub boundary_conditions: Vec<LabeledBoundaryCondition>,
}

impl CadAssembly {
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            boundary_conditions: Vec::new(),
        }
    }

    pub fn add_body(&mut self, body: CadBody) {
        self.bodies.push(body);
    }

    pub fn add_boundary_condition(&mut self, bc: LabeledBoundaryCondition) {
        self.boundary_conditions.push(bc);
    }

    /// Finds a face across all bodies in the assembly
    pub fn find_face(&self, face_name: &str) -> Option<(&CadBody, &CadFace)> {
        for body in &self.bodies {
            if let Some(face) = body.faces.get(face_name) {
                return Some((body, face));
            }
        }
        None
    }

    /// Resolves boundary conditions onto a structural octree mesh.
    /// Returns:
    /// - `fixed_dofs`: Master DOFs constrained by Dirichlet conditions
    /// - `prescribed_values`: Prescribed displacement values
    /// - `f_external`: Master load vector integrated from Neumann surface tractions / pressures
    pub fn resolve_boundary_conditions_on_mesh(
        &self,
        mesh: &StructuralMesh3D,
        tolerance: f64,
    ) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let total_dof = mesh.n_master * 3;
        let mut fixed_dofs = Vec::new();
        let mut prescribed_values = Vec::new();
        let mut f_external = vec![0.0; total_dof];

        for bc in &self.boundary_conditions {
            let (body, face) = match self.find_face(&bc.target_face_name) {
                Some(f) => f,
                None => {
                    eprintln!("Warning: Target CAD face '{}' not found in assembly.", bc.target_face_name);
                    continue;
                }
            };

            // Gather all vertex coordinates on the face
            let mut face_verts = Vec::new();
            for &ti in &face.triangle_indices {
                let tri = body.mesh.triangles[ti];
                face_verts.push(body.mesh.vertices[tri[0]]);
                face_verts.push(body.mesh.vertices[tri[1]]);
                face_verts.push(body.mesh.vertices[tri[2]]);
            }

            match &bc.bc_type {
                BoundaryConditionType::Dirichlet { components, values } => {
                    // Identify master nodes whose coordinates match any triangle on the target face within tolerance
                    for (m_idx, &master_node_id) in mesh.master_node_ids.iter().enumerate() {
                        let m_coord = mesh.nodes[master_node_id];
                        let mut is_on_face = false;
                        for &fv in &face_verts {
                            let dist = ((m_coord[0] - fv[0]).powi(2) + (m_coord[1] - fv[1]).powi(2) + (m_coord[2] - fv[2]).powi(2)).sqrt();
                            if dist <= tolerance {
                                is_on_face = true;
                                break;
                            }
                        }

                        if is_on_face {
                            for c in 0..3 {
                                if components[c] {
                                    let dof = m_idx * 3 + c;
                                    fixed_dofs.push(dof);
                                    prescribed_values.push(values[c]);
                                }
                            }
                        }
                    }
                }
                BoundaryConditionType::NeumannTraction { traction } => {
                    // Distribute total traction force F = traction * Area equally to nearby surface master nodes
                    let total_force = [
                        traction[0] * face.total_area,
                        traction[1] * face.total_area,
                        traction[2] * face.total_area,
                    ];
                    let mut matched_master_nodes = Vec::new();
                    for (m_idx, &master_node_id) in mesh.master_node_ids.iter().enumerate() {
                        let m_coord = mesh.nodes[master_node_id];
                        for &fv in &face_verts {
                            let dist = ((m_coord[0] - fv[0]).powi(2) + (m_coord[1] - fv[1]).powi(2) + (m_coord[2] - fv[2]).powi(2)).sqrt();
                            if dist <= tolerance {
                                matched_master_nodes.push(m_idx);
                                break;
                            }
                        }
                    }
                    if !matched_master_nodes.is_empty() {
                        let n_matched = matched_master_nodes.len() as f64;
                        for &m_idx in &matched_master_nodes {
                            f_external[m_idx * 3 + 0] += total_force[0] / n_matched;
                            f_external[m_idx * 3 + 1] += total_force[1] / n_matched;
                            f_external[m_idx * 3 + 2] += total_force[2] / n_matched;
                        }
                    }
                }
                BoundaryConditionType::NeumannPressure { pressure } => {
                    // Normal traction vector = - pressure * n
                    let tract = [
                        -pressure * face.normal[0],
                        -pressure * face.normal[1],
                        -pressure * face.normal[2],
                    ];
                    let total_force = [
                        tract[0] * face.total_area,
                        tract[1] * face.total_area,
                        tract[2] * face.total_area,
                    ];
                    let mut matched_master_nodes = Vec::new();
                    for (m_idx, &master_node_id) in mesh.master_node_ids.iter().enumerate() {
                        let m_coord = mesh.nodes[master_node_id];
                        for &fv in &face_verts {
                            let dist = ((m_coord[0] - fv[0]).powi(2) + (m_coord[1] - fv[1]).powi(2) + (m_coord[2] - fv[2]).powi(2)).sqrt();
                            if dist <= tolerance {
                                matched_master_nodes.push(m_idx);
                                break;
                            }
                        }
                    }
                    if !matched_master_nodes.is_empty() {
                        let n_matched = matched_master_nodes.len() as f64;
                        for &m_idx in &matched_master_nodes {
                            f_external[m_idx * 3 + 0] += total_force[0] / n_matched;
                            f_external[m_idx * 3 + 1] += total_force[1] / n_matched;
                            f_external[m_idx * 3 + 2] += total_force[2] / n_matched;
                        }
                    }
                }
                BoundaryConditionType::RobinFoundation { .. } => {
                    // Elastic foundation impedance operator
                }
            }
        }

        (fixed_dofs, prescribed_values, f_external)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cad_body_predicate_tagging() {
        let mesh = TriangleMesh3D::new_box([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let mat = MaterialProperties {
            name: "Aluminum".to_string(),
            youngs_modulus: 70000.0,
            poissons_ratio: 0.33,
            density: 2700.0,
        };

        let mut body = CadBody::new("CantileverBeam", mesh, mat);

        // Tag -X face (x = 0) as "ClampFace"
        body.label_face_by_predicate("ClampFace", |c, _n| c[0] < 1e-4);
        assert!(body.faces.contains_key("ClampFace"));
        let clamp = body.faces.get("ClampFace").unwrap();
        assert_eq!(clamp.triangle_indices.len(), 2);
        assert!((clamp.total_area - 1.0).abs() < 1e-6);

        // Tag +X face (x = 2) as "LoadFace"
        body.label_face_by_predicate("LoadFace", |c, _n| c[0] > 2.0 - 1e-4);
        assert!(body.faces.contains_key("LoadFace"));
        let load = body.faces.get("LoadFace").unwrap();
        assert_eq!(load.triangle_indices.len(), 2);
        assert!((load.total_area - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_obj_with_groups() {
        let obj_data = r#"
o Bracket
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
g BottomClamp
f 1 2 3
f 1 3 4
g TopLoad
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
f 5 6 7
"#;
        let mat = MaterialProperties::default();
        let body = CadBody::parse_obj(obj_data, mat).unwrap();

        assert_eq!(body.name, "Bracket");
        assert!(body.faces.contains_key("BottomClamp"));
        assert!(body.faces.contains_key("TopLoad"));
        assert_eq!(body.faces.get("BottomClamp").unwrap().triangle_indices.len(), 2);
        assert_eq!(body.faces.get("TopLoad").unwrap().triangle_indices.len(), 1);
    }
}
