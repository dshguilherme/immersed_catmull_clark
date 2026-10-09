//! Structural Hexahedral Mesh on Adaptive Octree with Hanging Node MPC Projection.

use crate::octree::OctreeMesh3D;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct StructuralMesh3D {
    pub nodes: Vec<[f64; 3]>,
    pub elem_nodes: Vec<[usize; 8]>,
    pub is_hanging: Vec<bool>,
    pub master_node_ids: Vec<usize>,
    pub n_nodes: usize,
    pub n_elements: usize,
    pub n_master: usize,
    pub c_matrix: Vec<HashMap<usize, f64>>, // Scalar constraint row: node -> {master_node: weight}
}

impl StructuralMesh3D {
    /// Builds an analysis-ready 3D structural mesh from an octree mesh.
    pub fn from_octree(octree: &OctreeMesh3D) -> Self {
        let leaf_indices = octree.leaf_indices();
        let n_elements = leaf_indices.len();

        let mut node_map: HashMap<[i64; 3], usize> = HashMap::new();
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut elem_nodes: Vec<[usize; 8]> = Vec::with_capacity(n_elements);

        // Grid quantization tolerance for clean node merging
        let tol = 1e-5;
        let quantize = |x: f64| -> i64 { (x / tol).round() as i64 };

        for &leaf_idx in &leaf_indices {
            let b = &octree.cells[leaf_idx].bounds;
            let corners = [
                [b.min[0], b.min[1], b.min[2]],
                [b.max[0], b.min[1], b.min[2]],
                [b.max[0], b.max[1], b.min[2]],
                [b.min[0], b.max[1], b.min[2]],
                [b.min[0], b.min[1], b.max[2]],
                [b.max[0], b.min[1], b.max[2]],
                [b.max[0], b.max[1], b.max[2]],
                [b.min[0], b.max[1], b.max[2]],
            ];

            let mut cell_corner_ids = [0; 8];
            for (c_idx, &pt) in corners.iter().enumerate() {
                let key = [quantize(pt[0]), quantize(pt[1]), quantize(pt[2])];
                let id = *node_map.entry(key).or_insert_with(|| {
                    let new_id = nodes.len();
                    nodes.push(pt);
                    new_id
                });
                cell_corner_ids[c_idx] = id;
            }
            elem_nodes.push(cell_corner_ids);
        }

        let n_nodes = nodes.len();
        let mut is_hanging = vec![false; n_nodes];
        let mut c_matrix: Vec<HashMap<usize, f64>> = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let mut row = HashMap::new();
            row.insert(i, 1.0);
            c_matrix.push(row);
        }

        // Identify edge-midpoint hanging nodes
        let hex_edges: [[usize; 2]; 12] = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ];

        for elem in &elem_nodes {
            for edge in &hex_edges {
                let n1 = elem[edge[0]];
                let n2 = elem[edge[1]];
                let p1 = nodes[n1];
                let p2 = nodes[n2];
                let mid = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2])];
                let key_mid = [quantize(mid[0]), quantize(mid[1]), quantize(mid[2])];

                if let Some(&mid_id) = node_map.get(&key_mid) {
                    if mid_id != n1 && mid_id != n2 {
                        is_hanging[mid_id] = true;
                        let mut row = HashMap::new();
                        row.insert(n1, 0.5);
                        row.insert(n2, 0.5);
                        c_matrix[mid_id] = row;
                    }
                }
            }
        }

        // Identify face-center hanging nodes
        let hex_faces: [[usize; 4]; 6] = [
            [0, 3, 2, 1], [4, 5, 6, 7],
            [0, 1, 5, 4], [3, 7, 6, 2],
            [0, 4, 7, 3], [1, 2, 6, 5],
        ];

        for elem in &elem_nodes {
            for face in &hex_faces {
                let f_nodes = [elem[face[0]], elem[face[1]], elem[face[2]], elem[face[3]]];
                let mut cen = [0.0; 3];
                for &fn_id in &f_nodes {
                    cen[0] += 0.25 * nodes[fn_id][0];
                    cen[1] += 0.25 * nodes[fn_id][1];
                    cen[2] += 0.25 * nodes[fn_id][2];
                }
                let key_cen = [quantize(cen[0]), quantize(cen[1]), quantize(cen[2])];

                if let Some(&cen_id) = node_map.get(&key_cen) {
                    if !f_nodes.contains(&cen_id) {
                        is_hanging[cen_id] = true;
                        let mut row = HashMap::new();
                        for &fn_id in &f_nodes {
                            row.insert(fn_id, 0.25);
                        }
                        c_matrix[cen_id] = row;
                    }
                }
            }
        }

        // Multi-level constraint propagation (C = C * C)
        for _ in 0..4 {
            let mut c_next = c_matrix.clone();
            for i in 0..n_nodes {
                if is_hanging[i] {
                    let mut resolved_row = HashMap::new();
                    for (&dep_node, &weight) in &c_matrix[i] {
                        for (&root_node, &sub_weight) in &c_matrix[dep_node] {
                            *resolved_row.entry(root_node).or_insert(0.0) += weight * sub_weight;
                        }
                    }
                    c_next[i] = resolved_row;
                }
            }
            c_matrix = c_next;
        }

        let master_node_ids: Vec<usize> = (0..n_nodes).filter(|&i| !is_hanging[i]).collect();
        let n_master = master_node_ids.len();

        Self {
            nodes,
            elem_nodes,
            is_hanging,
            master_node_ids,
            n_nodes,
            n_elements,
            n_master,
            c_matrix,
        }
    }

    /// Evaluates 3D linear strain state u_x = 0.01 * x on master nodes and checks patch test error.
    pub fn evaluate_patch_test_error(&self) -> f64 {
        // Linear displacement field: u = (0.01*x, 0.005*y, 0.002*z)
        let mut max_err: f64 = 0.0;
        for i in 0..self.n_nodes {
            let pt = self.nodes[i];
            let exact_ux = 0.01 * pt[0];

            let mut interpolated_ux = 0.0;
            for (&master_id, &weight) in &self.c_matrix[i] {
                let m_pt = self.nodes[master_id];
                interpolated_ux += weight * (0.01 * m_pt[0]);
            }

            let diff = (exact_ux - interpolated_ux).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
        max_err
    }
}
