//! Adaptive 2:1 Balanced Octree Data Structure in pure Rust.

#[derive(Debug, Clone)]
pub struct BoundingBox3D {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl BoundingBox3D {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    pub fn size(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    pub fn contains_point(&self, p: &[f64; 3], tol: f64) -> bool {
        p[0] >= self.min[0] - tol
            && p[0] <= self.max[0] + tol
            && p[1] >= self.min[1] - tol
            && p[1] <= self.max[1] + tol
            && p[2] >= self.min[2] - tol
            && p[2] <= self.max[2] + tol
    }
}

#[derive(Debug, Clone)]
pub struct OctreeCell {
    pub bounds: BoundingBox3D,
    pub level: usize,
    pub is_leaf: bool,
    pub children: Option<[usize; 8]>,
    pub parent: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct OctreeMesh3D {
    pub cells: Vec<OctreeCell>,
    pub root_bounds: BoundingBox3D,
    pub grid_res: [usize; 3],
}

impl OctreeMesh3D {
    /// Creates a regular background grid of root cells.
    pub fn new(bounds: BoundingBox3D, grid_res: [usize; 3]) -> Self {
        let mut cells = Vec::new();
        let dx = (bounds.max[0] - bounds.min[0]) / (grid_res[0] as f64);
        let dy = (bounds.max[1] - bounds.min[1]) / (grid_res[1] as f64);
        let dz = (bounds.max[2] - bounds.min[2]) / (grid_res[2] as f64);

        for iz in 0..grid_res[2] {
            for iy in 0..grid_res[1] {
                for ix in 0..grid_res[0] {
                    let cell_min = [
                        bounds.min[0] + (ix as f64) * dx,
                        bounds.min[1] + (iy as f64) * dy,
                        bounds.min[2] + (iz as f64) * dz,
                    ];
                    let cell_max = [cell_min[0] + dx, cell_min[1] + dy, cell_min[2] + dz];
                    cells.push(OctreeCell {
                        bounds: BoundingBox3D::new(cell_min, cell_max),
                        level: 0,
                        is_leaf: true,
                        children: None,
                        parent: None,
                    });
                }
            }
        }

        Self {
            cells,
            root_bounds: bounds,
            grid_res,
        }
    }

    /// Subdivides the specified leaf cell into 8 children.
    pub fn subdivide_cell(&mut self, cell_idx: usize) {
        if !self.cells[cell_idx].is_leaf {
            return;
        }

        let parent_bounds = self.cells[cell_idx].bounds.clone();
        let parent_level = self.cells[cell_idx].level;
        let c = parent_bounds.center();
        let p_min = parent_bounds.min;
        let p_max = parent_bounds.max;

        let child_bounds = [
            BoundingBox3D::new([p_min[0], p_min[1], p_min[2]], [c[0], c[1], c[2]]),
            BoundingBox3D::new([c[0], p_min[1], p_min[2]], [p_max[0], c[1], c[2]]),
            BoundingBox3D::new([c[0], c[1], p_min[2]], [p_max[0], p_max[1], c[2]]),
            BoundingBox3D::new([p_min[0], c[1], p_min[2]], [c[0], p_max[1], c[2]]),
            BoundingBox3D::new([p_min[0], p_min[1], c[2]], [c[0], c[1], p_max[2]]),
            BoundingBox3D::new([c[0], p_min[1], c[2]], [p_max[0], c[1], p_max[2]]),
            BoundingBox3D::new([c[0], c[1], c[2]], [p_max[0], p_max[1], p_max[2]]),
            BoundingBox3D::new([p_min[0], c[1], c[2]], [c[0], p_max[1], p_max[2]]),
        ];

        let start_idx = self.cells.len();
        let mut child_indices = [0; 8];

        for i in 0..8 {
            let ch_idx = start_idx + i;
            child_indices[i] = ch_idx;
            self.cells.push(OctreeCell {
                bounds: child_bounds[i].clone(),
                level: parent_level + 1,
                is_leaf: true,
                children: None,
                parent: Some(cell_idx),
            });
        }

        self.cells[cell_idx].is_leaf = false;
        self.cells[cell_idx].children = Some(child_indices);
    }

    /// Returns indices of all active leaf cells.
    pub fn leaf_indices(&self) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter_map(|(idx, cell)| if cell.is_leaf { Some(idx) } else { None })
            .collect()
    }

    /// Enforces strict 2:1 balancing across all adjacent leaves.
    pub fn balance_2_to_1(&mut self) {
        loop {
            let mut needs_refinement = Vec::new();
            let leaves = self.leaf_indices();

            for &i in &leaves {
                let cell_i = &self.cells[i];
                let b_i = &cell_i.bounds;
                let lvl_i = cell_i.level;

                for &j in &leaves {
                    if i == j {
                        continue;
                    }
                    let cell_j = &self.cells[j];
                    let lvl_j = cell_j.level;

                    if lvl_j > lvl_i + 1 {
                        // Check if cell_i and cell_j share an interface (touching)
                        let b_j = &cell_j.bounds;
                        let touches = !(b_i.max[0] < b_j.min[0] - 1e-6
                            || b_i.min[0] > b_j.max[0] + 1e-6
                            || b_i.max[1] < b_j.min[1] - 1e-6
                            || b_i.min[1] > b_j.max[1] + 1e-6
                            || b_i.max[2] < b_j.min[2] - 1e-6
                            || b_i.min[2] > b_j.max[2] + 1e-6);

                        if touches && !needs_refinement.contains(&i) {
                            needs_refinement.push(i);
                            break;
                        }
                    }
                }
            }

            if needs_refinement.is_empty() {
                break;
            }

            for cell_idx in needs_refinement {
                self.subdivide_cell(cell_idx);
            }
        }
    }
}
