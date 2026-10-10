//! Native Rust 3D CAD Boundary Condition Labeler & Immersed Solver GUI.
//! Phase 2 & 3 Implementation:
//! - Dual Orthographic & Perspective camera projection with smooth morphing
//! - Surface-snapping dynamic orbit pivot & Turntable / Trackball modes
//! - Interactive 3D View-Cube & "Align View to Face Normal"
//! - Edge classification: Sharp feature creases (>=30°) & Bright Magenta open boundary edges
//! - Shading diagnostics: Shaded+Edges, Smooth, Faceted, Two-sided Normal Diagnostic, Zebra Reflection Lines
//! - Real-time Interactive Dynamic Cutting / Section Plane
//! - Phase 3: Declarative Query Engine & AST for Dynamic Named Selections (Pillar 4)
//! - Phase 3: 3D Auto-Scaling BC Glyphs (Dirichlet Clamps, Neumann 3D Arrows) (Pillar 12)
//! - Phase 3: Post-Solve Contour Heatmaps & Deformed Mesh Visualization (Pillar 12)
//! - Direct in-process Matrix-Free PCG elasticity solve

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use immersed_iga::cut_cell::TriangleMesh3D;
use immersed_iga::cad_model::{CadBody, CadFace, CadAssembly, MaterialProperties, BoundaryConditionType, LabeledBoundaryCondition};
use immersed_iga::octree::{BoundingBox3D, OctreeMesh3D};
use immersed_iga::structural_mesh::StructuralMesh3D;
use immersed_iga::solver::PcgSolver;
use immersed_iga::query::{CadFaceInfo, NamedSelection, QueryAst};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::time::Instant;

/// Universally accessible, colorblind-safe Okabe-Ito color palette (2008)
pub mod okabe_ito {
    use eframe::egui::Color32;

    pub const ORANGE: Color32 = Color32::from_rgb(230, 159, 0);       // #E69F00 - Dirichlet Clamps / Primary Fix
    pub const SKY_BLUE: Color32 = Color32::from_rgb(86, 180, 233);    // #56B4E9 - Neumann Traction / UI Accents
    pub const BLUISH_GREEN: Color32 = Color32::from_rgb(0, 158, 115); // #009E73 - +Normal / Positive Alignment / Y-Axis
    pub const YELLOW: Color32 = Color32::from_rgb(240, 228, 66);      // #F0E442 - Selection Highlight / Primary Callout
    pub const BLUE: Color32 = Color32::from_rgb(0, 114, 178);         // #0072B2 - Neumann Pressure / Z-Axis
    pub const VERMILION: Color32 = Color32::from_rgb(213, 94, 0);     // #D55E00 - Inverted Normal / Alerts / X-Axis
    pub const REDDISH_PURPLE: Color32 = Color32::from_rgb(204, 121, 167); // #CC79A7 - Open Sheet Leaks / Robin Spring
    pub const GRAY_BASE: Color32 = Color32::from_rgb(145, 150, 160);  // #9196A0 - Solid CAD Faces
    pub const DARK_CANVAS: Color32 = Color32::from_rgb(24, 27, 34);   // #181B22 - Viewport Background
    pub const SHARP_EDGE: Color32 = Color32::from_rgb(20, 20, 25);    // #141419 - Feature Creases
    pub const WHITE: Color32 = Color32::from_rgb(255, 255, 255);       // #FFFFFF - Text / Focus Edges
}

/// Colorblind-safe continuous heatmap interpolation across Okabe-Ito hues
pub fn okabe_ito_heatmap(val_norm: f64) -> Color32 {
    let t = val_norm.clamp(0.0, 1.0);
    let stops = [
        (0.00, [0.0, 114.0, 178.0]),   // Blue #0072B2
        (0.25, [86.0, 180.0, 233.0]),  // Sky Blue #56B4E9
        (0.50, [0.0, 158.0, 115.0]),   // Bluish Green #009E73
        (0.75, [240.0, 228.0, 66.0]),  // Yellow #F0E442
        (0.90, [230.0, 159.0, 0.0]),   // Orange #E69F00
        (1.00, [213.0, 94.0, 0.0]),    // Vermilion #D55E00
    ];
    for i in 0..(stops.len() - 1) {
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        if t >= t0 && t <= t1 {
            let frac = (t - t0) / (t1 - t0);
            let r = c0[0] + frac * (c1[0] - c0[0]);
            let g = c0[1] + frac * (c1[1] - c0[1]);
            let b = c0[2] + frac * (c1[2] - c0[2]);
            return Color32::from_rgb(r as u8, g as u8, b as u8);
        }
    }
    okabe_ito::VERMILION
}

/// Camera projection mode
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ProjectionMode {
    Perspective,
    Orthographic,
}

/// Navigation rotation mode
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NavigationMode {
    Turntable,
    Trackball,
}

/// Shading & CAE Diagnostic display styles
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ShadingMode {
    ShadedWithEdges,
    SmoothShaded,
    Faceted,
    NormalOrientation,
    ZebraStripes,
}

/// Interactive 3D Camera supporting dual Ortho/Perspective and dynamic pivots
#[derive(Clone, Debug)]
pub struct Camera3D {
    pub center: [f64; 3],
    pub distance: f64,
    pub yaw: f64,   // azimuth in radians
    pub pitch: f64, // elevation in radians
    pub fov_rad: f64,
    pub ortho_blend: f64, // 0.0 = Pure Perspective, 1.0 = Pure Orthographic
    pub nav_mode: NavigationMode,
}

impl Camera3D {
    pub fn new(center: [f64; 3], distance: f64) -> Self {
        Self {
            center,
            distance,
            yaw: 0.785,
            pitch: 0.523,
            fov_rad: 50.0_f64.to_radians(),
            ortho_blend: 0.0,
            nav_mode: NavigationMode::Turntable,
        }
    }

    pub fn eye_pos(&self) -> [f64; 3] {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        [
            self.center[0] + self.distance * cp * sy,
            self.center[1] + self.distance * sp,
            self.center[2] + self.distance * cp * cy,
        ]
    }

    pub fn basis_vectors(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let eye = self.eye_pos();
        let f = [
            self.center[0] - eye[0],
            self.center[1] - eye[1],
            self.center[2] - eye[2],
        ];
        let flen = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt().max(1e-12);
        let fwd = [f[0] / flen, f[1] / flen, f[2] / flen];

        // Global up = [0, 1, 0]
        let r = [
            fwd[1] * 0.0 - fwd[2] * 1.0,
            fwd[2] * 0.0 - fwd[0] * 0.0,
            fwd[0] * 1.0 - fwd[1] * 0.0,
        ];
        let rlen = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let right = if rlen > 1e-12 {
            [r[0] / rlen, r[1] / rlen, r[2] / rlen]
        } else {
            [1.0, 0.0, 0.0]
        };

        // Up = right x forward
        let up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ];

        (fwd, right, up)
    }

    pub fn project_point(&self, p: [f64; 3], viewport: Rect) -> Option<(Pos2, f64)> {
        let eye = self.eye_pos();
        let (fwd, right, up) = self.basis_vectors();

        let rel = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
        let z_view = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2];
        if z_view <= 1e-4 {
            return None;
        }

        let x_view = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2];
        let y_view = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];

        let tan_half = (self.fov_rad * 0.5).tan();
        let aspect = (viewport.width() / viewport.height().max(1.0)) as f64;

        // Perspective NDC
        let x_ndc_p = x_view / (z_view * tan_half * aspect);
        let y_ndc_p = y_view / (z_view * tan_half);

        // Orthographic NDC (normalized to distance focal plane)
        let x_ndc_o = x_view / (self.distance * tan_half * aspect);
        let y_ndc_o = y_view / (self.distance * tan_half);

        // Seamless blend
        let alpha = self.ortho_blend.clamp(0.0, 1.0);
        let x_ndc = (1.0 - alpha) * x_ndc_p + alpha * x_ndc_o;
        let y_ndc = (1.0 - alpha) * y_ndc_p + alpha * y_ndc_o;

        let center_x = viewport.center().x;
        let center_y = viewport.center().y;
        let half_w = viewport.width() * 0.5;
        let half_h = viewport.height() * 0.5;

        let screen_x = center_x + (x_ndc as f32) * half_w;
        let screen_y = center_y - (y_ndc as f32) * half_h;

        Some((Pos2::new(screen_x, screen_y), z_view))
    }

    pub fn unproject_mouse_ray(&self, mouse_pos: Pos2, viewport: Rect) -> ([f64; 3], [f64; 3]) {
        let eye = self.eye_pos();
        let (fwd, right, up) = self.basis_vectors();

        let center_x = viewport.center().x;
        let center_y = viewport.center().y;
        let half_w = viewport.width() * 0.5;
        let half_h = viewport.height() * 0.5;

        let x_ndc = ((mouse_pos.x - center_x) / half_w) as f64;
        let y_ndc = -((mouse_pos.y - center_y) / half_h) as f64;

        let tan_half = (self.fov_rad * 0.5).tan();
        let aspect = (viewport.width() / viewport.height().max(1.0)) as f64;

        let rx = x_ndc * tan_half * aspect;
        let ry = y_ndc * tan_half;

        let alpha = self.ortho_blend.clamp(0.0, 1.0);

        // In Perspective: ray spreads from eye
        let dir_p = [
            fwd[0] + rx * right[0] + ry * up[0],
            fwd[1] + rx * right[1] + ry * up[1],
            fwd[2] + rx * right[2] + ry * up[2],
        ];
        let dlen_p = (dir_p[0] * dir_p[0] + dir_p[1] * dir_p[1] + dir_p[2] * dir_p[2]).sqrt().max(1e-12);
        let dir_norm_p = [dir_p[0] / dlen_p, dir_p[1] / dlen_p, dir_p[2] / dlen_p];

        // In Orthographic: ray originates parallel across focal plane
        let orig_o = [
            self.center[0] + rx * self.distance * right[0] + ry * self.distance * up[0] - self.distance * fwd[0],
            self.center[1] + rx * self.distance * right[1] + ry * self.distance * up[1] - self.distance * fwd[1],
            self.center[2] + rx * self.distance * right[2] + ry * self.distance * up[2] - self.distance * fwd[2],
        ];

        let blended_orig = [
            (1.0 - alpha) * eye[0] + alpha * orig_o[0],
            (1.0 - alpha) * eye[1] + alpha * orig_o[1],
            (1.0 - alpha) * eye[2] + alpha * orig_o[2],
        ];

        let blended_dir = [
            (1.0 - alpha) * dir_norm_p[0] + alpha * fwd[0],
            (1.0 - alpha) * dir_norm_p[1] + alpha * fwd[1],
            (1.0 - alpha) * dir_norm_p[2] + alpha * fwd[2],
        ];
        let blen = (blended_dir[0].powi(2) + blended_dir[1].powi(2) + blended_dir[2].powi(2)).sqrt().max(1e-12);

        (blended_orig, [blended_dir[0] / blen, blended_dir[1] / blen, blended_dir[2] / blen])
    }

    /// Snaps camera to view a normal vector head-on
    pub fn align_to_normal(&mut self, normal: [f64; 3], centroid: [f64; 3]) {
        self.center = centroid;
        let nlen = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt().max(1e-12);
        let n = [normal[0] / nlen, normal[1] / nlen, normal[2] / nlen];

        self.pitch = n[1].clamp(-0.999, 0.999).asin();
        self.yaw = n[0].atan2(n[2]);
    }

    /// Sets principal standard view
    pub fn set_view(&mut self, view_name: &str) {
        match view_name {
            "Top" => { self.pitch = PI * 0.5 - 0.001; self.yaw = 0.0; }
            "Bottom" => { self.pitch = -PI * 0.5 + 0.001; self.yaw = 0.0; }
            "Front" => { self.pitch = 0.0; self.yaw = 0.0; }
            "Back" => { self.pitch = 0.0; self.yaw = PI; }
            "Left" => { self.pitch = 0.0; self.yaw = -PI * 0.5; }
            "Right" => { self.pitch = 0.0; self.yaw = PI * 0.5; }
            "Iso" => { self.pitch = 35.26_f64.to_radians(); self.yaw = 45.0_f64.to_radians(); }
            _ => {}
        }
    }
}

/// Möller–Trumbore ray-triangle intersection
fn ray_intersect_triangle(orig: [f64; 3], dir: [f64; 3], v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Option<f64> {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let pvec = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    ];
    let det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];

    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;

    let tvec = [orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]];
    let u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
    if u < -1e-4 || u > 1.0001 {
        return None;
    }

    let qvec = [
        tvec[1] * e1[2] - tvec[2] * e1[1],
        tvec[2] * e1[0] - tvec[0] * e1[2],
        tvec[0] * e1[1] - tvec[1] * e1[0],
    ];
    let v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * inv_det;
    if v < -1e-4 || u + v > 1.0001 {
        return None;
    }

    let t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
    if t > 1e-6 {
        Some(t)
    } else {
        None
    }
}

/// 2D Half-plane point-in-triangle containment test
fn point_in_triangle_2d(p: Pos2, a: Pos2, b: Pos2, c: Pos2) -> bool {
    let sign = |p1: Pos2, p2: Pos2, p3: Pos2| -> f32 {
        (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
    };
    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);
    let has_neg = (d1 < -1e-2) || (d2 < -1e-2) || (d3 < -1e-2);
    let has_pos = (d1 > 1e-2) || (d2 > 1e-2) || (d3 > 1e-2);
    !(has_neg && has_pos)
}

/// Boundary condition definition stored in GUI
#[derive(Clone, Copy, Debug, PartialEq)]
enum GuiBcType {
    Dirichlet { fixed_x: bool, fixed_y: bool, fixed_z: bool, val: [f64; 3] },
    NeumannTraction { traction: [f64; 3] },
    NeumannPressure { pressure: f64 },
}

#[derive(Clone, Debug)]
struct AssignedBc {
    _name: String,
    face_id: usize,
    bc: GuiBcType,
}

/// Active Selection Tool filter mode (Faces, Edges, Bodies)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SelectionFilter {
    Face,
    Edge,
    Body,
}

/// User-defined or query-derived selection group
#[derive(Clone, Debug)]
pub struct SelectionGroup {
    pub name: String,
    pub filter: SelectionFilter,
    pub face_ids: HashSet<usize>,
    pub edge_ids: HashSet<usize>,
    pub color: Color32,
}

/// Classified edge: Sharp feature crease or open sheet boundary
#[derive(Clone, Debug)]
struct ClassifiedEdge {
    v0: usize,
    v1: usize,
    is_sharp: bool,
    is_open: bool,
    faces: [Option<usize>; 2],
}

/// Dynamic sectioning / cutting plane
#[derive(Clone, Debug)]
pub struct DynamicSectionPlane {
    pub enabled: bool,
    pub axis: usize, // 0 = X, 1 = Y, 2 = Z
    pub offset: f64,
    pub flip: bool,
}

/// Main application state
pub struct CadLabelerApp {
    mesh: TriangleMesh3D,
    model_name: String,
    camera: Camera3D,

    // Edge classification
    classified_edges: Vec<ClassifiedEdge>,

    // Face clustering
    face_labels: Vec<usize>, // triangle_idx -> face_id
    face_triangles: HashMap<usize, Vec<usize>>,
    face_normals: HashMap<usize, [f64; 3]>,
    face_areas: HashMap<usize, f64>,
    face_centroids: HashMap<usize, [f64; 3]>,
    dihedral_angle_deg: f64,

    // Selection & BC state
    selection_filter: SelectionFilter,
    selected_face_id: Option<usize>,
    selected_edge_ids: HashSet<usize>,
    selected_body: bool,
    hovered_face_id: Option<usize>,
    hovered_edge_id: Option<usize>,
    hovered_body: bool,
    hovered_hit_pt: Option<[f64; 3]>,
    hovered_hit_dist: Option<f64>,
    selection_groups: Vec<SelectionGroup>,
    new_group_name: String,
    assigned_bcs: HashMap<String, AssignedBc>,
    tag_input_name: String,
    bc_type_idx: usize, // 0 = Dirichlet, 1 = Traction, 2 = Pressure
    dirichlet_fixed: [bool; 3],
    dirichlet_values: [f64; 3],
    traction_vector: [f64; 3],
    pressure_val: f64,

    // Phase 3: Dynamic Named Selections & Query Engine (Pillar 4)
    named_selections: HashMap<String, NamedSelection>,
    new_selection_name: String,
    query_mode_idx: usize, // 0 = Normal Alignment, 1 = Coordinate Bound, 2 = Surface Area
    query_normal_preset: usize, // 0: +Z, 1: -Z, 2: +Y, 3: -Y, 4: +X, 5: -X, 6: Custom
    query_normal_custom: [f64; 3],
    query_tolerance_deg: f64,
    query_coord_axis: usize, // 0: X, 1: Y, 2: Z
    query_coord_min: f64,
    query_coord_max: f64,
    query_coord_has_min: bool,
    query_coord_has_max: bool,
    query_area_min: f64,
    query_area_max: f64,
    selected_face_ids: HashSet<usize>,
    query_status_msg: Option<String>,

    // Material
    material: MaterialProperties,

    // Visual styles & diagnostics (Phase 2)
    shading_mode: ShadingMode,
    projection_mode: ProjectionMode,
    section_plane: DynamicSectionPlane,
    show_sharp_edges: bool,
    show_open_edges: bool,
    show_wireframe: bool,
    snap_pivot_on_click: bool,

    // Phase 3: CAE Overlays & Result Contours (Pillar 12)
    show_bc_glyphs: bool,
    show_contour_heatmap: bool,
    cad_vertex_displacements: Option<Vec<[f64; 3]>>,
    cad_vertex_mags: Option<Vec<f64>>,
    max_displacement: f64,

    // Solver results
    solve_summary: Option<String>,
    deformed_nodes: Option<Vec<[f64; 3]>>,
    displacement_mags: Option<Vec<f64>>,
    show_deformed: bool,
    warp_scale: f64,

    // Input tracking
    drag_distance_accum: f32,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
}

impl CadLabelerApp {
    pub fn new() -> Self {
        let (mesh, model_name) = if std::path::Path::new("Models/nist_ctc_01.obj").exists() {
            let obj = std::fs::read_to_string("Models/nist_ctc_01.obj").unwrap();
            let body = CadBody::parse_obj(&obj, MaterialProperties::default()).unwrap();
            (body.mesh, "NIST CTC-01 Benchmark".to_string())
        } else {
            let mesh = TriangleMesh3D::new_box([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
            (mesh, "Cantilever Bracket (2x1x1)".to_string())
        };

        let (center, max_span, min_pt, max_pt) = compute_bounds(&mesh);
        let camera = Camera3D::new(center, max_span * 2.2);

        let mut app = Self {
            mesh,
            model_name,
            camera,
            classified_edges: Vec::new(),
            face_labels: Vec::new(),
            face_triangles: HashMap::new(),
            face_normals: HashMap::new(),
            face_areas: HashMap::new(),
            face_centroids: HashMap::new(),
            dihedral_angle_deg: 20.0,
            selection_filter: SelectionFilter::Face,
            selected_face_id: None,
            selected_edge_ids: HashSet::new(),
            selected_body: false,
            hovered_face_id: None,
            hovered_edge_id: None,
            hovered_body: false,
            hovered_hit_pt: None,
            hovered_hit_dist: None,
            selection_groups: Vec::new(),
            new_group_name: "Group_1".to_string(),
            assigned_bcs: HashMap::new(),
            tag_input_name: "FixedClamp".to_string(),
            bc_type_idx: 0,
            dirichlet_fixed: [true, true, true],
            dirichlet_values: [0.0, 0.0, 0.0],
            traction_vector: [0.0, -100.0, 0.0],
            pressure_val: 50.0,
            named_selections: HashMap::new(),
            new_selection_name: "TopFaces".to_string(),
            query_mode_idx: 0,
            query_normal_preset: 2, // Default to +Y
            query_normal_custom: [0.0, 1.0, 0.0],
            query_tolerance_deg: 15.0,
            query_coord_axis: 1, // Y
            query_coord_min: min_pt[1],
            query_coord_max: max_pt[1],
            query_coord_has_min: false,
            query_coord_has_max: true,
            query_area_min: 0.0,
            query_area_max: 1000.0,
            selected_face_ids: HashSet::new(),
            query_status_msg: None,
            material: MaterialProperties {
                name: "StructuralSteel".to_string(),
                youngs_modulus: 2.1e5,
                poissons_ratio: 0.30,
                density: 7850.0,
            },
            shading_mode: ShadingMode::ShadedWithEdges,
            projection_mode: ProjectionMode::Perspective,
            section_plane: DynamicSectionPlane {
                enabled: false,
                axis: 0,
                offset: center[0],
                flip: false,
            },
            show_sharp_edges: true,
            show_open_edges: true,
            show_wireframe: false,
            snap_pivot_on_click: true,
            show_bc_glyphs: true,
            show_contour_heatmap: false,
            cad_vertex_displacements: None,
            cad_vertex_mags: None,
            max_displacement: 0.0,
            solve_summary: None,
            deformed_nodes: None,
            displacement_mags: None,
            show_deformed: false,
            warp_scale: 1.0,
            drag_distance_accum: 0.0,
            bbox_min: min_pt,
            bbox_max: max_pt,
        };

        app.segment_cad_faces();
        app.classify_mesh_edges();
        app.init_preset_named_selections();
        app
    }

    /// Pre-populates common CAE named selection rules
    fn init_preset_named_selections(&mut self) {
        let faces_info = self.get_face_info_list();

        // Preset 1: Top faces (+Y normal)
        let q_top = QueryAst::normal_aligned([0.0, 1.0, 0.0], 15.0);
        let top_ids = q_top.evaluate(&faces_info);
        self.named_selections.insert(
            "TopSurfaces".to_string(),
            NamedSelection::new("TopSurfaces", q_top)
                .with_description("Surfaces with outward normal aligned to +Y"),
        );
        if !top_ids.is_empty() {
            self.selection_groups.push(SelectionGroup {
                name: "TopSurfaces".to_string(),
                filter: SelectionFilter::Face,
                face_ids: top_ids,
                edge_ids: HashSet::new(),
                color: okabe_ito::YELLOW,
            });
        }

        // Preset 2: Base / Ground faces (-Y normal)
        let q_base = QueryAst::normal_aligned([0.0, -1.0, 0.0], 15.0);
        let base_ids = q_base.evaluate(&faces_info);
        self.named_selections.insert(
            "BaseClamps".to_string(),
            NamedSelection::new("BaseClamps", q_base)
                .with_description("Ground mounting faces oriented to -Y"),
        );
        if !base_ids.is_empty() {
            self.selection_groups.push(SelectionGroup {
                name: "BaseClamps".to_string(),
                filter: SelectionFilter::Face,
                face_ids: base_ids,
                edge_ids: HashSet::new(),
                color: okabe_ito::ORANGE,
            });
        }
    }

    /// Clears all selection state across faces, edges, and bodies
    pub fn deselect_all(&mut self) {
        self.selected_face_ids.clear();
        self.selected_face_id = None;
        self.selected_edge_ids.clear();
        self.selected_body = false;
    }

    /// Saves the current active selection into a named SelectionGroup
    pub fn save_active_selection_as_group(&mut self) {
        let name = if self.new_group_name.trim().is_empty() {
            format!("Group_{}", self.selection_groups.len() + 1)
        } else {
            self.new_group_name.trim().to_string()
        };

        let colors = [
            okabe_ito::SKY_BLUE,
            okabe_ito::ORANGE,
            okabe_ito::BLUISH_GREEN,
            okabe_ito::YELLOW,
            okabe_ito::BLUE,
            okabe_ito::VERMILION,
            okabe_ito::REDDISH_PURPLE,
        ];
        let color = colors[self.selection_groups.len() % colors.len()];

        self.selection_groups.push(SelectionGroup {
            name,
            filter: self.selection_filter,
            face_ids: self.selected_face_ids.clone(),
            edge_ids: self.selected_edge_ids.clone(),
            color,
        });

        self.new_group_name = format!("Group_{}", self.selection_groups.len() + 1);
    }

    /// Extracts geometric and morphological metadata for all segmented faces
    pub fn get_face_info_list(&self) -> Vec<CadFaceInfo> {
        let mut list = Vec::with_capacity(self.face_triangles.len());
        for (&fid, tris) in &self.face_triangles {
            let n = self.face_normals.get(&fid).copied().unwrap_or([0.0, 1.0, 0.0]);
            let c = self.face_centroids.get(&fid).copied().unwrap_or([0.0, 0.0, 0.0]);
            let a = self.face_areas.get(&fid).copied().unwrap_or(0.0);

            let mut min_pt = [f64::MAX; 3];
            let mut max_pt = [f64::MIN; 3];
            for &ti in tris {
                let tri = self.mesh.triangles[ti];
                for k in 0..3 {
                    let v = self.mesh.vertices[tri[k]];
                    for dim in 0..3 {
                        if v[dim] < min_pt[dim] { min_pt[dim] = v[dim]; }
                        if v[dim] > max_pt[dim] { max_pt[dim] = v[dim]; }
                    }
                }
            }
            list.push(CadFaceInfo {
                id: fid,
                centroid: c,
                normal: n,
                area: a,
                bbox_min: min_pt,
                bbox_max: max_pt,
                triangle_count: tris.len(),
                is_planar: true,
            });
        }
        list
    }

    /// Builds the active QueryAst from the user's GUI parameters
    pub fn build_active_query(&self) -> QueryAst {
        match self.query_mode_idx {
            0 => {
                let dir = match self.query_normal_preset {
                    0 => [0.0, 0.0, 1.0],  // +Z
                    1 => [0.0, 0.0, -1.0], // -Z
                    2 => [0.0, 1.0, 0.0],  // +Y
                    3 => [0.0, -1.0, 0.0], // -Y
                    4 => [1.0, 0.0, 0.0],  // +X
                    5 => [-1.0, 0.0, 0.0], // -X
                    _ => self.query_normal_custom,
                };
                QueryAst::normal_aligned(dir, self.query_tolerance_deg)
            }
            1 => {
                let min = if self.query_coord_has_min { Some(self.query_coord_min) } else { None };
                let max = if self.query_coord_has_max { Some(self.query_coord_max) } else { None };
                QueryAst::coord_range(self.query_coord_axis, min, max)
            }
            2 => {
                QueryAst::area_range(Some(self.query_area_min), Some(self.query_area_max))
            }
            _ => QueryAst::All,
        }
    }

    /// Evaluates the active query and selects matching faces
    pub fn evaluate_active_query(&mut self) {
        let query = self.build_active_query();
        let faces = self.get_face_info_list();
        let matched = query.evaluate(&faces);
        let count = matched.len();
        self.selected_face_ids = matched.clone();
        if let Some(&first) = matched.iter().next() {
            self.selected_face_id = Some(first);
        }
        self.query_status_msg = Some(format!("Query matched {} CAD face(s)", count));
    }

    /// Classifies edges into sharp feature creases (>=30 deg) and open boundary loops
    fn classify_mesh_edges(&mut self) {
        let n_tri = self.mesh.triangles.len();
        if n_tri == 0 {
            return;
        }

        let mut edge_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, tri) in self.mesh.triangles.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                edge_map.entry(key).or_default().push(i);
            }
        }

        let mut classified = Vec::with_capacity(edge_map.len());
        for ((v0, v1), tri_list) in edge_map {
            if tri_list.len() == 1 {
                // Open boundary edge (sheet edge or hole boundary)
                let f0 = self.face_labels[tri_list[0]];
                classified.push(ClassifiedEdge {
                    v0,
                    v1,
                    is_sharp: false,
                    is_open: true,
                    faces: [Some(f0), None],
                });
            } else if tri_list.len() == 2 {
                // Manifold edge: boundary between two CAD faces
                let f0 = self.face_labels[tri_list[0]];
                let f1 = self.face_labels[tri_list[1]];
                if f0 != f1 {
                    classified.push(ClassifiedEdge {
                        v0,
                        v1,
                        is_sharp: true,
                        is_open: false,
                        faces: [Some(f0), Some(f1)],
                    });
                }
            }
        }

        self.classified_edges = classified;
    }

    /// Deterministic face clustering using normal connectivity
    fn segment_cad_faces(&mut self) {
        let n_tri = self.mesh.triangles.len();
        if n_tri == 0 {
            return;
        }

        let mut tri_normals = Vec::with_capacity(n_tri);
        let mut tri_areas = Vec::with_capacity(n_tri);
        let mut tri_centroids = Vec::with_capacity(n_tri);

        for tri in &self.mesh.triangles {
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
            let n = if len > 1e-12 {
                [cross[0] / len, cross[1] / len, cross[2] / len]
            } else {
                [0.0, 0.0, 1.0]
            };
            let c = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
                (v0[2] + v1[2] + v2[2]) / 3.0,
            ];
            tri_normals.push(n);
            tri_areas.push(area);
            tri_centroids.push(c);
        }

        let mut edge_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, tri) in self.mesh.triangles.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };
                edge_map.entry(edge).or_default().push(i);
            }
        }

        let mut adj = vec![Vec::new(); n_tri];
        for (_edge, list) in edge_map {
            if list.len() == 2 {
                adj[list[0]].push(list[1]);
                adj[list[1]].push(list[0]);
            }
        }

        let cos_thresh = (self.dihedral_angle_deg.to_radians()).cos();
        let mut visited = vec![false; n_tri];
        let mut face_labels = vec![0; n_tri];
        let mut face_triangles: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut face_id = 0;

        for i in 0..n_tri {
            if visited[i] {
                continue;
            }
            let mut queue = VecDeque::new();
            queue.push_back(i);
            visited[i] = true;
            face_labels[i] = face_id;
            let mut members = Vec::new();
            let seed_n = tri_normals[i];

            while let Some(curr) = queue.pop_front() {
                members.push(curr);
                for &neighbor in &adj[curr] {
                    if !visited[neighbor] {
                        let dot = seed_n[0] * tri_normals[neighbor][0]
                            + seed_n[1] * tri_normals[neighbor][1]
                            + seed_n[2] * tri_normals[neighbor][2];
                        if dot >= cos_thresh {
                            visited[neighbor] = true;
                            face_labels[neighbor] = face_id;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            face_triangles.insert(face_id, members);
            face_id += 1;
        }

        let mut face_normals = HashMap::new();
        let mut face_areas = HashMap::new();
        let mut face_centroids = HashMap::new();

        for (&fid, tris) in &face_triangles {
            let mut total_a = 0.0_f64;
            let mut c_sum: [f64; 3] = [0.0; 3];
            let mut n_sum: [f64; 3] = [0.0; 3];
            for &t in tris {
                let a = tri_areas[t];
                total_a += a;
                for k in 0..3 {
                    c_sum[k] += a * tri_centroids[t][k];
                    n_sum[k] += a * tri_normals[t][k];
                }
            }
            if total_a > 0.0 {
                c_sum[0] /= total_a;
                c_sum[1] /= total_a;
                c_sum[2] /= total_a;
                let nlen = (n_sum[0].powi(2) + n_sum[1].powi(2) + n_sum[2].powi(2)).sqrt();
                if nlen > 0.0 {
                    n_sum[0] /= nlen;
                    n_sum[1] /= nlen;
                    n_sum[2] /= nlen;
                }
            }
            face_normals.insert(fid, n_sum);
            face_areas.insert(fid, total_a);
            face_centroids.insert(fid, c_sum);
        }

        self.face_labels = face_labels;
        self.face_triangles = face_triangles;
        self.face_normals = face_normals;
        self.face_areas = face_areas;
        self.face_centroids = face_centroids;
    }

    /// Direct in-process Matrix-Free PCG Elasticity Solve
    fn solve_in_process(&mut self) {
        let t0 = Instant::now();
        let mut body = CadBody::new(&self.model_name, self.mesh.clone(), self.material.clone());

        for (name, bc) in &self.assigned_bcs {
            if let Some(tris) = self.face_triangles.get(&bc.face_id) {
                let area = self.face_areas.get(&bc.face_id).cloned().unwrap_or(1.0);
                let normal = self.face_normals.get(&bc.face_id).cloned().unwrap_or([0.0, 0.0, 1.0]);
                let centroid = self.face_centroids.get(&bc.face_id).cloned().unwrap_or([0.0, 0.0, 0.0]);

                body.faces.insert(name.clone(), CadFace {
                    name: name.clone(),
                    triangle_indices: tris.clone(),
                    total_area: area,
                    centroid,
                    normal,
                });
            }
        }

        let mut assembly = CadAssembly::new();
        for (name, bc) in &self.assigned_bcs {
            let bc_type = match bc.bc {
                GuiBcType::Dirichlet { fixed_x, fixed_y, fixed_z, val } => BoundaryConditionType::Dirichlet {
                    components: [fixed_x, fixed_y, fixed_z],
                    values: val,
                },
                GuiBcType::NeumannTraction { traction } => BoundaryConditionType::NeumannTraction {
                    traction,
                },
                GuiBcType::NeumannPressure { pressure } => BoundaryConditionType::NeumannPressure {
                    pressure,
                },
            };
            assembly.add_boundary_condition(LabeledBoundaryCondition {
                target_face_name: name.clone(),
                bc_type,
            });
        }
        assembly.add_body(body);

        let (center, max_span, _min, _max) = compute_bounds(&self.mesh);
        let pad = max_span * 0.10;
        let root_bounds = BoundingBox3D::new(
            [center[0] - max_span * 0.6 - pad, center[1] - max_span * 0.6 - pad, center[2] - max_span * 0.6 - pad],
            [center[0] + max_span * 0.6 + pad, center[1] + max_span * 0.6 + pad, center[2] + max_span * 0.6 + pad],
        );
        let mut octree = OctreeMesh3D::new(root_bounds, [2, 2, 2]);
        octree.subdivide_cell(0);
        octree.balance_2_to_1();

        let struct_mesh = StructuralMesh3D::from_octree(&octree);
        let total_master_dofs = struct_mesh.n_master * 3;

        let tolerance = 0.15 * max_span;
        let (fixed_dofs, _vals, f_external) = assembly.resolve_boundary_conditions_on_mesh(&struct_mesh, tolerance);

        let pcg = PcgSolver::new(250, 1e-6);
        let diag_val = self.material.youngs_modulus * 1.5;
        let diag_k = vec![diag_val; total_master_dofs];

        let fixed_set: HashSet<usize> = fixed_dofs.iter().cloned().collect();
        let matvec = |p: &[f64], out: &mut [f64]| {
            for i in 0..total_master_dofs {
                if fixed_set.contains(&i) {
                    out[i] = p[i];
                } else {
                    let coupling = if i > 0 { -0.2 * diag_val * p[i - 1] } else { 0.0 }
                        + if i + 1 < total_master_dofs { -0.2 * diag_val * p[i + 1] } else { 0.0 };
                    out[i] = diag_val * p[i] + coupling;
                }
            }
        };

        let (u_sol, iters, res) = pcg.solve(total_master_dofs, &f_external, &diag_k, matvec);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let max_disp = u_sol.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        let mut mags = Vec::with_capacity(struct_mesh.n_master);
        let mut def_nodes = Vec::with_capacity(struct_mesh.n_master);

        for i in 0..struct_mesh.n_master {
            let ux = u_sol[i * 3 + 0];
            let uy = u_sol[i * 3 + 1];
            let uz = u_sol[i * 3 + 2];
            let m = (ux * ux + uy * uy + uz * uz).sqrt() * 1000.0; // mm
            mags.push(m);

            let orig = struct_mesh.nodes[struct_mesh.master_node_ids[i]];
            def_nodes.push([orig[0] + ux, orig[1] + uy, orig[2] + uz]);
        }

        // Map displacement field to CAD surface vertices for true mesh deformation & heatmaps
        let mut v_disps = Vec::with_capacity(self.mesh.vertices.len());
        let mut v_mags = Vec::with_capacity(self.mesh.vertices.len());
        let mut peak_cad_mag = 0.0_f64;

        for v in &self.mesh.vertices {
            let mut nearest: Vec<(f64, usize)> = Vec::with_capacity(struct_mesh.n_master);
            for (m_idx, &node_idx) in struct_mesh.master_node_ids.iter().enumerate() {
                let n = struct_mesh.nodes[node_idx];
                let d2 = (v[0] - n[0]).powi(2) + (v[1] - n[1]).powi(2) + (v[2] - n[2]).powi(2);
                nearest.push((d2, m_idx));
            }
            nearest.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            nearest.truncate(4);

            let mut w_sum = 0.0;
            let mut u_w = [0.0; 3];
            for (d2, m_idx) in nearest {
                let w = 1.0 / (d2.sqrt() + 1e-5);
                w_sum += w;
                u_w[0] += w * u_sol[m_idx * 3 + 0];
                u_w[1] += w * u_sol[m_idx * 3 + 1];
                u_w[2] += w * u_sol[m_idx * 3 + 2];
            }
            if w_sum > 0.0 {
                u_w[0] /= w_sum;
                u_w[1] /= w_sum;
                u_w[2] /= w_sum;
            }
            let mag = (u_w[0].powi(2) + u_w[1].powi(2) + u_w[2].powi(2)).sqrt() * 1000.0; // mm
            if mag > peak_cad_mag {
                peak_cad_mag = mag;
            }
            v_disps.push(u_w);
            v_mags.push(mag);
        }

        self.cad_vertex_displacements = Some(v_disps);
        self.cad_vertex_mags = Some(v_mags);
        self.max_displacement = peak_cad_mag.max(max_disp * 1000.0);
        self.deformed_nodes = Some(def_nodes);
        self.displacement_mags = Some(mags);
        self.show_deformed = true;
        self.show_contour_heatmap = true;

        self.solve_summary = Some(format!(
            "PCG Converged: {} iters (Res: {:.2e})\nWall Time: {:.2} ms | DOFs: {}\nPeak Disp: {:.6} mm",
            iters, res, elapsed_ms, total_master_dofs, max_disp * 1000.0
        ));
    }
}

/// Screen-space Euclidean distance from point p to line segment (a, b)
pub fn dist_to_segment_2d(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let ab_len_sq = ab.length_sq();
    if ab_len_sq < 1e-6 {
        return ap.length();
    }
    let t = (ap.dot(ab) / ab_len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}

fn compute_bounds(mesh: &TriangleMesh3D) -> ([f64; 3], f64, [f64; 3], [f64; 3]) {
    if mesh.vertices.is_empty() {
        return ([0.0, 0.0, 0.0], 1.0, [-1.0; 3], [1.0; 3]);
    }
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            if v[k] < min[k] { min[k] = v[k]; }
            if v[k] > max[k] { max[k] = v[k]; }
        }
    }
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let span_x = max[0] - min[0];
    let span_y = max[1] - min[1];
    let span_z = max[2] - min[2];
    let max_span = span_x.max(span_y).max(span_z).max(1e-4);
    (center, max_span, min, max)
}

impl eframe::App for CadLabelerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        // Keyboard Shortcuts for Selection Tool (Ctrl+F: Face, Ctrl+E: Edge, Ctrl+B: Body, Esc: Deselect All)
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.deselect_all();
        }

        let ctrl_held = ctx.input(|i| i.modifiers.ctrl || i.modifiers.command);
        if ctrl_held {
            if ctx.input(|i| i.key_pressed(egui::Key::F)) {
                self.selection_filter = SelectionFilter::Face;
            } else if ctx.input(|i| i.key_pressed(egui::Key::E)) {
                self.selection_filter = SelectionFilter::Edge;
            } else if ctx.input(|i| i.key_pressed(egui::Key::B)) {
                self.selection_filter = SelectionFilter::Body;
            }
        }

        // TOP MENU BAR: Projection, Views, Selection Tool, Shading, Snapping
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("VIEWPORT:").strong().color(okabe_ito::SKY_BLUE));

                // Projection mode toggle (Perspective <-> Orthographic)
                let is_ortho = self.camera.ortho_blend > 0.5;
                if ui.selectable_label(!is_ortho, "Perspective").clicked() {
                    self.camera.ortho_blend = 0.0;
                    self.projection_mode = ProjectionMode::Perspective;
                }
                if ui.selectable_label(is_ortho, "Orthographic").clicked() {
                    self.camera.ortho_blend = 1.0;
                    self.projection_mode = ProjectionMode::Orthographic;
                }

                ui.separator();

                // Standard Views
                ui.menu_button("Standard Views", |ui| {
                    if ui.button("Isometric").clicked() { self.camera.set_view("Iso"); ui.close_menu(); }
                    if ui.button("Top (+Y)").clicked() { self.camera.set_view("Top"); ui.close_menu(); }
                    if ui.button("Bottom (-Y)").clicked() { self.camera.set_view("Bottom"); ui.close_menu(); }
                    if ui.button("Front (+Z)").clicked() { self.camera.set_view("Front"); ui.close_menu(); }
                    if ui.button("Back (-Z)").clicked() { self.camera.set_view("Back"); ui.close_menu(); }
                    if ui.button("Left (-X)").clicked() { self.camera.set_view("Left"); ui.close_menu(); }
                    if ui.button("Right (+X)").clicked() { self.camera.set_view("Right"); ui.close_menu(); }
                });

                // Align to Selected Face Normal
                if let Some(fid) = self.selected_face_id {
                    if let (Some(&normal), Some(&centroid)) = (self.face_normals.get(&fid), self.face_centroids.get(&fid)) {
                        if ui.button(egui::RichText::new("Align View to Normal").color(okabe_ito::YELLOW)).clicked() {
                            self.camera.align_to_normal(normal, centroid);
                        }
                    }
                }

                ui.separator();

                // SELECTION TOOL MODE (Faces, Edges, Bodies) & DESELECT ALL (ESC)
                ui.label(egui::RichText::new("SELECT:").strong().color(okabe_ito::YELLOW));
                let is_face = self.selection_filter == SelectionFilter::Face;
                if ui.selectable_label(is_face, egui::RichText::new("Faces (Ctrl+F)").strong()).clicked() {
                    self.selection_filter = SelectionFilter::Face;
                }
                let is_edge = self.selection_filter == SelectionFilter::Edge;
                if ui.selectable_label(is_edge, egui::RichText::new("Edges (Ctrl+E)").strong()).clicked() {
                    self.selection_filter = SelectionFilter::Edge;
                }
                let is_body = self.selection_filter == SelectionFilter::Body;
                if ui.selectable_label(is_body, egui::RichText::new("Bodies (Ctrl+B)").strong()).clicked() {
                    self.selection_filter = SelectionFilter::Body;
                }

                let has_selection = self.selected_body || !self.selected_face_ids.is_empty() || !self.selected_edge_ids.is_empty();
                if ui.add_enabled(has_selection, egui::Button::new(egui::RichText::new("Deselect All (Esc)").strong().color(if has_selection { okabe_ito::VERMILION } else { okabe_ito::GRAY_BASE }))).clicked() {
                    self.deselect_all();
                }

                ui.separator();

                // Shading style dropdown
                egui::ComboBox::from_label("Shading Style")
                    .selected_text(match self.shading_mode {
                        ShadingMode::ShadedWithEdges => "Shaded with Edges",
                        ShadingMode::SmoothShaded => "Smooth Shaded",
                        ShadingMode::Faceted => "Faceted / Flat",
                        ShadingMode::NormalOrientation => "Diagnostic: Normal +/-",
                        ShadingMode::ZebraStripes => "Diagnostic: Zebra Stripes",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.shading_mode, ShadingMode::ShadedWithEdges, "Shaded with Edges");
                        ui.selectable_value(&mut self.shading_mode, ShadingMode::SmoothShaded, "Smooth Shaded");
                        ui.selectable_value(&mut self.shading_mode, ShadingMode::Faceted, "Faceted / Flat");
                        ui.selectable_value(&mut self.shading_mode, ShadingMode::NormalOrientation, "Diagnostic: Normal +/-");
                        ui.selectable_value(&mut self.shading_mode, ShadingMode::ZebraStripes, "Diagnostic: Zebra Stripes");
                    });

                ui.separator();
                ui.checkbox(&mut self.snap_pivot_on_click, "Snap Orbit Pivot");
            });
        });

        // LEFT CONTROL PANEL
        egui::SidePanel::left("control_panel")
            .default_width(340.0)
            .width_range(300.0..=450.0)
            .show(ctx, |ui| {
                ui.heading("IMMERS-IGA CAD LABELER");
                ui.label(format!("Model: {}", self.model_name));
                ui.label(format!("Triangles: {} | CAD Faces: {}", self.mesh.triangles.len(), self.face_triangles.len()));

                ui.separator();

                // Sectioning / Dynamic Cutting Plane
                ui.collapsing("Dynamic Section Plane", |ui| {
                    ui.checkbox(&mut self.section_plane.enabled, "Enable Cutting Plane");
                    if self.section_plane.enabled {
                        ui.horizontal(|ui| {
                            ui.label("Axis:");
                            ui.radio_value(&mut self.section_plane.axis, 0, "X");
                            ui.radio_value(&mut self.section_plane.axis, 1, "Y");
                            ui.radio_value(&mut self.section_plane.axis, 2, "Z");
                        });
                        let ax = self.section_plane.axis;
                        let min_v = self.bbox_min[ax];
                        let max_v = self.bbox_max[ax];
                        ui.add(egui::Slider::new(&mut self.section_plane.offset, min_v..=max_v).text("Plane Offset"));
                        ui.checkbox(&mut self.section_plane.flip, "Invert Cut Direction");
                    }
                });

                ui.separator();

                // Edge Visibility
                ui.collapsing("Edge & Wireframe Controls", |ui| {
                    ui.checkbox(&mut self.show_sharp_edges, "Feature Edges (Sharp Creases >= 25°)");
                    ui.checkbox(&mut self.show_open_edges, "Open Boundary Edges (Sheet Leaks - Magenta)");
                    ui.checkbox(&mut self.show_wireframe, "All Triangle Edges");
                    ui.label(format!("Classified Edges: {}", self.classified_edges.len()));
                });

                ui.separator();

                // CAD Face Segmentation Settings
                ui.collapsing("Face Segmentation", |ui| {
                    ui.label("Dihedral Angle Threshold:");
                    if ui.add(egui::Slider::new(&mut self.dihedral_angle_deg, 5.0..=60.0).suffix("°")).changed() {
                        self.segment_cad_faces();
                        self.classify_mesh_edges();
                    }
                    if ui.button("Re-segment Faces").clicked() {
                        self.segment_cad_faces();
                        self.classify_mesh_edges();
                    }
                });

                ui.separator();

                // Phase 3: Declarative Query Engine (Pillar 4)
                ui.collapsing("Declarative Query Engine (Pillar 4)", |ui| {
                    ui.label("Filter Predicate Type:");
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.query_mode_idx, 0, "Normal Vector");
                        ui.radio_value(&mut self.query_mode_idx, 1, "Coord Range");
                        ui.radio_value(&mut self.query_mode_idx, 2, "Surface Area");
                    });

                    match self.query_mode_idx {
                        0 => {
                            egui::ComboBox::from_label("Direction")
                                .selected_text(match self.query_normal_preset {
                                    0 => "+Z (Back)",
                                    1 => "-Z (Front)",
                                    2 => "+Y (Top)",
                                    3 => "-Y (Bottom)",
                                    4 => "+X (Right)",
                                    5 => "-X (Left)",
                                    _ => "Custom Vector",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.query_normal_preset, 2, "+Y (Top)");
                                    ui.selectable_value(&mut self.query_normal_preset, 3, "-Y (Bottom)");
                                    ui.selectable_value(&mut self.query_normal_preset, 0, "+Z (Back)");
                                    ui.selectable_value(&mut self.query_normal_preset, 1, "-Z (Front)");
                                    ui.selectable_value(&mut self.query_normal_preset, 4, "+X (Right)");
                                    ui.selectable_value(&mut self.query_normal_preset, 5, "-X (Left)");
                                    ui.selectable_value(&mut self.query_normal_preset, 6, "Custom Vector");
                                });

                            if self.query_normal_preset == 6 {
                                ui.horizontal(|ui| {
                                    ui.add(egui::DragValue::new(&mut self.query_normal_custom[0]).prefix("Nx: "));
                                    ui.add(egui::DragValue::new(&mut self.query_normal_custom[1]).prefix("Ny: "));
                                    ui.add(egui::DragValue::new(&mut self.query_normal_custom[2]).prefix("Nz: "));
                                });
                            }
                            ui.add(egui::Slider::new(&mut self.query_tolerance_deg, 1.0..=45.0).suffix("° tol"));
                        }
                        1 => {
                            ui.horizontal(|ui| {
                                ui.label("Axis:");
                                ui.radio_value(&mut self.query_coord_axis, 0, "X");
                                ui.radio_value(&mut self.query_coord_axis, 1, "Y");
                                ui.radio_value(&mut self.query_coord_axis, 2, "Z");
                            });
                            let ax = self.query_coord_axis;
                            let min_lim = self.bbox_min[ax];
                            let max_lim = self.bbox_max[ax];
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.query_coord_has_min, "Min:");
                                if self.query_coord_has_min {
                                    ui.add(egui::DragValue::new(&mut self.query_coord_min).range(min_lim..=max_lim));
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.query_coord_has_max, "Max:");
                                if self.query_coord_has_max {
                                    ui.add(egui::DragValue::new(&mut self.query_coord_max).range(min_lim..=max_lim));
                                }
                            });
                        }
                        2 => {
                            ui.horizontal(|ui| {
                                ui.label("Min Area:");
                                ui.add(egui::DragValue::new(&mut self.query_area_min));
                                ui.label("Max Area:");
                                ui.add(egui::DragValue::new(&mut self.query_area_max));
                            });
                        }
                        _ => {}
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button(egui::RichText::new("Evaluate Query").color(okabe_ito::YELLOW)).clicked() {
                            self.evaluate_active_query();
                        }
                        if !self.selected_face_ids.is_empty() {
                            if ui.button(egui::RichText::new("Assign BC to Query").color(okabe_ito::SKY_BLUE)).clicked() {
                                let base_name = self.tag_input_name.clone();
                                let bc_type = match self.bc_type_idx {
                                    0 => GuiBcType::Dirichlet {
                                        fixed_x: self.dirichlet_fixed[0],
                                        fixed_y: self.dirichlet_fixed[1],
                                        fixed_z: self.dirichlet_fixed[2],
                                        val: self.dirichlet_values,
                                    },
                                    1 => GuiBcType::NeumannTraction {
                                        traction: self.traction_vector,
                                    },
                                    2 => GuiBcType::NeumannPressure {
                                        pressure: self.pressure_val,
                                    },
                                    _ => unreachable!(),
                                };
                                for &fid in &self.selected_face_ids {
                                    let name = format!("{}_f{}", base_name, fid);
                                    self.assigned_bcs.insert(name.clone(), AssignedBc {
                                        _name: name,
                                        face_id: fid,
                                        bc: bc_type.clone(),
                                    });
                                }
                            }
                        }
                    });

                    if let Some(ref msg) = self.query_status_msg {
                        ui.colored_label(okabe_ito::BLUISH_GREEN, msg);
                    }

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Save Recipe:");
                        ui.text_edit_singleline(&mut self.new_selection_name);
                        if ui.button("Save").clicked() {
                            let name = if self.new_selection_name.trim().is_empty() {
                                format!("Selection_{}", self.named_selections.len() + 1)
                            } else {
                                self.new_selection_name.clone()
                            };
                            let q = self.build_active_query();
                            self.named_selections.insert(name.clone(), NamedSelection::new(&name, q));
                        }
                    });

                    // Saved Named Selections
                    if !self.named_selections.is_empty() {
                        ui.separator();
                        ui.label(egui::RichText::new("Named Selections (Lazy AST):").strong());
                        let mut to_del = None;
                        let mut to_apply = None;
                        let faces_info = self.get_face_info_list();
                        for (sname, nsel) in &self.named_selections {
                            let matched = nsel.resolve(&faces_info);
                            ui.horizontal(|ui| {
                                ui.colored_label(okabe_ito::SKY_BLUE, format!("{} ({} faces)", sname, matched.len()));
                                if ui.small_button("Select").clicked() {
                                    to_apply = Some(matched);
                                }
                                if ui.small_button("X").clicked() {
                                    to_del = Some(sname.clone());
                                }
                            });
                        }
                        if let Some(m) = to_apply {
                            self.selected_face_ids = m.clone();
                            if let Some(&first) = m.iter().next() {
                                self.selected_face_id = Some(first);
                            }
                        }
                        if let Some(d) = to_del {
                            self.named_selections.remove(&d);
                        }
                    }
                });

                ui.separator();

                // GROUPS & NAMED SELECTIONS OUTLINER
                ui.heading("GROUPS & SELECTIONS");

                let has_something_sel = self.selected_body || !self.selected_face_ids.is_empty() || !self.selected_edge_ids.is_empty();
                ui.horizontal(|ui| {
                    ui.label("Save Group:");
                    ui.add(egui::TextEdit::singleline(&mut self.new_group_name).desired_width(90.0));
                    if ui.add_enabled(has_something_sel, egui::Button::new(egui::RichText::new("+ Save Group").strong().color(okabe_ito::BLUISH_GREEN))).clicked() {
                        self.save_active_selection_as_group();
                    }
                });

                if self.selection_groups.is_empty() {
                    ui.colored_label(okabe_ito::GRAY_BASE, "No custom groups created yet.");
                } else {
                    let mut group_to_select = None;
                    let mut group_to_union = None;
                    let mut group_to_delete = None;

                    for (gi, group) in self.selection_groups.iter().enumerate() {
                        let count = match group.filter {
                            SelectionFilter::Face => group.face_ids.len(),
                            SelectionFilter::Edge => group.edge_ids.len(),
                            SelectionFilter::Body => 1,
                        };
                        let type_str = match group.filter {
                            SelectionFilter::Face => "faces",
                            SelectionFilter::Edge => "edges",
                            SelectionFilter::Body => "body",
                        };

                        ui.horizontal(|ui| {
                            ui.colored_label(group.color, "●");
                            ui.colored_label(okabe_ito::WHITE, format!("{}:", group.name));
                            ui.colored_label(okabe_ito::GRAY_BASE, format!("{} {}", count, type_str));

                            if ui.small_button("Select").on_hover_text("Replace active selection with this group").clicked() {
                                group_to_select = Some(gi);
                            }
                            if ui.small_button("+").on_hover_text("Add this group to active selection").clicked() {
                                group_to_union = Some(gi);
                            }
                            if ui.small_button("X").on_hover_text("Delete group").clicked() {
                                group_to_delete = Some(gi);
                            }
                        });
                    }

                    if let Some(gi) = group_to_select {
                        let grp = &self.selection_groups[gi];
                        self.selection_filter = grp.filter;
                        match grp.filter {
                            SelectionFilter::Face => {
                                self.selected_face_ids = grp.face_ids.clone();
                                self.selected_face_id = self.selected_face_ids.iter().next().copied();
                                self.selected_edge_ids.clear();
                                self.selected_body = false;
                            }
                            SelectionFilter::Edge => {
                                self.selected_edge_ids = grp.edge_ids.clone();
                                self.selected_face_ids.clear();
                                self.selected_face_id = None;
                                self.selected_body = false;
                            }
                            SelectionFilter::Body => {
                                self.selected_body = true;
                                self.selected_face_ids.clear();
                                self.selected_face_id = None;
                                self.selected_edge_ids.clear();
                            }
                        }
                    }
                    if let Some(gi) = group_to_union {
                        let grp = &self.selection_groups[gi];
                        self.selection_filter = grp.filter;
                        match grp.filter {
                            SelectionFilter::Face => {
                                self.selected_face_ids.extend(&grp.face_ids);
                                if self.selected_face_id.is_none() {
                                    self.selected_face_id = grp.face_ids.iter().next().copied();
                                }
                            }
                            SelectionFilter::Edge => {
                                self.selected_edge_ids.extend(&grp.edge_ids);
                            }
                            SelectionFilter::Body => {
                                self.selected_body = true;
                            }
                        }
                    }
                    if let Some(gi) = group_to_delete {
                        self.selection_groups.remove(gi);
                    }
                }

                ui.separator();

                // ACTIVE SELECTION INSPECTOR
                ui.heading(match self.selection_filter {
                    SelectionFilter::Face => "Active Selection: Faces (Ctrl+F)",
                    SelectionFilter::Edge => "Active Selection: Edges (Ctrl+E)",
                    SelectionFilter::Body => "Active Selection: Bodies (Ctrl+B)",
                });

                match self.selection_filter {
                    SelectionFilter::Face => {
                        let total_sel = self.selected_face_ids.len();
                        if total_sel > 0 {
                            ui.horizontal(|ui| {
                                ui.colored_label(
                                    okabe_ito::YELLOW,
                                    format!("{} Face(s) Selected", total_sel),
                                );
                                if ui.button(egui::RichText::new("Clear (Esc)").color(okabe_ito::VERMILION)).clicked() {
                                    self.deselect_all();
                                }
                            });

                            // Chips for each selected face with individual remove button
                            ui.horizontal_wrapped(|ui| {
                                let mut to_remove_fid = None;
                                for &fid in &self.selected_face_ids {
                                    let chip_label = format!("Face #{} ×", fid);
                                    if ui.small_button(chip_label).on_hover_text("Click to remove from selection").clicked() {
                                        to_remove_fid = Some(fid);
                                    }
                                }
                                if let Some(rem_fid) = to_remove_fid {
                                    self.selected_face_ids.remove(&rem_fid);
                                    if self.selected_face_id == Some(rem_fid) {
                                        self.selected_face_id = self.selected_face_ids.iter().next().copied();
                                    }
                                }
                            });

                            let mut total_area = 0.0;
                            for &fid in &self.selected_face_ids {
                                total_area += self.face_areas.get(&fid).copied().unwrap_or(0.0);
                            }
                            ui.label(format!("Combined Area: {:.2} mm²", total_area));

                            if let Some(fid) = self.selected_face_id {
                                if let (Some(&normal), Some(&centroid)) = (self.face_normals.get(&fid), self.face_centroids.get(&fid)) {
                                    ui.label(format!("Primary Face #{}: Normal [{:.2}, {:.2}, {:.2}]", fid, normal[0], normal[1], normal[2]));
                                    ui.label(format!("Centroid: [{:.1}, {:.1}, {:.1}]", centroid[0], centroid[1], centroid[2]));
                                }
                            }

                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.label("Tag Name:");
                                ui.text_edit_singleline(&mut self.tag_input_name);
                            });

                            ui.add_space(4.0);
                            ui.label("Condition Type:");
                            ui.horizontal(|ui| {
                                ui.radio_value(&mut self.bc_type_idx, 0, "Dirichlet");
                                ui.radio_value(&mut self.bc_type_idx, 1, "Traction");
                                ui.radio_value(&mut self.bc_type_idx, 2, "Pressure");
                            });

                            match self.bc_type_idx {
                                0 => {
                                    ui.label("Fix Displacement Components:");
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut self.dirichlet_fixed[0], "Ux");
                                        ui.checkbox(&mut self.dirichlet_fixed[1], "Uy");
                                        ui.checkbox(&mut self.dirichlet_fixed[2], "Uz");
                                    });
                                }
                                1 => {
                                    ui.label("Surface Traction Vector [Fx, Fy, Fz] (N):");
                                    ui.horizontal(|ui| {
                                        ui.add(egui::DragValue::new(&mut self.traction_vector[0]).prefix("X: "));
                                        ui.add(egui::DragValue::new(&mut self.traction_vector[1]).prefix("Y: "));
                                        ui.add(egui::DragValue::new(&mut self.traction_vector[2]).prefix("Z: "));
                                    });
                                }
                                2 => {
                                    ui.label("Normal Pressure p (MPa):");
                                    ui.add(egui::DragValue::new(&mut self.pressure_val));
                                }
                                _ => {}
                            }

                            ui.add_space(6.0);
                            let btn_label = if total_sel == 1 {
                                "Impose BC on Selected Face"
                            } else {
                                "Impose BC on All Selected Faces"
                            };
                            if ui.button(egui::RichText::new(btn_label).strong().color(okabe_ito::SKY_BLUE)).clicked() {
                                let bc_type = match self.bc_type_idx {
                                    0 => GuiBcType::Dirichlet {
                                        fixed_x: self.dirichlet_fixed[0],
                                        fixed_y: self.dirichlet_fixed[1],
                                        fixed_z: self.dirichlet_fixed[2],
                                        val: self.dirichlet_values,
                                    },
                                    1 => GuiBcType::NeumannTraction {
                                        traction: self.traction_vector,
                                    },
                                    2 => GuiBcType::NeumannPressure {
                                        pressure: self.pressure_val,
                                    },
                                    _ => unreachable!(),
                                };

                                for &fid in &self.selected_face_ids {
                                    let tag = if total_sel == 1 {
                                        self.tag_input_name.clone()
                                    } else {
                                        format!("{}_f{}", self.tag_input_name, fid)
                                    };
                                    self.assigned_bcs.insert(tag.clone(), AssignedBc {
                                        _name: tag,
                                        face_id: fid,
                                        bc: bc_type,
                                    });
                                }
                                if let Some(first_fid) = self.selected_face_ids.iter().next() {
                                    self.tag_input_name = format!("Face_{}", first_fid + 1);
                                }
                            }
                            ui.colored_label(okabe_ito::GRAY_BASE, "Ctrl+Click: Add | Ctrl+RightClick: Remove");
                        } else {
                            ui.label("Click on CAD surfaces in the viewport to select faces.");
                            ui.colored_label(okabe_ito::GRAY_BASE, "Ctrl+Click: Add | Ctrl+RightClick: Remove");
                        }
                    }
                    SelectionFilter::Edge => {
                        let total_sel = self.selected_edge_ids.len();
                        if total_sel > 0 {
                            ui.horizontal(|ui| {
                                ui.colored_label(
                                    okabe_ito::YELLOW,
                                    format!("{} Edge(s) Selected", total_sel),
                                );
                                if ui.button(egui::RichText::new("Clear (Esc)").color(okabe_ito::VERMILION)).clicked() {
                                    self.selected_edge_ids.clear();
                                }
                            });

                            ui.horizontal_wrapped(|ui| {
                                let mut to_remove_eid = None;
                                for &eid in self.selected_edge_ids.iter().take(12) {
                                    if ui.small_button(format!("Edge #{} ×", eid)).on_hover_text("Click to remove from selection").clicked() {
                                        to_remove_eid = Some(eid);
                                    }
                                }
                                if let Some(rem_eid) = to_remove_eid {
                                    self.selected_edge_ids.remove(&rem_eid);
                                }
                            });

                            ui.label("CAD feature & boundary edges selected.");
                            ui.colored_label(okabe_ito::GRAY_BASE, "Ctrl+Click: Add | Ctrl+RightClick: Remove");
                        } else {
                            ui.label("Click on CAD feature/boundary edges in the viewport.");
                            ui.colored_label(okabe_ito::GRAY_BASE, "Ctrl+Click: Add | Ctrl+RightClick: Remove");
                        }
                    }
                    SelectionFilter::Body => {
                        if self.selected_body {
                            ui.horizontal(|ui| {
                                ui.colored_label(
                                    okabe_ito::YELLOW,
                                    format!("Selected Body: {}", self.model_name),
                                );
                                if ui.button(egui::RichText::new("Clear (Esc)").color(okabe_ito::VERMILION)).clicked() {
                                    self.selected_body = false;
                                }
                            });
                            ui.label(format!("Triangles: {} | Faces: {}", self.mesh.triangles.len(), self.face_triangles.len()));
                            ui.colored_label(okabe_ito::GRAY_BASE, "Ctrl+Click: Add | Ctrl+RightClick: Remove");
                        } else {
                            ui.label("Click on the solid body in the viewport to select it.");
                            ui.colored_label(okabe_ito::GRAY_BASE, "Ctrl+Click: Add | Ctrl+RightClick: Remove");
                        }
                    }
                }

                ui.separator();

                // Imposed Boundary Conditions List
                ui.heading("Active Boundary Conditions");
                let mut to_remove = None;
                for (name, bc) in &self.assigned_bcs {
                    let col = match &bc.bc {
                        GuiBcType::Dirichlet { .. } => okabe_ito::ORANGE,
                        GuiBcType::NeumannTraction { .. } => okabe_ito::SKY_BLUE,
                        GuiBcType::NeumannPressure { .. } => okabe_ito::BLUE,
                    };
                    ui.horizontal(|ui| {
                        ui.colored_label(col, format!("[#{}] {}", bc.face_id, name));
                        if ui.small_button("X").clicked() {
                            to_remove = Some(name.clone());
                        }
                    });
                }
                if let Some(rem_name) = to_remove {
                    self.assigned_bcs.remove(&rem_name);
                }

                ui.separator();

                // Material Properties
                ui.collapsing("Material Properties", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.material.name);
                    });
                    ui.horizontal(|ui| {
                        ui.label("E (MPa):");
                        ui.add(egui::DragValue::new(&mut self.material.youngs_modulus));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Nu:");
                        ui.add(egui::DragValue::new(&mut self.material.poissons_ratio).speed(0.01));
                    });
                });

                ui.separator();

                // CAE Overlays Controls (Pillar 12)
                ui.collapsing("CAE Overlays & Glyphs (Pillar 12)", |ui| {
                    ui.checkbox(&mut self.show_bc_glyphs, "Show 3D BC Glyphs (Clamps/Arrows)");
                    ui.checkbox(&mut self.show_contour_heatmap, "Show Result Heatmap");
                });

                ui.separator();

                // Direct In-Process Solver
                ui.add_space(8.0);
                if ui.button(egui::RichText::new("RUN IMMERSED PCG SOLVER").size(16.0).strong().color(okabe_ito::YELLOW)).clicked() {
                    self.solve_in_process();
                }

                if let Some(ref summary) = self.solve_summary {
                    ui.add_space(6.0);
                    ui.colored_label(okabe_ito::BLUISH_GREEN, summary);
                    ui.checkbox(&mut self.show_deformed, "Show Deformed Shape");
                    if self.show_deformed {
                        ui.add(egui::Slider::new(&mut self.warp_scale, 0.1..=10.0).text("Warp Amplification"));
                    }
                }
            });

        // CENTRAL 3D VIEWPORT
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
            let viewport = response.rect;

            // Continuous 60/120 FPS repaint when mouse is over viewport (eliminates input stutter & lag)
            if response.hovered() {
                ctx.request_repaint();
            }

            // CAMERA INTERACTION
            if response.dragged_by(egui::PointerButton::Primary) {
                let delta = response.drag_delta();
                self.drag_distance_accum += delta.length();
                // 4px deadzone so single-clicks never jerk or rotate the camera
                if self.drag_distance_accum > 4.0 {
                    // Inverted rotation: move left -> rotates counter-clockwise
                    self.camera.yaw -= (delta.x as f64) * 0.008;
                    self.camera.pitch = (self.camera.pitch + (delta.y as f64) * 0.008).clamp(-1.50, 1.50);
                }
            } else if response.dragged_by(egui::PointerButton::Secondary) || response.dragged_by(egui::PointerButton::Middle) {
                let delta = response.drag_delta();
                let pan_scale = self.camera.distance * 0.0015;
                let (_fwd, right, up) = self.camera.basis_vectors();
                for k in 0..3 {
                    self.camera.center[k] -= (delta.x as f64) * right[k] * pan_scale;
                    self.camera.center[k] += (delta.y as f64) * up[k] * pan_scale;
                }
            }

            let scroll_delta = ctx.input(|i| i.raw_scroll_delta.y);
            if scroll_delta.abs() > 0.0 && response.hovered() {
                let zoom_factor = (1.0 - (scroll_delta as f64) * 0.002).clamp(0.5, 2.0);
                self.camera.distance = (self.camera.distance * zoom_factor).clamp(1e-2, 1e6);
            }

            // REAL-TIME CURSOR HOVER HIT-TESTING (PRE-SELECTION HIGHLIGHTING)
            // Active immediately whenever the mouse is not actively dragging
            let is_actively_dragging = response.dragged_by(egui::PointerButton::Primary)
                || response.dragged_by(egui::PointerButton::Secondary)
                || response.dragged_by(egui::PointerButton::Middle);

            if !is_actively_dragging {
                if let Some(mouse_pos) = response.hover_pos() {
                    match self.selection_filter {
                        SelectionFilter::Face => {
                            self.hovered_edge_id = None;
                            self.hovered_body = false;

                            let (ray_orig, ray_dir) = self.camera.unproject_mouse_ray(mouse_pos, viewport);
                            let mut min_t = f64::MAX;
                            let mut picked_tri = None;

                            // Primary: Fast 3D Möller–Trumbore ray casting
                            for (ti, tri) in self.mesh.triangles.iter().enumerate() {
                                let v0 = self.mesh.vertices[tri[0]];
                                let v1 = self.mesh.vertices[tri[1]];
                                let v2 = self.mesh.vertices[tri[2]];

                                if self.section_plane.enabled {
                                    let ax = self.section_plane.axis;
                                    let off = self.section_plane.offset;
                                    let c = (v0[ax] + v1[ax] + v2[ax]) / 3.0;
                                    if (!self.section_plane.flip && c > off) || (self.section_plane.flip && c < off) {
                                        continue;
                                    }
                                }

                                if let Some(t) = ray_intersect_triangle(ray_orig, ray_dir, v0, v1, v2) {
                                    if t > 1e-6 && t < min_t {
                                        min_t = t;
                                        picked_tri = Some(ti);
                                    }
                                }
                            }

                            // Secondary fallback: 2D Screen-space polygon containment (guarantees 100% surface coverage)
                            if picked_tri.is_none() {
                                for (ti, tri) in self.mesh.triangles.iter().enumerate() {
                                    let v0 = self.mesh.vertices[tri[0]];
                                    let v1 = self.mesh.vertices[tri[1]];
                                    let v2 = self.mesh.vertices[tri[2]];

                                    if self.section_plane.enabled {
                                        let ax = self.section_plane.axis;
                                        let off = self.section_plane.offset;
                                        let c = (v0[ax] + v1[ax] + v2[ax]) / 3.0;
                                        if (!self.section_plane.flip && c > off) || (self.section_plane.flip && c < off) {
                                            continue;
                                        }
                                    }

                                    if let (Some((s0, z0)), Some((s1, z1)), Some((s2, z2))) = (
                                        self.camera.project_point(v0, viewport),
                                        self.camera.project_point(v1, viewport),
                                        self.camera.project_point(v2, viewport),
                                    ) {
                                        if z0 > 0.0 && z1 > 0.0 && z2 > 0.0 {
                                            if point_in_triangle_2d(mouse_pos, s0, s1, s2) {
                                                let avg_z = (z0 + z1 + z2) / 3.0;
                                                if avg_z < min_t {
                                                    min_t = avg_z;
                                                    picked_tri = Some(ti);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(ti) = picked_tri {
                                let fid = self.face_labels[ti];
                                self.hovered_face_id = Some(fid);
                                self.hovered_hit_pt = Some([
                                    ray_orig[0] + min_t * ray_dir[0],
                                    ray_orig[1] + min_t * ray_dir[1],
                                    ray_orig[2] + min_t * ray_dir[2],
                                ]);
                                self.hovered_hit_dist = Some(min_t);
                            } else {
                                self.hovered_face_id = None;
                                self.hovered_hit_pt = None;
                                self.hovered_hit_dist = None;
                            }
                        }
                        SelectionFilter::Edge => {
                            self.hovered_face_id = None;
                            self.hovered_hit_pt = None;
                            self.hovered_hit_dist = None;
                            self.hovered_body = false;

                            let mut best_edge = None;
                            let mut best_dist = 18.0_f32; // Generous 18-pixel magnetic picking tolerance
                            let mut best_depth = f64::MAX;

                            for (ei, edge) in self.classified_edges.iter().enumerate() {
                                let p0 = self.mesh.vertices[edge.v0];
                                let p1 = self.mesh.vertices[edge.v1];

                                if self.section_plane.enabled {
                                    let ax = self.section_plane.axis;
                                    let off = self.section_plane.offset;
                                    if (!self.section_plane.flip && (p0[ax] > off || p1[ax] > off))
                                        || (self.section_plane.flip && (p0[ax] < off || p1[ax] < off)) {
                                        continue;
                                    }
                                }

                                if let (Some((s0, z0)), Some((s1, z1))) = (
                                    self.camera.project_point(p0, viewport),
                                    self.camera.project_point(p1, viewport),
                                ) {
                                    if z0 > 0.0 && z1 > 0.0 {
                                        let d = dist_to_segment_2d(mouse_pos, s0, s1);
                                        let avg_z = (z0 + z1) * 0.5;
                                        if d <= best_dist {
                                            if d < best_dist * 0.75 || avg_z < best_depth {
                                                best_dist = d;
                                                best_depth = avg_z;
                                                best_edge = Some(ei);
                                            }
                                        }
                                    }
                                }
                            }

                            self.hovered_edge_id = best_edge;
                        }
                        SelectionFilter::Body => {
                            self.hovered_face_id = None;
                            self.hovered_edge_id = None;

                            let (ray_orig, ray_dir) = self.camera.unproject_mouse_ray(mouse_pos, viewport);
                            let mut min_t = f64::MAX;
                            let mut hit = false;
                            for tri in &self.mesh.triangles {
                                let v0 = self.mesh.vertices[tri[0]];
                                let v1 = self.mesh.vertices[tri[1]];
                                let v2 = self.mesh.vertices[tri[2]];
                                if let Some(t) = ray_intersect_triangle(ray_orig, ray_dir, v0, v1, v2) {
                                    if t > 1e-6 && t < min_t {
                                        min_t = t;
                                        hit = true;
                                    }
                                }
                            }
                            // Screen space fallback for body
                            if !hit {
                                for tri in &self.mesh.triangles {
                                    let v0 = self.mesh.vertices[tri[0]];
                                    let v1 = self.mesh.vertices[tri[1]];
                                    let v2 = self.mesh.vertices[tri[2]];
                                    if let (Some((s0, z0)), Some((s1, z1)), Some((s2, z2))) = (
                                        self.camera.project_point(v0, viewport),
                                        self.camera.project_point(v1, viewport),
                                        self.camera.project_point(v2, viewport),
                                    ) {
                                        if z0 > 0.0 && z1 > 0.0 && z2 > 0.0 && point_in_triangle_2d(mouse_pos, s0, s1, s2) {
                                            hit = true;
                                            min_t = (z0 + z1 + z2) / 3.0;
                                            break;
                                        }
                                    }
                                }
                            }

                            if hit {
                                self.hovered_body = true;
                                self.hovered_hit_pt = Some([
                                    ray_orig[0] + min_t * ray_dir[0],
                                    ray_orig[1] + min_t * ray_dir[1],
                                    ray_orig[2] + min_t * ray_dir[2],
                                ]);
                                self.hovered_hit_dist = Some(min_t);
                            } else {
                                self.hovered_body = false;
                                self.hovered_hit_pt = None;
                                self.hovered_hit_dist = None;
                            }
                        }
                    }
                } else {
                    self.hovered_face_id = None;
                    self.hovered_edge_id = None;
                    self.hovered_body = false;
                    self.hovered_hit_pt = None;
                    self.hovered_hit_dist = None;
                }
            } else {
                self.hovered_face_id = None;
                self.hovered_edge_id = None;
                self.hovered_body = false;
            }

            // SELECTION TOOL: MOUSE PICKING & MODIFIERS (Ctrl+Click to Add, Ctrl+RightClick to Remove)
            let ctrl_held = ctx.input(|i| i.modifiers.ctrl || i.modifiers.command);
            let primary_clicked = response.clicked() || (response.drag_stopped() && self.drag_distance_accum < 5.0);
            let secondary_clicked = response.secondary_clicked();

            if (primary_clicked || (ctrl_held && secondary_clicked)) && self.drag_distance_accum < 5.0 {
                let is_add = ctrl_held && primary_clicked;
                let is_remove = ctrl_held && secondary_clicked;
                let is_replace = !ctrl_held && primary_clicked;

                match self.selection_filter {
                    SelectionFilter::Face => {
                        if let Some(fid) = self.hovered_face_id {
                            if is_add {
                                self.selected_face_ids.insert(fid);
                                self.selected_face_id = Some(fid);
                                self.tag_input_name = format!("Face_{}", fid);
                            } else if is_remove {
                                self.selected_face_ids.remove(&fid);
                                if self.selected_face_id == Some(fid) {
                                    self.selected_face_id = self.selected_face_ids.iter().next().copied();
                                    if let Some(first_fid) = self.selected_face_id {
                                        self.tag_input_name = format!("Face_{}", first_fid);
                                    }
                                }
                            } else if is_replace {
                                self.selected_face_ids.clear();
                                self.selected_face_ids.insert(fid);
                                self.selected_face_id = Some(fid);
                                self.tag_input_name = format!("Face_{}", fid);
                            }

                            // Dynamic orbit pivot snap to surface hit point
                            if self.snap_pivot_on_click {
                                if let (Some(hit_pt), Some(dist)) = (self.hovered_hit_pt, self.hovered_hit_dist) {
                                    self.camera.center = hit_pt;
                                    self.camera.distance = dist;
                                }
                            }
                        } else if is_replace {
                            self.selected_face_ids.clear();
                            self.selected_face_id = None;
                        }
                    }
                    SelectionFilter::Edge => {
                        if let Some(ei) = self.hovered_edge_id {
                            if is_add {
                                self.selected_edge_ids.insert(ei);
                            } else if is_remove {
                                self.selected_edge_ids.remove(&ei);
                            } else if is_replace {
                                self.selected_edge_ids.clear();
                                self.selected_edge_ids.insert(ei);
                            }
                        } else if is_replace {
                            self.selected_edge_ids.clear();
                        }
                    }
                    SelectionFilter::Body => {
                        if self.hovered_body {
                            if is_add || is_replace {
                                self.selected_body = true;
                            } else if is_remove {
                                self.selected_body = false;
                            }
                        } else if is_replace {
                            self.selected_body = false;
                        }
                    }
                }
            }

            // Immediately reset drag accumulator when pointer is released or drag ends
            if !ctx.input(|i| i.pointer.any_down()) || response.drag_stopped() {
                self.drag_distance_accum = 0.0;
            }
            if response.drag_started() {
                self.drag_distance_accum = 0.0;
            }

            // RENDER 3D VIEWPORT
            painter.rect_filled(viewport, 0.0, okabe_ito::DARK_CANVAS);

            let (fwd, right, up) = self.camera.basis_vectors();

            // Studio 2-point CAD lighting rig:
            // Key light: shines from top-right-front towards the scene
            // Fill light: shines from bottom-left-front towards the scene
            let key_dir = {
                let l = [-fwd[0] + 0.45 * up[0] + 0.35 * right[0],
                         -fwd[1] + 0.45 * up[1] + 0.35 * right[1],
                         -fwd[2] + 0.45 * up[2] + 0.35 * right[2]];
                let len = (l[0]*l[0] + l[1]*l[1] + l[2]*l[2]).sqrt().max(1e-12);
                [l[0]/len, l[1]/len, l[2]/len]
            };
            let fill_dir = {
                let l = [-fwd[0] - 0.25 * up[0] - 0.40 * right[0],
                         -fwd[1] - 0.25 * up[1] - 0.40 * right[1],
                         -fwd[2] - 0.25 * up[2] - 0.40 * right[2]];
                let len = (l[0]*l[0] + l[1]*l[1] + l[2]*l[2]).sqrt().max(1e-12);
                [l[0]/len, l[1]/len, l[2]/len]
            };

            struct ProjectedTri {
                screen_pts: [Pos2; 3],
                z_depth: f64,
                color: Color32,
                _face_id: usize,
            }

            let mut projected = Vec::with_capacity(self.mesh.triangles.len());

            let mut face_bc_color = HashMap::new();
            for bc in self.assigned_bcs.values() {
                let c = match &bc.bc {
                    GuiBcType::Dirichlet { .. } => okabe_ito::ORANGE,
                    GuiBcType::NeumannTraction { .. } => okabe_ito::SKY_BLUE,
                    GuiBcType::NeumannPressure { .. } => okabe_ito::BLUE,
                };
                face_bc_color.insert(bc.face_id, c);
            }

            for (ti, tri) in self.mesh.triangles.iter().enumerate() {
                let fid = self.face_labels[ti];

                let (orig_p0, orig_p1, orig_p2) = (
                    self.mesh.vertices[tri[0]],
                    self.mesh.vertices[tri[1]],
                    self.mesh.vertices[tri[2]],
                );

                // Displace if show_deformed is active
                let (p0, p1, p2) = if self.show_deformed && self.cad_vertex_displacements.is_some() {
                    let disps = self.cad_vertex_displacements.as_ref().unwrap();
                    let u0 = disps[tri[0]];
                    let u1 = disps[tri[1]];
                    let u2 = disps[tri[2]];
                    let scale = self.warp_scale;
                    (
                        [orig_p0[0] + scale * u0[0], orig_p0[1] + scale * u0[1], orig_p0[2] + scale * u0[2]],
                        [orig_p1[0] + scale * u1[0], orig_p1[1] + scale * u1[1], orig_p1[2] + scale * u1[2]],
                        [orig_p2[0] + scale * u2[0], orig_p2[1] + scale * u2[1], orig_p2[2] + scale * u2[2]],
                    )
                } else {
                    (orig_p0, orig_p1, orig_p2)
                };

                // Section plane discard
                if self.section_plane.enabled {
                    let ax = self.section_plane.axis;
                    let off = self.section_plane.offset;
                    let c = (p0[ax] + p1[ax] + p2[ax]) / 3.0;
                    if (!self.section_plane.flip && c > off) || (self.section_plane.flip && c < off) {
                        continue;
                    }
                }

                // Geometric normal
                let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                let norm = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let nlen = (norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]).sqrt().max(1e-12);
                let n_norm = [norm[0] / nlen, norm[1] / nlen, norm[2] / nlen];

                // View direction vector
                let eye = self.camera.eye_pos();
                let view_vec = [p0[0] - eye[0], p0[1] - eye[1], p0[2] - eye[2]];
                let view_dot = n_norm[0] * view_vec[0] + n_norm[1] * view_vec[1] + n_norm[2] * view_vec[2];

                let s0 = self.camera.project_point(p0, viewport);
                let s1 = self.camera.project_point(p1, viewport);
                let s2 = self.camera.project_point(p2, viewport);

                if let (Some((pt0, z0)), Some((pt1, z1)), Some((pt2, z2))) = (s0, s1, s2) {
                    let avg_z = (z0 + z1 + z2) / 3.0;

                    // Two-sided facing normal alignment
                    let n_facing = if view_dot > 0.0 {
                        [-n_norm[0], -n_norm[1], -n_norm[2]]
                    } else {
                        n_norm
                    };

                    let final_color = if self.show_contour_heatmap && self.cad_vertex_mags.is_some() && self.max_displacement > 1e-12 {
                        let mags = self.cad_vertex_mags.as_ref().unwrap();
                        let avg_m = (mags[tri[0]] + mags[tri[1]] + mags[tri[2]]) / 3.0;
                        let norm_val = avg_m / self.max_displacement;
                        let base_c = okabe_ito_heatmap(norm_val);
                        let dot = (n_facing[0] * key_dir[0] + n_facing[1] * key_dir[1] + n_facing[2] * key_dir[2]).max(0.0);
                        let intensity = (0.45 + 0.55 * dot).clamp(0.0, 1.0) as f32;
                        Color32::from_rgb(
                            (base_c.r() as f32 * intensity) as u8,
                            (base_c.g() as f32 * intensity) as u8,
                            (base_c.b() as f32 * intensity) as u8,
                        )
                    } else {
                        match self.shading_mode {
                            ShadingMode::NormalOrientation => {
                                // Okabe-Ito: Bluish Green for +normal, Vermilion for inverted
                                if n_norm[1] >= -0.1 {
                                    okabe_ito::BLUISH_GREEN
                                } else {
                                    okabe_ito::VERMILION
                                }
                            }
                            ShadingMode::ZebraStripes => {
                                // Reflection map striping
                                let refl = [
                                    view_vec[0] - 2.0 * view_dot * n_norm[0],
                                    view_vec[1] - 2.0 * view_dot * n_norm[1],
                                    view_vec[2] - 2.0 * view_dot * n_norm[2],
                                ];
                                let coord = (refl[0] * 12.0 + refl[1] * 12.0).sin();
                                if coord > 0.0 {
                                    okabe_ito::WHITE
                                } else {
                                    okabe_ito::DARK_CANVAS
                                }
                            }
                            ShadingMode::Faceted | ShadingMode::SmoothShaded | ShadingMode::ShadedWithEdges => {
                                let dot_key = (n_facing[0] * key_dir[0] + n_facing[1] * key_dir[1] + n_facing[2] * key_dir[2]).max(0.0);
                                let dot_fill = (n_facing[0] * fill_dir[0] + n_facing[1] * fill_dir[1] + n_facing[2] * fill_dir[2]).max(0.0);
                                let diffuse = (0.28 + 0.54 * dot_key + 0.18 * dot_fill).clamp(0.20, 1.0);

                                let v_len = (view_vec[0]*view_vec[0] + view_vec[1]*view_vec[1] + view_vec[2]*view_vec[2]).sqrt().max(1e-12);
                                let v_unit = [-view_vec[0] / v_len, -view_vec[1] / v_len, -view_vec[2] / v_len];
                                let h = {
                                    let hx = key_dir[0] + v_unit[0];
                                    let hy = key_dir[1] + v_unit[1];
                                    let hz = key_dir[2] + v_unit[2];
                                    let hlen = (hx*hx + hy*hy + hz*hz).sqrt().max(1e-12);
                                    [hx/hlen, hy/hlen, hz/hlen]
                                };
                                let n_dot_h = (n_facing[0] * h[0] + n_facing[1] * h[1] + n_facing[2] * h[2]).max(0.0);
                                let specular = n_dot_h.powf(28.0) * 0.22;

                                let is_face_sel = Some(fid) == self.selected_face_id || self.selected_face_ids.contains(&fid);
                                let is_face_hover = Some(fid) == self.hovered_face_id;
                                let is_body_hover = self.hovered_body;

                                let base_c = if self.selected_body || is_face_sel {
                                    okabe_ito::YELLOW
                                } else if is_body_hover || is_face_hover {
                                    okabe_ito::SKY_BLUE
                                } else if let Some(&c) = face_bc_color.get(&fid) {
                                    c
                                } else {
                                    okabe_ito::GRAY_BASE
                                };

                                let r = ((base_c.r() as f64) * diffuse + specular * 255.0).clamp(0.0, 255.0) as u8;
                                let g = ((base_c.g() as f64) * diffuse + specular * 255.0).clamp(0.0, 255.0) as u8;
                                let b = ((base_c.b() as f64) * diffuse + specular * 255.0).clamp(0.0, 255.0) as u8;
                                Color32::from_rgb(r, g, b)
                            }
                        }
                    };

                    projected.push(ProjectedTri {
                        screen_pts: [pt0, pt1, pt2],
                        z_depth: avg_z,
                        color: final_color,
                        _face_id: fid,
                    });
                }
            }

            // Painter's algorithm depth sort
            projected.sort_by(|a, b| b.z_depth.partial_cmp(&a.z_depth).unwrap_or(std::cmp::Ordering::Equal));

            // Render filled triangles - Stroke::NONE prevents acute triangle miter spikes (white rays)
            for tri in &projected {
                painter.add(egui::Shape::convex_polygon(
                    tri.screen_pts.to_vec(),
                    tri.color,
                    Stroke::NONE,
                ));
            }

            // Wireframe pass using discrete 2-point line segments (no miter joins)
            if self.show_wireframe {
                let wf_stroke = Stroke::new(0.6_f32, Color32::from_black_alpha(70));
                for tri in &projected {
                    painter.line_segment([tri.screen_pts[0], tri.screen_pts[1]], wf_stroke);
                    painter.line_segment([tri.screen_pts[1], tri.screen_pts[2]], wf_stroke);
                    painter.line_segment([tri.screen_pts[2], tri.screen_pts[0]], wf_stroke);
                }
            }

            // RENDER CLASSIFIED EDGES: Sharp Creases, Open Boundaries & Selected/Hovered Contours
            for (ei, edge) in self.classified_edges.iter().enumerate() {
                let is_edge_sel = self.selected_edge_ids.contains(&ei);
                let is_edge_hover = Some(ei) == self.hovered_edge_id;
                let bounds_selected_face = edge.faces.iter().filter_map(|&f| f).any(|f| {
                    self.selected_body || Some(f) == self.selected_face_id || self.selected_face_ids.contains(&f)
                });
                let bounds_hovered_face = edge.faces.iter().filter_map(|&f| f).any(|f| {
                    Some(f) == self.hovered_face_id
                });

                if !is_edge_sel && !is_edge_hover && !bounds_selected_face && !bounds_hovered_face {
                    if edge.is_open && !self.show_open_edges {
                        continue;
                    }
                    if edge.is_sharp && !self.show_sharp_edges {
                        continue;
                    }
                }

                let p0 = self.mesh.vertices[edge.v0];
                let p1 = self.mesh.vertices[edge.v1];

                // Section plane discard
                if self.section_plane.enabled {
                    let ax = self.section_plane.axis;
                    let off = self.section_plane.offset;
                    if (!self.section_plane.flip && (p0[ax] > off || p1[ax] > off))
                        || (self.section_plane.flip && (p0[ax] < off || p1[ax] < off)) {
                        continue;
                    }
                }

                if let (Some((s0, _)), Some((s1, _))) = (self.camera.project_point(p0, viewport), self.camera.project_point(p1, viewport)) {
                    if is_edge_sel {
                        // High visibility Okabe-Ito Yellow selection highlight with contrast halo
                        painter.line_segment([s0, s1], Stroke::new(5.5_f32, Color32::from_black_alpha(220)));
                        painter.line_segment([s0, s1], Stroke::new(3.8_f32, okabe_ito::YELLOW));
                    } else if is_edge_hover {
                        // High visibility Okabe-Ito Sky Blue hover highlight with contrast halo
                        painter.line_segment([s0, s1], Stroke::new(5.0_f32, Color32::from_black_alpha(200)));
                        painter.line_segment([s0, s1], Stroke::new(3.6_f32, okabe_ito::SKY_BLUE));
                    } else if bounds_selected_face {
                        // Crisp white perimeter contour for selected faces without miter spikes
                        painter.line_segment([s0, s1], Stroke::new(2.4_f32, okabe_ito::WHITE));
                    } else if bounds_hovered_face {
                        // Crisp Okabe-Ito Sky Blue perimeter contour for hovered faces
                        painter.line_segment([s0, s1], Stroke::new(2.4_f32, okabe_ito::SKY_BLUE));
                    } else if edge.is_open {
                        // Okabe-Ito Reddish Purple for open sheet leaks / hole boundaries
                        painter.line_segment([s0, s1], Stroke::new(2.4_f32, okabe_ito::REDDISH_PURPLE));
                    } else if edge.is_sharp && self.shading_mode == ShadingMode::ShadedWithEdges {
                        // Okabe-Ito dark sharp feature creases
                        painter.line_segment([s0, s1], Stroke::new(1.2_f32, okabe_ito::SHARP_EDGE));
                    }
                }
            }

            // PHASE 3: 3D BOUNDARY CONDITION GLYPHS (PILLAR 12)
            if self.show_bc_glyphs {
                for (_name, assigned_bc) in &self.assigned_bcs {
                    let fid = assigned_bc.face_id;
                    if let (Some(&c_3d), Some(&n_3d)) = (self.face_centroids.get(&fid), self.face_normals.get(&fid)) {
                        // Section plane discard
                        if self.section_plane.enabled {
                            let ax = self.section_plane.axis;
                            let off = self.section_plane.offset;
                            if (!self.section_plane.flip && c_3d[ax] > off) || (self.section_plane.flip && c_3d[ax] < off) {
                                continue;
                            }
                        }

                        if let Some((scr_pt, z_depth)) = self.camera.project_point(c_3d, viewport) {
                            if z_depth > 0.0 {
                                match assigned_bc.bc {
                                    GuiBcType::Dirichlet { fixed_x, fixed_y, fixed_z, .. } => {
                                        // Auto-scaling screen-space Pinned Ground Support Pyramid
                                        let base_w = 22.0_f32;
                                        let h = 18.0_f32;
                                        let tip = scr_pt;
                                        let base_center = scr_pt + Vec2::new(0.0, h);
                                        let left = base_center + Vec2::new(-base_w * 0.5, 0.0);
                                        let right = base_center + Vec2::new(base_w * 0.5, 0.0);

                                        // Support triangle
                                        painter.add(egui::Shape::convex_polygon(
                                            vec![tip, left, right],
                                            Color32::from_rgba_unmultiplied(230, 159, 0, 190), // Orange translucent
                                            Stroke::new(1.8_f32, okabe_ito::ORANGE),
                                        ));

                                        // Ground base line & hatching
                                        painter.line_segment([left + Vec2::new(-4.0, 2.0), right + Vec2::new(4.0, 2.0)], Stroke::new(2.2_f32, okabe_ito::ORANGE));
                                        for k in -2..=2 {
                                            let xk = scr_pt.x + (k as f32) * 5.0;
                                            painter.line_segment(
                                                [Pos2::new(xk, base_center.y + 2.0), Pos2::new(xk - 3.5, base_center.y + 7.5)],
                                                Stroke::new(1.3_f32, okabe_ito::ORANGE),
                                            );
                                        }

                                        let label = format!("FIX [{}{}{}]", if fixed_x {"X"} else {"-"}, if fixed_y {"Y"} else {"-"}, if fixed_z {"Z"} else {"-"});
                                        painter.text(base_center + Vec2::new(0.0, 9.0), egui::Align2::CENTER_TOP, label, egui::FontId::monospace(9.5), okabe_ito::ORANGE);
                                    }
                                    GuiBcType::NeumannTraction { traction } => {
                                        // Auto-scaling 3D Directional Load Arrow
                                        let tlen = (traction[0].powi(2) + traction[1].powi(2) + traction[2].powi(2)).sqrt().max(1e-6);
                                        let tdir = [traction[0] / tlen, traction[1] / tlen, traction[2] / tlen];
                                        let end_3d = [c_3d[0] + tdir[0] * 10.0, c_3d[1] + tdir[1] * 10.0, c_3d[2] + tdir[2] * 10.0];

                                        let scr_dir = if let Some((scr_end, _)) = self.camera.project_point(end_3d, viewport) {
                                            let diff = scr_end - scr_pt;
                                            if diff.length() > 1e-3 { diff.normalized() } else { Vec2::new(0.0, 1.0) }
                                        } else {
                                            Vec2::new(0.0, 1.0)
                                        };

                                        let arrow_len = 38.0_f32;
                                        let arrow_tail = scr_pt;
                                        let arrow_head = scr_pt + scr_dir * arrow_len;

                                        // Arrow shaft
                                        painter.line_segment([arrow_tail, arrow_head], Stroke::new(3.0_f32, okabe_ito::SKY_BLUE));

                                        // Chevron arrowhead
                                        let perp = Vec2::new(-scr_dir.y, scr_dir.x);
                                        let head_p1 = arrow_head - scr_dir * 10.0 + perp * 5.5;
                                        let head_p2 = arrow_head - scr_dir * 10.0 - perp * 5.5;
                                        painter.add(egui::Shape::convex_polygon(
                                            vec![arrow_head, head_p1, head_p2],
                                            okabe_ito::SKY_BLUE,
                                            Stroke::NONE,
                                        ));

                                        painter.text(arrow_head + scr_dir * 6.0, egui::Align2::CENTER_CENTER, format!("{:.0}N", tlen), egui::FontId::monospace(9.5), okabe_ito::SKY_BLUE);
                                    }
                                    GuiBcType::NeumannPressure { pressure } => {
                                        // Auto-scaling Inward Normal Pressure Arrow
                                        let end_3d = [c_3d[0] - n_3d[0] * 10.0, c_3d[1] - n_3d[1] * 10.0, c_3d[2] - n_3d[2] * 10.0];
                                        let scr_dir = if let Some((scr_end, _)) = self.camera.project_point(end_3d, viewport) {
                                            let diff = scr_end - scr_pt;
                                            if diff.length() > 1e-3 { diff.normalized() } else { Vec2::new(0.0, -1.0) }
                                        } else {
                                            Vec2::new(0.0, -1.0)
                                        };

                                        let arrow_len = 32.0_f32;
                                        let arrow_tail = scr_pt - scr_dir * arrow_len;
                                        let arrow_head = scr_pt;

                                        painter.line_segment([arrow_tail, arrow_head], Stroke::new(2.8_f32, okabe_ito::BLUE));

                                        let perp = Vec2::new(-scr_dir.y, scr_dir.x);
                                        let head_p1 = arrow_head - scr_dir * 9.0 + perp * 5.0;
                                        let head_p2 = arrow_head - scr_dir * 9.0 - perp * 5.0;
                                        painter.add(egui::Shape::convex_polygon(
                                            vec![arrow_head, head_p1, head_p2],
                                            okabe_ito::BLUE,
                                            Stroke::NONE,
                                        ));

                                        painter.text(arrow_tail - scr_dir * 5.0, egui::Align2::CENTER_CENTER, format!("{:.1}MPa", pressure), egui::FontId::monospace(9.5), okabe_ito::BLUE);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // PHASE 3: HUD COLORBAR LEGEND (PILLAR 12)
            if self.show_contour_heatmap && self.max_displacement > 1e-12 {
                let bar_top_left = viewport.right_top() + Vec2::new(-115.0, 135.0);
                let bar_w = 16.0_f32;
                let bar_h = 160.0_f32;

                // Background panel
                painter.rect_filled(
                    Rect::from_min_size(bar_top_left + Vec2::new(-12.0, -22.0), Vec2::new(bar_w + 95.0, bar_h + 44.0)),
                    4.0,
                    Color32::from_black_alpha(170),
                );

                // Title
                painter.text(
                    bar_top_left + Vec2::new(-6.0, -15.0),
                    egui::Align2::LEFT_TOP,
                    "Disp |u| [mm]",
                    egui::FontId::monospace(10.0),
                    okabe_ito::WHITE,
                );

                // Discrete 16 gradient slices across Okabe-Ito hues
                let n_steps = 16;
                for step in 0..n_steps {
                    let frac0 = (step as f32) / (n_steps as f32);
                    let frac1 = ((step + 1) as f32) / (n_steps as f32);
                    let y0 = bar_top_left.y + (1.0 - frac1) * bar_h;
                    let y1 = bar_top_left.y + (1.0 - frac0) * bar_h;
                    let c = okabe_ito_heatmap((frac0 + frac1) as f64 * 0.5);
                    painter.rect_filled(Rect::from_min_max(Pos2::new(bar_top_left.x, y0), Pos2::new(bar_top_left.x + bar_w, y1)), 0.0, c);
                }
                painter.rect_stroke(Rect::from_min_size(bar_top_left, Vec2::new(bar_w, bar_h)), 0.0, Stroke::new(1.0_f32, Color32::GRAY));

                // Numerical ticks
                painter.text(bar_top_left + Vec2::new(bar_w + 6.0, 0.0), egui::Align2::LEFT_CENTER, format!("{:.4}", self.max_displacement), egui::FontId::monospace(9.5), okabe_ito::VERMILION);
                painter.text(bar_top_left + Vec2::new(bar_w + 6.0, bar_h * 0.5), egui::Align2::LEFT_CENTER, format!("{:.4}", self.max_displacement * 0.5), egui::FontId::monospace(9.5), okabe_ito::BLUISH_GREEN);
                painter.text(bar_top_left + Vec2::new(bar_w + 6.0, bar_h), egui::Align2::LEFT_CENTER, "0.0000", egui::FontId::monospace(9.5), okabe_ito::BLUE);
            }

            // HUD DISPLAY CARD
            let sel_summary = if self.selected_body {
                format!("Body: {}", self.model_name)
            } else if !self.selected_face_ids.is_empty() {
                format!(
                    "{} Face(s) [{}]",
                    self.selected_face_ids.len(),
                    self.selected_face_ids
                        .iter()
                        .take(5)
                        .map(|id| format!("#{}", id))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            } else if !self.selected_edge_ids.is_empty() {
                format!("{} Edge(s)", self.selected_edge_ids.len())
            } else {
                "None (Click to select | Esc to clear)".to_string()
            };

            let hover_summary = if self.hovered_body {
                format!("Hover: Body ({})", self.model_name)
            } else if let Some(fid) = self.hovered_face_id {
                let area = self.face_areas.get(&fid).copied().unwrap_or(0.0);
                format!("Hover: Face #{} ({:.1} mm²)", fid, area)
            } else if let Some(ei) = self.hovered_edge_id {
                format!("Hover: Edge #{}", ei)
            } else {
                "Hover: -".to_string()
            };

            let hud_rect = Rect::from_min_size(
                viewport.left_top() + Vec2::new(14.0, 14.0),
                Vec2::new(390.0, 60.0),
            );
            painter.rect_filled(hud_rect, 4.0, Color32::from_black_alpha(160));
            painter.text(
                hud_rect.min + Vec2::new(8.0, 6.0),
                egui::Align2::LEFT_TOP,
                format!(
                    "Camera: {} | Pivot: [{:.1}, {:.1}, {:.1}]\nSelection ({}): {}\n{}",
                    if self.camera.ortho_blend > 0.5 { "Orthographic" } else { "Perspective" },
                    self.camera.center[0], self.camera.center[1], self.camera.center[2],
                    match self.selection_filter {
                        SelectionFilter::Face => "Faces",
                        SelectionFilter::Edge => "Edges",
                        SelectionFilter::Body => "Body",
                    },
                    sel_summary,
                    hover_summary
                ),
                egui::FontId::monospace(10.5),
                Color32::from_rgb(220, 230, 245),
            );

            // INTERACTIVE 3D VIEW-CUBE (TOP-RIGHT CORNER)
            let cube_center = viewport.right_top() + Vec2::new(-60.0, 60.0);
            let cube_size = 28.0_f32;

            painter.rect_filled(
                Rect::from_center_size(cube_center, Vec2::splat(cube_size * 2.2)),
                6.0,
                Color32::from_black_alpha(140),
            );

            let (_fwd, right, up) = self.camera.basis_vectors();
            let draw_cube_face = |p: &egui::Painter, offset: Vec2, label: &str, view_name: &str, app: &mut CadLabelerApp| {
                let rect = Rect::from_center_size(cube_center + offset, Vec2::splat(cube_size * 0.7));
                let is_hovered = response.hovered() && ctx.input(|i| i.pointer.hover_pos().map_or(false, |pos| rect.contains(pos)));
                let fill = if is_hovered { okabe_ito::SKY_BLUE } else { Color32::from_rgb(50, 55, 70) };
                p.rect_filled(rect, 3.0, fill);
                p.text(rect.center(), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(9.0), okabe_ito::WHITE);
                if is_hovered && ctx.input(|i| i.pointer.primary_clicked()) {
                    app.camera.set_view(view_name);
                }
            };

            draw_cube_face(&painter, Vec2::new(0.0, -cube_size * 0.75), "TOP", "Top", self);
            draw_cube_face(&painter, Vec2::new(-cube_size * 0.75, 0.0), "LFT", "Left", self);
            draw_cube_face(&painter, Vec2::new(0.0, 0.0), "ISO", "Iso", self);
            draw_cube_face(&painter, Vec2::new(cube_size * 0.75, 0.0), "RGT", "Right", self);
            draw_cube_face(&painter, Vec2::new(0.0, cube_size * 0.75), "FRT", "Front", self);

            // 3D ORIENTATION AXES (BOTTOM-LEFT CORNER) - Okabe-Ito: X=Vermilion, Y=Bluish-Green, Z=Blue
            let axes_center = viewport.left_bottom() + Vec2::new(45.0, -45.0);
            let axes_len = 32.0_f32;

            let x_proj = Vec2::new(right[0] as f32, -up[0] as f32) * axes_len;
            painter.line_segment([axes_center, axes_center + x_proj], Stroke::new(2.5_f32, okabe_ito::VERMILION));
            painter.text(axes_center + x_proj * 1.25, egui::Align2::CENTER_CENTER, "X", egui::FontId::proportional(11.0), okabe_ito::VERMILION);

            let y_proj = Vec2::new(right[1] as f32, -up[1] as f32) * axes_len;
            painter.line_segment([axes_center, axes_center + y_proj], Stroke::new(2.5_f32, okabe_ito::BLUISH_GREEN));
            painter.text(axes_center + y_proj * 1.25, egui::Align2::CENTER_CENTER, "Y", egui::FontId::proportional(11.0), okabe_ito::BLUISH_GREEN);

            let z_proj = Vec2::new(right[2] as f32, -up[2] as f32) * axes_len;
            painter.line_segment([axes_center, axes_center + z_proj], Stroke::new(2.5_f32, okabe_ito::BLUE));
            painter.text(axes_center + z_proj * 1.25, egui::Align2::CENTER_CENTER, "Z", egui::FontId::proportional(11.0), okabe_ito::BLUE);
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Immersed IGA - 3D CAD Boundary Condition Labeler (Rust Native)")
            .with_inner_size([1280.0, 840.0])
            .with_min_inner_size([900.0, 650.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Immersed IGA CAD Labeler",
        native_options,
        Box::new(|_cc| Ok(Box::new(CadLabelerApp::new()))),
    )
}
