//! Labeled CAD Geometry, Bodies, Faces, and Boundary Condition Dispatcher.
//! Enables named tagging of CAD surfaces/domains for material assignment and BC imposition.

use std::collections::HashMap;
use crate::cut_cell::TriangleMesh3D;

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

    /// Loads a CAD assembly and boundary conditions from a JSON string
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let root = parse_json(json_str)?;
        let obj = root.as_object().ok_or("Root JSON must be an object")?;

        let cad_file = obj.get("cad_file").and_then(|v| v.as_str()).unwrap_or("CantileverBracket");
        
        let mat_name = obj.get("material")
            .and_then(|m| m.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("StructuralSteel");
        let youngs = obj.get("material")
            .and_then(|m| m.get("youngs_modulus"))
            .and_then(|v| v.as_f64())
            .unwrap_or(2.1e5);
        let poissons = obj.get("material")
            .and_then(|m| m.get("poissons_ratio"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.30);
        let density = obj.get("material")
            .and_then(|m| m.get("density"))
            .and_then(|v| v.as_f64())
            .unwrap_or(7850.0);

        let material = MaterialProperties {
            name: mat_name.to_string(),
            youngs_modulus: youngs,
            poissons_ratio: poissons,
            density,
        };

        // Determine mesh: if cad_file ends with .obj and file exists, load it
        let mut body = if cad_file.ends_with(".obj") && std::path::Path::new(cad_file).exists() {
            let obj_content = std::fs::read_to_string(cad_file)
                .map_err(|e| format!("Failed to read CAD file '{}': {}", cad_file, e))?;
            CadBody::parse_obj(&obj_content, material)?
        } else {
            // Default box mesh [0,0,0] to [2,1,1]
            let min_b = [0.0, 0.0, 0.0];
            let max_b = [2.0, 1.0, 1.0];
            let mesh = TriangleMesh3D::new_box(min_b, max_b);
            CadBody::new(cad_file, mesh, material)
        };

        let mut assembly = CadAssembly::new();

        // Process labeled faces and BCs from JSON
        if let Some(faces_arr) = obj.get("faces").and_then(|v| v.as_array()) {
            for face_val in faces_arr {
                if let Some(face_obj) = face_val.as_object() {
                    let name = face_obj.get("name").and_then(|v| v.as_str()).unwrap_or("Face");
                    let mut tri_indices = Vec::new();
                    if let Some(tri_arr) = face_obj.get("triangles").and_then(|v| v.as_array()) {
                        for t in tri_arr {
                            if let Some(idx) = t.as_f64() {
                                tri_indices.push(idx as usize);
                            }
                        }
                    }

                    // Compute area, centroid, normal for face
                    let mut area = 0.0;
                    let mut c_sum = [0.0; 3];
                    let mut n_sum = [0.0; 3];
                    for &ti in &tri_indices {
                        if ti < body.mesh.triangles.len() {
                            let (n, a) = body.triangle_normal_and_area(ti);
                            area += a;
                            let tri = body.mesh.triangles[ti];
                            for k in 0..3 {
                                c_sum[k] += a * (body.mesh.vertices[tri[0]][k] + body.mesh.vertices[tri[1]][k] + body.mesh.vertices[tri[2]][k]) / 3.0;
                                n_sum[k] += a * n[k];
                            }
                        }
                    }
                    if area > 0.0 {
                        c_sum[0] /= area; c_sum[1] /= area; c_sum[2] /= area;
                        let nl = (n_sum[0].powi(2) + n_sum[1].powi(2) + n_sum[2].powi(2)).sqrt();
                        if nl > 0.0 { n_sum[0] /= nl; n_sum[1] /= nl; n_sum[2] /= nl; }
                    }

                    body.faces.insert(name.to_string(), CadFace {
                        name: name.to_string(),
                        triangle_indices: tri_indices,
                        total_area: area,
                        centroid: c_sum,
                        normal: n_sum,
                    });

                    // Parse Boundary Condition if present
                    if let Some(bc_obj) = face_obj.get("bc").and_then(|v| v.as_object()) {
                        let bc_type_str = bc_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match bc_type_str {
                            "Dirichlet" => {
                                let mut components = [true, true, true];
                                if let Some(comp_arr) = bc_obj.get("components").and_then(|v| v.as_array()) {
                                    if comp_arr.len() >= 3 {
                                        for i in 0..3 {
                                            components[i] = comp_arr[i].as_bool().unwrap_or(true);
                                        }
                                    }
                                }
                                let mut values = [0.0, 0.0, 0.0];
                                if let Some(val_arr) = bc_obj.get("values").and_then(|v| v.as_array()) {
                                    if val_arr.len() >= 3 {
                                        for i in 0..3 {
                                            values[i] = val_arr[i].as_f64().unwrap_or(0.0);
                                        }
                                    }
                                }
                                assembly.add_boundary_condition(LabeledBoundaryCondition {
                                    target_face_name: name.to_string(),
                                    bc_type: BoundaryConditionType::Dirichlet { components, values },
                                });
                            }
                            "NeumannTraction" => {
                                let mut traction = [0.0, 0.0, 0.0];
                                if let Some(t_arr) = bc_obj.get("traction").and_then(|v| v.as_array()) {
                                    if t_arr.len() >= 3 {
                                        for i in 0..3 {
                                            traction[i] = t_arr[i].as_f64().unwrap_or(0.0);
                                        }
                                    }
                                }
                                assembly.add_boundary_condition(LabeledBoundaryCondition {
                                    target_face_name: name.to_string(),
                                    bc_type: BoundaryConditionType::NeumannTraction { traction },
                                });
                            }
                            "NeumannPressure" => {
                                let pressure = bc_obj.get("pressure").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                assembly.add_boundary_condition(LabeledBoundaryCondition {
                                    target_face_name: name.to_string(),
                                    bc_type: BoundaryConditionType::NeumannPressure { pressure },
                                });
                            }
                            "RobinFoundation" => {
                                let kn = bc_obj.get("normal_stiffness").and_then(|v| v.as_f64()).unwrap_or(1e4);
                                let kt = bc_obj.get("tangential_stiffness").and_then(|v| v.as_f64()).unwrap_or(1e4);
                                assembly.add_boundary_condition(LabeledBoundaryCondition {
                                    target_face_name: name.to_string(),
                                    bc_type: BoundaryConditionType::RobinFoundation {
                                        normal_stiffness: kn,
                                        tangential_stiffness: kt,
                                    },
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        assembly.add_body(body);
        Ok(assembly)
    }
}

// ============================================================================
// ZERO-DEPENDENCY JSON VALUE PARSER
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, JsonValue>> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|o| o.get(key))
    }
}

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;
    skip_whitespace(&chars, &mut pos);
    let val = parse_json_value(&chars, &mut pos)?;
    skip_whitespace(&chars, &mut pos);
    Ok(val)
}

fn skip_whitespace(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && (chars[*pos].is_whitespace() || chars[*pos] == '\r' || chars[*pos] == '\n') {
        *pos += 1;
    }
}

fn parse_json_value(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
    skip_whitespace(chars, pos);
    if *pos >= chars.len() {
        return Err("Unexpected end of JSON input".to_string());
    }

    match chars[*pos] {
        '{' => parse_json_object(chars, pos),
        '[' => parse_json_array(chars, pos),
        '"' => parse_json_string(chars, pos).map(JsonValue::String),
        't' | 'f' => parse_json_bool(chars, pos),
        'n' => parse_json_null(chars, pos),
        '-' | '0'..='9' => parse_json_number(chars, pos),
        c => Err(format!("Unexpected character '{}' at position {}", c, pos)),
    }
}

fn parse_json_object(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
    *pos += 1; // skip '{'
    let mut map = HashMap::new();
    skip_whitespace(chars, pos);

    if *pos < chars.len() && chars[*pos] == '}' {
        *pos += 1;
        return Ok(JsonValue::Object(map));
    }

    loop {
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] != '"' {
            return Err(format!("Expected string key in object at position {}", pos));
        }
        let key = parse_json_string(chars, pos)?;
        skip_whitespace(chars, pos);
        if *pos >= chars.len() || chars[*pos] != ':' {
            return Err(format!("Expected ':' after key '{}' at position {}", key, pos));
        }
        *pos += 1; // skip ':'
        let val = parse_json_value(chars, pos)?;
        map.insert(key, val);

        skip_whitespace(chars, pos);
        if *pos < chars.len() && chars[*pos] == ',' {
            *pos += 1;
            continue;
        } else if *pos < chars.len() && chars[*pos] == '}' {
            *pos += 1;
            break;
        } else {
            return Err(format!("Expected ',' or '}}' at position {}", pos));
        }
    }

    Ok(JsonValue::Object(map))
}

fn parse_json_array(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
    *pos += 1; // skip '['
    let mut vec = Vec::new();
    skip_whitespace(chars, pos);

    if *pos < chars.len() && chars[*pos] == ']' {
        *pos += 1;
        return Ok(JsonValue::Array(vec));
    }

    loop {
        let val = parse_json_value(chars, pos)?;
        vec.push(val);
        skip_whitespace(chars, pos);
        if *pos < chars.len() && chars[*pos] == ',' {
            *pos += 1;
            continue;
        } else if *pos < chars.len() && chars[*pos] == ']' {
            *pos += 1;
            break;
        } else {
            return Err(format!("Expected ',' or ']' at position {}", pos));
        }
    }

    Ok(JsonValue::Array(vec))
}

fn parse_json_string(chars: &[char], pos: &mut usize) -> Result<String, String> {
    *pos += 1; // skip opening '"'
    let mut s = String::new();
    while *pos < chars.len() {
        let c = chars[*pos];
        *pos += 1;
        if c == '"' {
            return Ok(s);
        } else if c == '\\' {
            if *pos >= chars.len() {
                return Err("Unterminated escape sequence in string".to_string());
            }
            let esc = chars[*pos];
            *pos += 1;
            match esc {
                '"' => s.push('"'),
                '\\' => s.push('\\'),
                '/' => s.push('/'),
                'n' => s.push('\n'),
                't' => s.push('\t'),
                'r' => s.push('\r'),
                other => s.push(other),
            }
        } else {
            s.push(c);
        }
    }
    Err("Unterminated string literal".to_string())
}

fn parse_json_number(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
    let start = *pos;
    if chars[*pos] == '-' {
        *pos += 1;
    }
    while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '.' || chars[*pos] == 'e' || chars[*pos] == 'E' || chars[*pos] == '+' || chars[*pos] == '-') {
        *pos += 1;
    }
    let num_str: String = chars[start..*pos].iter().collect();
    let num: f64 = num_str.parse().map_err(|e| format!("Invalid number '{}': {:?}", num_str, e))?;
    Ok(JsonValue::Number(num))
}

fn parse_json_bool(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
    if chars[*pos..].starts_with(&['t', 'r', 'u', 'e']) {
        *pos += 4;
        Ok(JsonValue::Bool(true))
    } else if chars[*pos..].starts_with(&['f', 'a', 'l', 's', 'e']) {
        *pos += 5;
        Ok(JsonValue::Bool(false))
    } else {
        Err(format!("Invalid boolean token at position {}", pos))
    }
}

fn parse_json_null(chars: &[char], pos: &mut usize) -> Result<JsonValue, String> {
    if chars[*pos..].starts_with(&['n', 'u', 'l', 'l']) {
        *pos += 4;
        Ok(JsonValue::Null)
    } else {
        Err(format!("Invalid null token at position {}", pos))
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

    #[test]
    fn test_parse_json_and_cad_assembly() {
        let json_input = r#"{
            "cad_file": "CantileverTest",
            "material": {
                "name": "Steel",
                "youngs_modulus": 200000.0,
                "poissons_ratio": 0.28,
                "density": 7800.0
            },
            "faces": [
                {
                    "name": "FixFace",
                    "triangles": [8, 9],
                    "bc": {
                        "type": "Dirichlet",
                        "components": [true, true, true],
                        "values": [0.0, 0.0, 0.0]
                    }
                },
                {
                    "name": "ForceFace",
                    "triangles": [10, 11],
                    "bc": {
                        "type": "NeumannTraction",
                        "traction": [0.0, -50.0, 0.0]
                    }
                }
            ]
        }"#;

        let assembly = CadAssembly::from_json(json_input).expect("Failed to parse CAD JSON");
        assert_eq!(assembly.bodies.len(), 1);
        assert_eq!(assembly.bodies[0].material.name, "Steel");
        assert_eq!(assembly.bodies[0].material.youngs_modulus, 200000.0);
        assert_eq!(assembly.boundary_conditions.len(), 2);
        assert!(assembly.bodies[0].faces.contains_key("FixFace"));
        assert!(assembly.bodies[0].faces.contains_key("ForceFace"));
    }
}
