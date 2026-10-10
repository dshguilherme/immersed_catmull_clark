//! Generational Arena Topological Storage & Bi-directional Navigation.
//! Pillar 1: Avoids pointer graphs and cyclic references using generational slot storage.
//! Features:
//! - Small, Copy-capable EntityHandles (Index + Generation + Kind).
//! - Bi-directional upward (Vertex -> Edge -> Face -> Body -> Assembly) and downward links.
//! - Bubble-up (traverse sub-entity to parent body/assembly) and Drill-down (expand body into sub-topologies).

use std::collections::HashSet;

/// Topological entity hierarchy levels
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Vertex,
    Edge,
    Face,
    Wire,
    Body,
    Assembly,
}

/// Small, Copy-capable generational handle (8 bytes on 64-bit systems)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntityHandle {
    pub index: u32,
    pub generation: u32,
    pub kind: EntityKind,
}

impl EntityHandle {
    pub fn new(index: u32, generation: u32, kind: EntityKind) -> Self {
        Self { index, generation, kind }
    }
}

/// Generational slot entry
#[derive(Clone, Debug)]
struct Slot<T> {
    data: Option<T>,
    generation: u32,
}

impl<T> Slot<T> {
    fn new(data: T) -> Self {
        Self {
            data: Some(data),
            generation: 1,
        }
    }
}

/// Topological Vertex entity
#[derive(Clone, Debug)]
pub struct TopoVertex {
    pub point: [f64; 3],
    pub up_edges: Vec<EntityHandle>,
}

/// Topological Edge entity
#[derive(Clone, Debug)]
pub struct TopoEdge {
    pub v0: EntityHandle,
    pub v1: EntityHandle,
    pub up_faces: Vec<EntityHandle>,
    pub is_sharp: bool,
    pub is_boundary: bool,
}

/// Topological Face entity
#[derive(Clone, Debug)]
pub struct TopoFace {
    pub down_edges: Vec<EntityHandle>,
    pub down_vertices: Vec<EntityHandle>,
    pub up_body: Option<EntityHandle>,
    pub area: f64,
    pub normal: [f64; 3],
    pub centroid: [f64; 3],
    pub triangle_indices: Vec<usize>,
}

/// Topological Solid Body entity
#[derive(Clone, Debug)]
pub struct TopoBody {
    pub name: String,
    pub down_faces: Vec<EntityHandle>,
    pub up_assembly: Option<EntityHandle>,
}

/// Topological Assembly entity
#[derive(Clone, Debug)]
pub struct TopoAssembly {
    pub name: String,
    pub down_bodies: Vec<EntityHandle>,
}

/// Central generational topological arena
#[derive(Clone, Debug, Default)]
pub struct TopologyArena {
    vertices: Vec<Slot<TopoVertex>>,
    edges: Vec<Slot<TopoEdge>>,
    faces: Vec<Slot<TopoFace>>,
    bodies: Vec<Slot<TopoBody>>,
    assemblies: Vec<Slot<TopoAssembly>>,
}

impl TopologyArena {
    pub fn new() -> Self {
        Self::default()
    }

    // --- Entity Allocators ---

    pub fn insert_vertex(&mut self, point: [f64; 3]) -> EntityHandle {
        let index = self.vertices.len() as u32;
        let v = TopoVertex {
            point,
            up_edges: Vec::new(),
        };
        self.vertices.push(Slot::new(v));
        EntityHandle::new(index, 1, EntityKind::Vertex)
    }

    pub fn insert_edge(&mut self, v0: EntityHandle, v1: EntityHandle, is_sharp: bool, is_boundary: bool) -> EntityHandle {
        let index = self.edges.len() as u32;
        let handle = EntityHandle::new(index, 1, EntityKind::Edge);
        let e = TopoEdge {
            v0,
            v1,
            up_faces: Vec::new(),
            is_sharp,
            is_boundary,
        };
        self.edges.push(Slot::new(e));

        // Connect upward links from vertices
        if let Some(slot) = self.vertices.get_mut(v0.index as usize) {
            if let Some(ref mut vert) = slot.data {
                vert.up_edges.push(handle);
            }
        }
        if let Some(slot) = self.vertices.get_mut(v1.index as usize) {
            if let Some(ref mut vert) = slot.data {
                vert.up_edges.push(handle);
            }
        }

        handle
    }

    pub fn insert_face(
        &mut self,
        down_edges: Vec<EntityHandle>,
        down_vertices: Vec<EntityHandle>,
        area: f64,
        normal: [f64; 3],
        centroid: [f64; 3],
        triangle_indices: Vec<usize>,
    ) -> EntityHandle {
        let index = self.faces.len() as u32;
        let handle = EntityHandle::new(index, 1, EntityKind::Face);
        let f = TopoFace {
            down_edges: down_edges.clone(),
            down_vertices,
            up_body: None,
            area,
            normal,
            centroid,
            triangle_indices,
        };
        self.faces.push(Slot::new(f));

        // Connect upward links from edges
        for eh in down_edges {
            if let Some(slot) = self.edges.get_mut(eh.index as usize) {
                if let Some(ref mut edge) = slot.data {
                    edge.up_faces.push(handle);
                }
            }
        }

        handle
    }

    pub fn insert_body(&mut self, name: impl Into<String>, down_faces: Vec<EntityHandle>) -> EntityHandle {
        let index = self.bodies.len() as u32;
        let handle = EntityHandle::new(index, 1, EntityKind::Body);
        let b = TopoBody {
            name: name.into(),
            down_faces: down_faces.clone(),
            up_assembly: None,
        };
        self.bodies.push(Slot::new(b));

        // Connect upward links from faces
        for fh in down_faces {
            if let Some(slot) = self.faces.get_mut(fh.index as usize) {
                if let Some(ref mut face) = slot.data {
                    face.up_body = Some(handle);
                }
            }
        }

        handle
    }

    pub fn insert_assembly(&mut self, name: impl Into<String>, down_bodies: Vec<EntityHandle>) -> EntityHandle {
        let index = self.assemblies.len() as u32;
        let handle = EntityHandle::new(index, 1, EntityKind::Assembly);
        let a = TopoAssembly {
            name: name.into(),
            down_bodies: down_bodies.clone(),
        };
        self.assemblies.push(Slot::new(a));

        // Connect upward links from bodies
        for bh in down_bodies {
            if let Some(slot) = self.bodies.get_mut(bh.index as usize) {
                if let Some(ref mut body) = slot.data {
                    body.up_assembly = Some(handle);
                }
            }
        }

        handle
    }

    // --- Generational Handle Getters ---

    pub fn get_vertex(&self, h: EntityHandle) -> Option<&TopoVertex> {
        if h.kind != EntityKind::Vertex { return None; }
        self.vertices.get(h.index as usize).and_then(|s| if s.generation == h.generation { s.data.as_ref() } else { None })
    }

    pub fn get_edge(&self, h: EntityHandle) -> Option<&TopoEdge> {
        if h.kind != EntityKind::Edge { return None; }
        self.edges.get(h.index as usize).and_then(|s| if s.generation == h.generation { s.data.as_ref() } else { None })
    }

    pub fn get_face(&self, h: EntityHandle) -> Option<&TopoFace> {
        if h.kind != EntityKind::Face { return None; }
        self.faces.get(h.index as usize).and_then(|s| if s.generation == h.generation { s.data.as_ref() } else { None })
    }

    pub fn get_body(&self, h: EntityHandle) -> Option<&TopoBody> {
        if h.kind != EntityKind::Body { return None; }
        self.bodies.get(h.index as usize).and_then(|s| if s.generation == h.generation { s.data.as_ref() } else { None })
    }

    pub fn get_assembly(&self, h: EntityHandle) -> Option<&TopoAssembly> {
        if h.kind != EntityKind::Assembly { return None; }
        self.assemblies.get(h.index as usize).and_then(|s| if s.generation == h.generation { s.data.as_ref() } else { None })
    }

    // --- Hierarchy Navigation Operations ---

    /// Bubble-up: Traverses a sub-entity upwards directly to parent Body or Assembly
    pub fn bubble_up(&self, handle: EntityHandle, target: EntityKind) -> Option<EntityHandle> {
        if handle.kind == target {
            return Some(handle);
        }

        match handle.kind {
            EntityKind::Vertex => {
                let v = self.get_vertex(handle)?;
                let first_edge = v.up_edges.first()?;
                self.bubble_up(*first_edge, target)
            }
            EntityKind::Edge => {
                let e = self.get_edge(handle)?;
                let first_face = e.up_faces.first()?;
                self.bubble_up(*first_face, target)
            }
            EntityKind::Face => {
                let f = self.get_face(handle)?;
                let body_h = f.up_body?;
                if target == EntityKind::Body {
                    Some(body_h)
                } else {
                    self.bubble_up(body_h, target)
                }
            }
            EntityKind::Body => {
                let b = self.get_body(handle)?;
                if target == EntityKind::Body {
                    Some(handle)
                } else if target == EntityKind::Assembly {
                    b.up_assembly
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Drill-down: Expands an assembly or body handle into its boundary constituent sub-topologies
    pub fn drill_down(&self, handle: EntityHandle, target: EntityKind) -> Vec<EntityHandle> {
        if handle.kind == target {
            return vec![handle];
        }

        match handle.kind {
            EntityKind::Assembly => {
                if let Some(a) = self.get_assembly(handle) {
                    let mut results = Vec::new();
                    for &bh in &a.down_bodies {
                        results.extend(self.drill_down(bh, target));
                    }
                    results
                } else {
                    Vec::new()
                }
            }
            EntityKind::Body => {
                if let Some(b) = self.get_body(handle) {
                    if target == EntityKind::Face {
                        b.down_faces.clone()
                    } else {
                        let mut results = HashSet::new();
                        for &fh in &b.down_faces {
                            for sub in self.drill_down(fh, target) {
                                results.insert(sub);
                            }
                        }
                        results.into_iter().collect()
                    }
                } else {
                    Vec::new()
                }
            }
            EntityKind::Face => {
                if let Some(f) = self.get_face(handle) {
                    match target {
                        EntityKind::Edge => f.down_edges.clone(),
                        EntityKind::Vertex => f.down_vertices.clone(),
                        _ => Vec::new(),
                    }
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generational_arena_and_navigation() {
        let mut arena = TopologyArena::new();

        // Build a simple 3D triangle face
        let v0 = arena.insert_vertex([0.0, 0.0, 0.0]);
        let v1 = arena.insert_vertex([1.0, 0.0, 0.0]);
        let v2 = arena.insert_vertex([0.0, 1.0, 0.0]);

        let e0 = arena.insert_edge(v0, v1, true, false);
        let e1 = arena.insert_edge(v1, v2, true, false);
        let e2 = arena.insert_edge(v2, v0, true, false);

        let f0 = arena.insert_face(
            vec![e0, e1, e2],
            vec![v0, v1, v2],
            0.5,
            [0.0, 0.0, 1.0],
            [0.33, 0.33, 0.0],
            vec![0],
        );

        let body = arena.insert_body("SolidBracket", vec![f0]);
        let asm = arena.insert_assembly("MainChassis", vec![body]);

        // Test Bubble-up from Vertex to Body and Assembly
        let parent_body = arena.bubble_up(v0, EntityKind::Body);
        assert_eq!(parent_body, Some(body));

        let parent_asm = arena.bubble_up(v0, EntityKind::Assembly);
        assert_eq!(parent_asm, Some(asm));

        // Test Drill-down from Assembly to Faces
        let faces = arena.drill_down(asm, EntityKind::Face);
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0], f0);

        // Test Drill-down from Body to Edges
        let edges = arena.drill_down(body, EntityKind::Edge);
        assert_eq!(edges.len(), 3);
    }
}
