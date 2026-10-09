import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np

def convert_step_or_msh(input_path, out_obj_path=None, mesh_size=None):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    if mesh_size is not None and mesh_size > 0:
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        
    gmsh.open(input_path)
    
    # If 2D surface mesh isn't already present, generate it
    # For .step, this meshes the B-Rep surfaces
    # For .msh, it reads existing mesh
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.stp', '.step', '.iges', '.igs', '.brep']:
        gmsh.model.mesh.generate(2)
        
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    
    # Extract 3-node triangles (type 2 in Gmsh)
    tri_tags, tri_node_tags = gmsh.model.mesh.getElementsByType(2)
    
    if len(tri_tags) > 0:
        tri_node_tags = np.array(tri_node_tags, dtype=np.int64).reshape(-1, 3)
        faces = np.vectorize(tag_to_idx.get)(tri_node_tags)
    else:
        # Check for 4-node quads (type 3 in Gmsh) and split into 2 triangles
        quad_tags, quad_node_tags = gmsh.model.mesh.getElementsByType(3)
        if len(quad_tags) > 0:
            quads = np.array(quad_node_tags, dtype=np.int64).reshape(-1, 4)
            q_idx = np.vectorize(tag_to_idx.get)(quads)
            f1 = q_idx[:, [0, 1, 2]]
            f2 = q_idx[:, [0, 2, 3]]
            faces = np.vstack([f1, f2])
        else:
            faces = np.empty((0, 3), dtype=np.int64)
            
    gmsh.finalize()
    return nodes, faces

def convert_vtu(vtu_path):
    tree = ET.parse(vtu_path)
    root = tree.getroot()
    
    # Find Points DataArray
    points_elem = root.find(".//Piece/Points/DataArray")
    if points_elem is None:
        raise ValueError("Could not find Points DataArray in VTU")
    coords_text = points_elem.text.strip()
    nodes = np.fromstring(coords_text, sep=' ', dtype=np.float64).reshape(-1, 3)
    
    # Find Cells connectivity and types
    conn_elem = root.find(".//Piece/Cells/DataArray[@Name='connectivity']")
    types_elem = root.find(".//Piece/Cells/DataArray[@Name='types']")
    offsets_elem = root.find(".//Piece/Cells/DataArray[@Name='offsets']")
    
    conn = np.fromstring(conn_elem.text.strip(), sep=' ', dtype=np.int64)
    types = np.fromstring(types_elem.text.strip(), sep=' ', dtype=np.int32)
    offsets = np.fromstring(offsets_elem.text.strip(), sep=' ', dtype=np.int64)
    
    faces = []
    prev_offset = 0
    for cell_type, offset in zip(types, offsets):
        cell_conn = conn[prev_offset:offset]
        prev_offset = offset
        if cell_type == 5: # Triangle
            faces.append(cell_conn[:3])
        elif cell_type == 9: # Quad
            faces.append([cell_conn[0], cell_conn[1], cell_conn[2]])
            faces.append([cell_conn[0], cell_conn[2], cell_conn[3]])
        elif cell_type == 10: # Tetrahedron: extract 4 boundary triangular faces
            faces.append([cell_conn[0], cell_conn[1], cell_conn[2]])
            faces.append([cell_conn[0], cell_conn[1], cell_conn[3]])
            faces.append([cell_conn[1], cell_conn[2], cell_conn[3]])
            faces.append([cell_conn[0], cell_conn[2], cell_conn[3]])
            
    faces = np.array(faces, dtype=np.int64)
    return nodes, faces

def export_obj(nodes, faces, out_obj_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_obj_path)), exist_ok=True)
    with open(out_obj_path, "w") as f:
        for v in nodes:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified B-Rep/Mesh to OBJ converter (STEP/MSH/VTU)")
    parser.add_argument("input_file", type=str, help="Path to input CAD/mesh file")
    parser.add_argument("out_obj", type=str, help="Path to output OBJ file")
    parser.add_argument("--mesh_size", type=float, default=None, help="Target mesh element size")
    args = parser.parse_args()
    
    ext = os.path.splitext(args.input_file)[1].lower()
    if ext == '.vtu':
        nodes, faces = convert_vtu(args.input_file)
    else:
        nodes, faces = convert_step_or_msh(args.input_file, args.out_obj, args.mesh_size)
        
    export_obj(nodes, faces, args.out_obj)
    print(f"Exported {len(nodes)} vertices and {len(faces)} triangular faces to {args.out_obj}")
