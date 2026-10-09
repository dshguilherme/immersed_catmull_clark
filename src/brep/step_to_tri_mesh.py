import os
import sys
import argparse
import numpy as np

def step_to_tri_mesh(step_path, out_obj_path=None, mesh_size=None):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    if mesh_size is not None and mesh_size > 0:
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        
    gmsh.open(step_path)
    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    
    # Extract nodes and 2D triangular elements
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
    
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    
    # Get 3-node triangles (element type 2 in Gmsh)
    tri_tags, tri_node_tags = gmsh.model.mesh.getElementsByType(2)
    tri_node_tags = np.array(tri_node_tags, dtype=np.int64).reshape(-1, 3)
    
    faces = np.vectorize(tag_to_idx.get)(tri_node_tags)
    
    gmsh.finalize()
    
    if out_obj_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_obj_path)), exist_ok=True)
        with open(out_obj_path, "w") as f:
            for v in nodes:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for tri in faces:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
                
    return nodes, faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert STEP to triangular surface OBJ via Gmsh")
    parser.add_argument("step_file", type=str, help="Path to input STEP file")
    parser.add_argument("out_obj", type=str, help="Path to output OBJ file")
    parser.add_argument("--mesh_size", type=float, default=None, help="Target mesh element size")
    args = parser.parse_args()
    
    nodes, faces = step_to_tri_mesh(args.step_file, args.out_obj, args.mesh_size)
    print(f"Exported {len(nodes)} vertices and {len(faces)} triangular faces to {args.out_obj}")
