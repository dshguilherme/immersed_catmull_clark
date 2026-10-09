#!/usr/bin/env python3
"""
Interactive 3D CAD Boundary Condition Labeler & Immersed Solver GUI.
Zero-MATLAB, pure Python (PyVista/VTK + Tkinter) + Native Rust solver integration.

Enables manual/interactive click-to-pick surface selection, CAD face clustering,
Dirichlet/Neumann/Robin boundary condition assignment, material definition,
and direct execution of the Rust matrix-free PCG solver with 3D deformed visualization.
"""

import sys
import os
import json
import subprocess
import threading
import time
from collections import deque
import numpy as np

try:
    import pyvista as pv
    import vtk
except ImportError:
    print("Error: PyVista and VTK are required. Run: pip install pyvista vtk")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


class CadBcLabeler:
    """Core interactive model, face segmenter, and solver coordinator."""

    def __init__(self, cad_path=None, angle_deg=20.0, off_screen=False):
        self.cad_path = cad_path
        self.angle_deg = angle_deg
        self.off_screen = off_screen

        # Mesh and segmentation
        self.mesh = None
        self.cell_normals = None
        self.face_labels = None
        self.num_faces = 0
        self.face_cell_map = {}  # face_id -> list of cell indices
        self.face_info = {}      # face_id -> dict(area, normal, centroid)

        # User boundary conditions: dict(face_name -> dict(face_id, type, params, triangles))
        self.assigned_bcs = {}
        self.selected_face_id = None

        # Material properties
        self.material = {
            "name": "StructuralSteel",
            "youngs_modulus": 2.1e5,  # MPa
            "poissons_ratio": 0.30,
            "density": 7850.0         # kg/m^3
        }

        # Plotter and UI
        self.plotter = None
        self.base_actor = None
        self.highlight_actor = None
        self.bc_actors = {}
        self.hud_text_actor = None
        self.tk_app = None

        # Solution state
        self.solution_data = None
        self.deformed_actor = None

        self._load_mesh()
        self._segment_cad_faces()

    def _load_mesh(self):
        """Loads CAD geometry from file or synthesizes a cantilever beam benchmark."""
        if self.cad_path and os.path.exists(self.cad_path):
            print(f"[CAD Labeler] Loading 3D model from '{self.cad_path}'...")
            self.mesh = pv.read(self.cad_path)
        elif os.path.exists("Models/nist_ctc_01.obj"):
            self.cad_path = "Models/nist_ctc_01.obj"
            print(f"[CAD Labeler] Defaulting to NIST CAD Model '{self.cad_path}'...")
            self.mesh = pv.read(self.cad_path)
        else:
            print("[CAD Labeler] Synthesizing default Cantilever Beam benchmark (2x1x1)...")
            self.cad_path = "benchmarks/cantilever_bracket.obj"
            box = pv.Box(bounds=(0.0, 2.0, 0.0, 1.0, 0.0, 1.0)).triangulate()
            self.mesh = box
            os.makedirs("benchmarks", exist_ok=True)
            self.mesh.save(self.cad_path)

        if not self.mesh.is_all_triangles:
            print("[CAD Labeler] Triangulating surface mesh...")
            self.mesh = self.mesh.triangulate()

        # Compute cell normals
        self.mesh = self.mesh.compute_normals(cell_normals=True, point_normals=False)
        self.cell_normals = self.mesh.cell_data['Normals']
        print(f"[CAD Labeler] Mesh loaded: {self.mesh.n_points} vertices, {self.mesh.n_cells} triangles.")

    def _segment_cad_faces(self):
        """Groups contiguous triangles into CAD feature faces via dihedral normal threshold."""
        n_cells = self.mesh.n_cells
        cos_thresh = np.cos(np.radians(self.angle_deg))

        # Build edge -> cell connectivity
        edge_to_cells = {}
        for cid in range(n_cells):
            cell = self.mesh.get_cell(cid)
            pt_ids = cell.point_ids
            for i in range(3):
                edge = tuple(sorted((pt_ids[i], pt_ids[(i + 1) % 3])))
                edge_to_cells.setdefault(edge, []).append(cid)

        adj = [[] for _ in range(n_cells)]
        for _, c_list in edge_to_cells.items():
            if len(c_list) == 2:
                adj[c_list[0]].append(c_list[1])
                adj[c_list[1]].append(c_list[0])

        visited = np.zeros(n_cells, dtype=bool)
        face_labels = -np.ones(n_cells, dtype=int)
        current_face = 0

        for start_cell in range(n_cells):
            if visited[start_cell]:
                continue
            queue = deque([start_cell])
            visited[start_cell] = True
            face_labels[start_cell] = current_face
            face_cells = [start_cell]
            base_n = self.cell_normals[start_cell]

            while queue:
                curr = queue.popleft()
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        # Check normal alignment with seed face normal
                        if np.dot(base_n, self.cell_normals[neighbor]) >= cos_thresh:
                            visited[neighbor] = True
                            face_labels[neighbor] = current_face
                            face_cells.append(neighbor)
                            queue.append(neighbor)

            self.face_cell_map[current_face] = face_cells

            # Compute face metrics: area, normal, centroid
            face_submesh = self.mesh.extract_cells(face_cells)
            area = float(face_submesh.area)
            center = list(np.mean(face_submesh.points, axis=0))
            avg_normal = list(np.mean(self.cell_normals[face_cells], axis=0))
            n_len = np.linalg.norm(avg_normal)
            if n_len > 1e-12:
                avg_normal = [c / n_len for c in avg_normal]

            self.face_info[current_face] = {
                "area": area,
                "normal": avg_normal,
                "centroid": center,
                "cell_count": len(face_cells)
            }
            current_face += 1

        self.face_labels = face_labels
        self.num_faces = current_face
        print(f"[CAD Labeler] Segmented {n_cells} triangles into {self.num_faces} feature faces (angle = {self.angle_deg}°).")

    def select_face(self, face_id):
        """Highlights a specific CAD face in the 3D viewport."""
        if face_id < 0 or face_id >= self.num_faces:
            return

        self.selected_face_id = face_id
        info = self.face_info[face_id]
        print(f"[CAD Labeler] Selected Face #{face_id}: Triangles={info['cell_count']}, Area={info['area']:.4f}, Normal={np.round(info['normal'], 3)}")

        if self.plotter is not None:
            # Highlight selected face in vibrant gold
            cells = self.face_cell_map[face_id]
            submesh = self.mesh.extract_cells(cells)
            self.highlight_actor = self.plotter.add_mesh(
                submesh,
                color="#FFD700",
                line_width=3,
                name="highlight_selected_face",
                opacity=0.95,
                render=True
            )
            self._update_hud()

        if self.tk_app:
            self.tk_app.update_inspector(face_id, info)

    def assign_boundary_condition(self, name, bc_type, params=None):
        """Assigns a Dirichlet, Neumann, or Robin boundary condition to the currently selected face."""
        if self.selected_face_id is None:
            print("Error: No face currently selected.")
            return False

        face_id = self.selected_face_id
        triangles = self.face_cell_map[face_id]
        info = self.face_info[face_id]

        bc_entry = {
            "name": name,
            "face_id": face_id,
            "type": bc_type,
            "triangles": triangles,
            "params": params or {},
            "area": info["area"],
            "normal": info["normal"],
            "centroid": info["centroid"]
        }
        self.assigned_bcs[name] = bc_entry
        print(f"[CAD Labeler] Assigned BC '{name}' ({bc_type}) to Face #{face_id}.")

        self._render_bc_visuals(name, bc_entry)
        self._update_hud()
        if self.tk_app:
            self.tk_app.refresh_bc_list()
        return True

    def remove_boundary_condition(self, name):
        """Removes an assigned boundary condition."""
        if name in self.assigned_bcs:
            bc = self.assigned_bcs.pop(name)
            # Remove overlay actor
            actor_name = f"bc_overlay_{name}"
            arrow_name = f"bc_arrow_{name}"
            if self.plotter:
                self.plotter.remove_actor(actor_name)
                self.plotter.remove_actor(arrow_name)
                self.plotter.render()
            print(f"[CAD Labeler] Removed BC '{name}'.")
            self._update_hud()
            if self.tk_app:
                self.tk_app.refresh_bc_list()

    def _render_bc_visuals(self, name, bc_entry):
        """Renders color-coded 3D surface highlights and traction vector arrows."""
        if not self.plotter:
            return

        cells = bc_entry["triangles"]
        submesh = self.mesh.extract_cells(cells)
        bc_type = bc_entry["type"]
        actor_name = f"bc_overlay_{name}"
        arrow_name = f"bc_arrow_{name}"

        # Color palette: Dirichlet = Crimson Red, Neumann = Vivid Cyan, Pressure = Orange
        if bc_type == "Dirichlet":
            color = "#E63946"
        elif bc_type == "NeumannTraction":
            color = "#00F0FF"
        elif bc_type == "NeumannPressure":
            color = "#FF9E00"
        else:
            color = "#9B5DE5"

        self.plotter.add_mesh(submesh, color=color, name=actor_name, opacity=0.90, render=True)

        # If Neumann Traction, draw directional 3D arrow glyph
        if bc_type == "NeumannTraction" and "traction" in bc_entry["params"]:
            t_vec = np.array(bc_entry["params"]["traction"], dtype=float)
            norm = np.linalg.norm(t_vec)
            if norm > 1e-12:
                direction = t_vec / norm
                centroid = np.array(bc_entry["centroid"])
                # Compute bounding box span for scaling
                span = np.array(self.mesh.bounds[1::2]) - np.array(self.mesh.bounds[0::2])
                scale = float(np.max(span) * 0.15)
                arrow = pv.Arrow(start=centroid, direction=direction, scale=scale)
                self.plotter.add_mesh(arrow, color="#00F0FF", name=arrow_name, render=True)

    def _update_hud(self):
        """Updates top-left on-screen status text in PyVista window."""
        if not self.plotter:
            return

        lines = [
            "IMMERSED IGA - INTERACTIVE 3D CAD LABELER",
            f"Model: {os.path.basename(str(self.cad_path))} ({self.num_faces} faces, {self.mesh.n_cells} triangles)",
            f"Selected Face: #{self.selected_face_id if self.selected_face_id is not None else 'None'}",
            f"Assigned BCs: {len(self.assigned_bcs)}",
            "--------------------------------------------------",
            "Shortcuts: [Left Click]=Select  [D]=Clamp  [N]=Traction  [C]=Clear  [E]=Export  [S]=Solve Rust"
        ]
        text = "\n".join(lines)
        if self.hud_text_actor:
            self.plotter.remove_actor(self.hud_text_actor)
        self.hud_text_actor = self.plotter.add_text(
            text,
            position="upper_left",
            font_size=10,
            color="#E0E6ED",
            name="hud_text"
        )
        self.plotter.render()

    def export_json_configuration(self, output_path="benchmarks/cad_model_input.json"):
        """Serializes labeled CAD faces, boundary conditions, and materials to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        faces_data = []
        for name, bc in self.assigned_bcs.items():
            bc_data = {"type": bc["type"]}
            if bc["type"] == "Dirichlet":
                bc_data["components"] = bc["params"].get("components", [True, True, True])
                bc_data["values"] = bc["params"].get("values", [0.0, 0.0, 0.0])
            elif bc["type"] == "NeumannTraction":
                bc_data["traction"] = bc["params"].get("traction", [0.0, -100.0, 0.0])
            elif bc["type"] == "NeumannPressure":
                bc_data["pressure"] = bc["params"].get("pressure", 50.0)
            elif bc["type"] == "RobinFoundation":
                bc_data["normal_stiffness"] = bc["params"].get("normal_stiffness", 1e4)
                bc_data["tangential_stiffness"] = bc["params"].get("tangential_stiffness", 1e4)

            faces_data.append({
                "name": name,
                "triangles": [int(t) for t in bc["triangles"]],
                "bc": bc_data
            })

        config = {
            "cad_file": str(self.cad_path),
            "material": self.material,
            "faces": faces_data
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"[CAD Labeler] Exported configuration with {len(faces_data)} labeled BCs to '{output_path}'.")
        return output_path

    def solve_in_rust(self):
        """Invokes the native Rust solve_cad binary and renders the 3D deformed solution."""
        json_path = self.export_json_configuration()
        print(f"[CAD Labeler] Launching Native Rust Solver on '{json_path}'...")

        # Find executable: release first, then debug
        bin_release = os.path.abspath("rust/target/release/solve_cad.exe")
        bin_debug = os.path.abspath("rust/target/debug/solve_cad.exe")

        if os.path.exists(bin_release):
            exe_path = bin_release
        elif os.path.exists(bin_debug):
            exe_path = bin_debug
        else:
            print("[CAD Labeler] Compiling release binary solve_cad.exe...")
            subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "run_rust.ps1", "build", "--release", "--bin", "solve_cad"], check=True)
            exe_path = bin_release

        t0 = time.time()
        result = subprocess.run([exe_path, os.path.abspath(json_path)], capture_output=True, text=True)
        elapsed = (time.time() - t0) * 1000.0

        if result.returncode != 0:
            print(f"[CAD Labeler] Rust Solver Error:\n{result.stderr}")
            if self.tk_app:
                self.tk_app.log(f"Rust Solver Failed:\n{result.stderr}")
            return False

        print(f"[CAD Labeler] Rust Solver completed in {elapsed:.2f} ms:\n{result.stdout}")

        # Ingest solution JSON
        sol_file = "benchmarks/cad_solution.json"
        if not os.path.exists(sol_file):
            sol_file = "rust/benchmarks/cad_solution.json"

        if os.path.exists(sol_file):
            with open(sol_file, "r") as f:
                self.solution_data = json.load(f)
            self._render_solution(self.solution_data)
            if self.tk_app:
                self.tk_app.display_solution_stats(self.solution_data)
            return True
        return False

    def _render_solution(self, sol):
        """Renders deformed 3D point cloud / warped structural mesh in PyVista."""
        if not self.plotter or "nodes" not in sol or "displacements" not in sol:
            return

        nodes = np.array(sol["nodes"])
        disps = np.array(sol["displacements"])
        mag = np.linalg.norm(disps, axis=1) * 1000.0  # mm

        # Scale displacement for visualization (warp amplification)
        span = np.array(self.mesh.bounds[1::2]) - np.array(self.mesh.bounds[0::2])
        max_span = float(np.max(span))
        max_disp = float(np.max(mag))
        warp_factor = (0.20 * max_span) / max(max_disp, 1e-6)

        deformed_pts = nodes + disps * warp_factor

        cloud = pv.PolyData(deformed_pts)
        cloud["Disp_Mag_mm"] = mag

        if self.deformed_actor:
            self.plotter.remove_actor(self.deformed_actor)

        self.deformed_actor = self.plotter.add_mesh(
            cloud,
            scalars="Disp_Mag_mm",
            cmap="turbo",
            point_size=12,
            render_points_as_spheres=True,
            scalar_bar_args={"title": "Displacement (mm)", "color": "white"},
            name="deformed_field"
        )
        self.plotter.render()
        print(f"[CAD Labeler] Visualized 3D solution: Max Disp = {max_disp:.6f} mm (Warp scale = {warp_factor:.1f}x).")

    def _on_click_pick(self, pos):
        """Handles 2D mouse click to 3D ray-casting cell pick."""
        if not self.plotter:
            return

        picker = vtk.vtkCellPicker()
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        cell_id = picker.GetCellId()

        if cell_id >= 0 and cell_id < len(self.face_labels):
            face_id = int(self.face_labels[cell_id])
            self.select_face(face_id)

    def launch_interactive(self):
        """Launches PyVista 3D Plotter and the Tkinter inspector panel side-by-side."""
        print("[CAD Labeler] Launching Interactive 3D Session...")
        self.plotter = pv.Plotter(window_size=(1024, 768), title="Immersed IGA - CAD BC Labeler")
        self.plotter.set_background("#181920", top="#252836")
        self.plotter.add_axes()

        # Add base mesh
        self.base_actor = self.plotter.add_mesh(
            self.mesh,
            color="#A8B2C1",
            show_edges=True,
            edge_color="#4F5D75",
            opacity=0.85,
            name="cad_base_mesh"
        )

        # Enable left-click picking via track_click_position
        self.plotter.track_click_position(callback=self._on_click_pick, side="left")

        # Keybindings
        self.plotter.add_key_event("d", lambda: self.assign_boundary_condition(
            f"Dirichlet_Face_{self.selected_face_id}", "Dirichlet", {"components": [True, True, True], "values": [0.0, 0.0, 0.0]}
        ))
        self.plotter.add_key_event("n", lambda: self.assign_boundary_condition(
            f"Neumann_Face_{self.selected_face_id}", "NeumannTraction", {"traction": [0.0, -100.0, 0.0]}
        ))
        self.plotter.add_key_event("c", lambda: self.remove_boundary_condition(
            f"Dirichlet_Face_{self.selected_face_id}"
        ) if f"Dirichlet_Face_{self.selected_face_id}" in self.assigned_bcs else self.remove_boundary_condition(
            f"Neumann_Face_{self.selected_face_id}"
        ))
        self.plotter.add_key_event("e", lambda: self.export_json_configuration())
        self.plotter.add_key_event("s", lambda: self.solve_in_rust())

        self._update_hud()

        # Launch Tkinter UI in parallel if available
        if TK_AVAILABLE:
            def run_tk():
                root = tk.Tk()
                self.tk_app = TkinterInspector(root, self)
                root.mainloop()

            tk_thread = threading.Thread(target=run_tk, daemon=True)
            tk_thread.start()

        self.plotter.show()


class TkinterInspector:
    """Tkinter control panel for interactive property editing, BC tagging, and solver invocation."""

    def __init__(self, root, controller: CadBcLabeler):
        self.root = root
        self.ctrl = controller
        self.root.title("CAD BC Inspector & Solver Controls")
        self.root.geometry("460x720")
        self.root.configure(bg="#22232A")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#22232A", foreground="#E0E6ED", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), background="#3A3F58", foreground="#FFFFFF")
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#00F0FF")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background="#00ADB5", foreground="#FFFFFF")

        self._build_widgets()

    def _build_widgets(self):
        # 1. Model Info
        hdr_frame = tk.Frame(self.root, bg="#22232A", padx=10, pady=5)
        hdr_frame.pack(fill="x")
        ttk.Label(hdr_frame, text="3D CAD MODEL PROPERTIES", style="Header.TLabel").pack(anchor="w")

        info_txt = f"Model: {os.path.basename(str(self.ctrl.cad_path))}\nTriangles: {self.ctrl.mesh.n_cells}  |  CAD Faces: {self.ctrl.num_faces}"
        self.lbl_model_info = ttk.Label(hdr_frame, text=info_txt)
        self.lbl_model_info.pack(anchor="w", pady=2)

        # 2. Selected Face Inspector
        face_frame = tk.LabelFrame(self.root, text=" Selected CAD Face ", bg="#22232A", fg="#FFD700", padx=10, pady=8)
        face_frame.pack(fill="x", padx=10, pady=5)

        self.lbl_face_status = ttk.Label(face_frame, text="Click on any 3D face to select...")
        self.lbl_face_status.pack(anchor="w")

        # Label Name Input
        name_row = tk.Frame(face_frame, bg="#22232A")
        name_row.pack(fill="x", pady=4)
        ttk.Label(name_row, text="Tag Name:").pack(side="left")
        self.ent_face_name = tk.Entry(name_row, bg="#2D3142", fg="#FFFFFF", insertbackground="white")
        self.ent_face_name.insert(0, "ClampFace")
        self.ent_face_name.pack(side="left", fill="x", expand=True, padx=5)

        # 3. BC Assignment
        bc_frame = tk.LabelFrame(self.root, text=" Impose Boundary Condition ", bg="#22232A", fg="#00F0FF", padx=10, pady=8)
        bc_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(bc_frame, text="Condition Type:").pack(anchor="w")
        self.cmb_bc_type = ttk.Combobox(bc_frame, values=["Dirichlet", "NeumannTraction", "NeumannPressure", "RobinFoundation"], state="readonly")
        self.cmb_bc_type.current(0)
        self.cmb_bc_type.pack(fill="x", pady=3)

        # Values row
        val_row = tk.Frame(bc_frame, bg="#22232A")
        val_row.pack(fill="x", pady=4)
        ttk.Label(val_row, text="Values [X, Y, Z]:").pack(side="left")
        self.ent_values = tk.Entry(val_row, bg="#2D3142", fg="#FFFFFF", insertbackground="white")
        self.ent_values.insert(0, "0.0, 0.0, 0.0")
        self.ent_values.pack(side="left", fill="x", expand=True, padx=5)

        btn_apply = tk.Button(bc_frame, text="Apply BC to Selected Face", bg="#00ADB5", fg="white", font=("Segoe UI", 9, "bold"), command=self._on_apply_bc)
        btn_apply.pack(fill="x", pady=4)

        # 4. Assigned BCs Table
        list_frame = tk.LabelFrame(self.root, text=" Imposed Boundary Conditions ", bg="#22232A", fg="#E0E6ED", padx=10, pady=5)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.bc_listbox = tk.Listbox(list_frame, bg="#1E1E24", fg="#E0E6ED", selectbackground="#00ADB5", height=5)
        self.bc_listbox.pack(fill="both", expand=True, side="left")
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.bc_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.bc_listbox.config(yscrollcommand=scrollbar.set)

        btn_rem = tk.Button(self.root, text="Delete Selected BC", bg="#E63946", fg="white", font=("Segoe UI", 8, "bold"), command=self._on_delete_bc)
        btn_rem.pack(fill="x", padx=10, pady=2)

        # 5. Solver & Pipeline Controls
        action_frame = tk.Frame(self.root, bg="#22232A", padx=10, pady=8)
        action_frame.pack(fill="x")

        btn_export = tk.Button(action_frame, text="Export JSON Model", bg="#4F5D75", fg="white", font=("Segoe UI", 9, "bold"), command=self.ctrl.export_json_configuration)
        btn_export.pack(side="left", fill="x", expand=True, padx=2)

        btn_solve = tk.Button(action_frame, text="SOLVE WITH NATIVE RUST", bg="#FFD700", fg="#1E1E24", font=("Segoe UI", 10, "bold"), command=self.ctrl.solve_in_rust)
        btn_solve.pack(side="right", fill="x", expand=True, padx=2)

        # 6. Status Log
        log_frame = tk.LabelFrame(self.root, text=" Solver Output Console ", bg="#22232A", fg="#E0E6ED", padx=10, pady=5)
        log_frame.pack(fill="x", padx=10, pady=5)
        self.txt_log = tk.Text(log_frame, height=5, bg="#181920", fg="#00FF66", font=("Consolas", 8))
        self.txt_log.pack(fill="x")
        self.log("Ready. Select a CAD surface in 3D viewport to tag boundary conditions.")

    def update_inspector(self, face_id, info):
        """Updates inspector labels when user clicks a face."""
        txt = f"Face #{face_id} | Triangles: {info['cell_count']} | Area: {info['area']:.3f}\n" \
              f"Normal: [{info['normal'][0]:.2f}, {info['normal'][1]:.2f}, {info['normal'][2]:.2f}]\n" \
              f"Centroid: [{info['centroid'][0]:.1f}, {info['centroid'][1]:.1f}, {info['centroid'][2]:.1f}]"
        self.lbl_face_status.config(text=txt)
        self.ent_face_name.delete(0, tk.END)
        self.ent_face_name.insert(0, f"Face_{face_id}")

    def refresh_bc_list(self):
        """Refreshes the active BC listbox."""
        self.bc_listbox.delete(0, tk.END)
        for name, bc in self.ctrl.assigned_bcs.items():
            self.bc_listbox.insert(tk.END, f"[{bc['type']}] {name} (Face #{bc['face_id']}, Area={bc['area']:.2f})")

    def _on_apply_bc(self):
        name = self.ent_face_name.get().strip()
        bc_type = self.cmb_bc_type.get()
        val_str = self.ent_values.get().strip()

        if not name:
            messagebox.showwarning("Warning", "Please provide a valid face tag name.")
            return

        params = {}
        try:
            if bc_type == "Dirichlet":
                vals = [float(x.strip()) for x in val_str.split(",")]
                params["components"] = [True, True, True]
                params["values"] = vals if len(vals) >= 3 else [0.0, 0.0, 0.0]
            elif bc_type == "NeumannTraction":
                vals = [float(x.strip()) for x in val_str.split(",")]
                params["traction"] = vals if len(vals) >= 3 else [0.0, -100.0, 0.0]
            elif bc_type == "NeumannPressure":
                params["pressure"] = float(val_str)
            elif bc_type == "RobinFoundation":
                vals = [float(x.strip()) for x in val_str.split(",")]
                params["normal_stiffness"] = vals[0]
                params["tangential_stiffness"] = vals[1] if len(vals) > 1 else vals[0]
        except Exception as e:
            messagebox.showerror("Format Error", f"Failed to parse values '{val_str}': {e}")
            return

        ok = self.ctrl.assign_boundary_condition(name, bc_type, params)
        if ok:
            self.log(f"Imposed {bc_type} BC on '{name}'.")

    def _on_delete_bc(self):
        sel = self.bc_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        name = list(self.ctrl.assigned_bcs.keys())[idx]
        self.ctrl.remove_boundary_condition(name)
        self.log(f"Deleted BC '{name}'.")

    def display_solution_stats(self, sol):
        msg = f"PCG Converged in {sol.get('pcg_iters', 0)} iters | Time: {sol.get('wall_time_ms', 0):.2f} ms\n" \
              f"Peak Disp: {sol.get('peak_disp_mm', 0):.6f} mm | DOFs: {sol.get('n_master_dofs', 0)}"
        self.log(msg)

    def log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)


def run_automated_test():
    """Automated verification mode: executes face picking, BC assignment, JSON export, and Rust PCG solve."""
    print("========================================================================")
    print("  RUNNING AUTOMATED CAD LABELER & SOLVER TEST SUITE                     ")
    print("========================================================================")

    # 1. Initialize labeler on NIST CAD model
    labeler = CadBcLabeler(cad_path="Models/nist_ctc_01.obj", angle_deg=20.0, off_screen=True)
    assert labeler.num_faces > 0, "No CAD faces detected!"
    print(f"[TEST 1/5] Mesh loaded and segmented into {labeler.num_faces} faces.")

    # 2. Programmatically select clamp face (e.g. face 0 or minimum X)
    min_x_face = min(labeler.face_info.items(), key=lambda kv: kv[1]["centroid"][0])[0]
    labeler.select_face(min_x_face)
    labeler.assign_boundary_condition("FixedClamp", "Dirichlet", {"components": [True, True, True], "values": [0.0, 0.0, 0.0]})
    assert "FixedClamp" in labeler.assigned_bcs
    print(f"[TEST 2/5] Successfully tagged Dirichlet clamp on Face #{min_x_face}.")

    # 3. Select load face (e.g. maximum X)
    max_x_face = max(labeler.face_info.items(), key=lambda kv: kv[1]["centroid"][0])[0]
    labeler.select_face(max_x_face)
    labeler.assign_boundary_condition("LoadFace", "NeumannTraction", {"traction": [0.0, -100.0, 0.0]})
    assert "LoadFace" in labeler.assigned_bcs
    print(f"[TEST 3/5] Successfully tagged Neumann traction on Face #{max_x_face}.")

    # 4. Export JSON configuration
    json_path = labeler.export_json_configuration("benchmarks/cad_model_input.json")
    assert os.path.exists(json_path), "JSON config file was not created!"
    print(f"[TEST 4/5] Successfully exported JSON input to '{json_path}'.")

    # 5. Execute Rust solver
    success = labeler.solve_in_rust()
    assert success, "Native Rust solve_cad execution failed!"
    assert labeler.solution_data is not None, "No solution data retrieved!"
    peak_disp = labeler.solution_data.get("peak_disp_mm", 0.0)
    assert peak_disp > 0.0, f"Expected non-zero displacement, got {peak_disp}"
    print(f"[TEST 5/5] Rust Solver succeeded! Peak Displacement = {peak_disp:.6f} mm, Time = {labeler.solution_data.get('wall_time_ms', 0):.2f} ms")

    print("========================================================================")
    print("  ALL CAD LABELER AUTOMATED TESTS PASSED (100% OK)                      ")
    print("========================================================================")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--test", "-t", "--headless"):
        run_automated_test()
    else:
        cad_arg = sys.argv[1] if len(sys.argv) > 1 else None
        app = CadBcLabeler(cad_path=cad_arg)
        app.launch_interactive()
