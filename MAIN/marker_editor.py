#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Marker Placement Tool for SMPL Body Model

This script allows you to:
1. Visualize the SMPL body model in 3D using psbody.mesh
2. Click on the mesh to place markers at specific vertices
3. Name and edit markers
4. Save marker definitions to an NPZ file

Requirements:
    - psbody.mesh (required for visualization)
    - body_visualizer (for mesh colors)

Usage:
    python marker_editor.py [--model MODEL_PATH] [--output OUTPUT_FILE]

Commands (interactive mode):
    - add(x, y, z)          - Add marker at position (x, y, z)
    - add_vertex(idx)       - Add marker at vertex index
    - rename('old', 'new')  - Rename marker
    - delete('name')        - Delete marker
    - save()                - Save markers to file
    - list()                - List all markers
    - quit()                - Exit editor
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "glx"

import sys
import argparse
import numpy as np
import torch
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from human_body_prior.body_model.body_model import BodyModel

# Import visualization tools - psbody mesh is required
try:
    from psbody.mesh import Mesh, MeshViewer
    from body_visualizer.tools.vis_tools import colors
    PSBODY_AVAILABLE = True
except ImportError:
    print("Error: psbody.mesh is required for this script.")
    print("Please install it to use the marker editor.")
    PSBODY_AVAILABLE = False
    sys.exit(1)


class MarkerEditor:
    """Interactive tool for placing and naming markers on SMPL body model."""
    
    def __init__(self, model_path: str, output_file: str = "markers.npz"):
        """
        Initialize the marker editor.
        
        Args:
            model_path: Path to SMPL model file (.npz)
            output_file: Path to save marker definitions
        """
        self.model_path = model_path
        self.output_file = output_file
        self.markers = {}  # {name: vertex_index}
        self.marker_positions = {}  # {name: (x, y, z)}
        self.temp_marker_count = 0
        
        # Initialize body model
        print(f"Loading body model from {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bm = BodyModel(model_path, num_betas=10, num_dmpls=None).to(self.device)
        
        # Generate default body mesh
        with torch.no_grad():
            body_output = self.bm(
                betas=torch.zeros(1, 10).to(self.device),
                pose_body=torch.zeros(1, 63).to(self.device),
                root_orient=torch.zeros(1, 3).to(self.device),
                trans=torch.zeros(1, 3).to(self.device)
            )
        
        self.vertices = body_output.v.cpu().numpy()[0]
        self.faces = self.bm.f.cpu().numpy() if hasattr(self.bm, 'f') else self.bm.faces
        
        print(f"Body model loaded: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} faces")
        print(f"Device: {self.device}")
    
    def find_closest_vertex(self, point: np.ndarray) -> int:
        """
        Find the closest vertex to a given 3D point.
        
        Args:
            point: 3D point (x, y, z)
        
        Returns:
            Index of the closest vertex
        """
        distances = np.linalg.norm(self.vertices - point, axis=1)
        return int(np.argmin(distances))
    
    def add_marker(self, vertex_idx: int, name: Optional[str] = None) -> str:
        """
        Add a marker at a specific vertex.
        
        Args:
            vertex_idx: Index of the vertex
            name: Optional name for the marker
        
        Returns:
            Name of the added marker
        """
        if name is None:
            name = f"marker_{self.temp_marker_count:03d}"
            self.temp_marker_count += 1
        
        self.markers[name] = vertex_idx
        self.marker_positions[name] = self.vertices[vertex_idx].copy()
        
        print(f"Added marker '{name}' at vertex {vertex_idx}: {self.marker_positions[name]}")
        return name
    
    def rename_marker(self, old_name: str, new_name: str):
        """Rename a marker."""
        if old_name in self.markers:
            self.markers[new_name] = self.markers.pop(old_name)
            self.marker_positions[new_name] = self.marker_positions.pop(old_name)
            print(f"Renamed marker '{old_name}' -> '{new_name}'")
        else:
            print(f"Warning: Marker '{old_name}' not found")
    
    def delete_marker(self, name: str):
        """Delete a marker by name."""
        if name in self.markers:
            self.markers.pop(name)
            self.marker_positions.pop(name)
            print(f"Deleted marker '{name}'")
        else:
            print(f"Warning: Marker '{name}' not found")
    
    def save_markers(self, filepath: Optional[str] = None):
        """
        Save marker definitions to NPZ file.
        
        Args:
            filepath: Optional custom save path
        """
        if filepath is None:
            filepath = self.output_file
        
        if not self.markers:
            print("No markers to save!")
            return
        
        # Prepare data
        marker_names = list(self.markers.keys())
        marker_indices = np.array([self.markers[name] for name in marker_names], dtype=np.int32)
        marker_coords = np.array([self.marker_positions[name] for name in marker_names], dtype=np.float32)
        
        # Save to NPZ
        np.savez(
            filepath,
            marker_names=marker_names,
            marker_indices=marker_indices,
            marker_positions=marker_coords,
            model_path=self.model_path
        )
        
        print(f"\nSaved {len(self.markers)} markers to {filepath}")
        print("Markers:")
        for name in marker_names:
            idx = self.markers[name]
            pos = self.marker_positions[name]
            print(f"  {name}: vertex {idx} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    def load_markers(self, filepath: str):
        """
        Load marker definitions from NPZ file.
        
        Args:
            filepath: Path to NPZ file
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return
        
        data = np.load(filepath, allow_pickle=True)
        
        marker_names = data['marker_names'].tolist()
        marker_indices = data['marker_indices']
        
        self.markers = {}
        self.marker_positions = {}
        
        for name, idx in zip(marker_names, marker_indices):
            self.markers[name] = int(idx)
            self.marker_positions[name] = self.vertices[idx].copy()
        
        self.temp_marker_count = len(self.markers)
        
        print(f"Loaded {len(self.markers)} markers from {filepath}")
    
    def run_psbody_interactive(self):
        """Run interactive editor using psbody.mesh."""
        print("\n" + "="*70)
        print("Interactive Marker Editor")
        print("="*70)
        print("Commands (type in console):")
        print("  add(x, y, z)          - Add marker at position (x, y, z)")
        print("  add_vertex(idx)       - Add marker at vertex index")
        print("  rename('old', 'new')  - Rename marker")
        print("  delete('name')        - Delete marker")
        print("  save()                - Save markers to file")
        print("  list()                - List all markers")
        print("  quit()                - Exit editor")
        print("="*70)
        
        # Create mesh viewer
        mv = MeshViewer(keepalive=False)
        
        # Create body mesh
        body_mesh = Mesh(v=self.vertices, f=self.faces, vc=colors['grey'])
        mv.set_static_meshes([body_mesh])
        
        # Helper functions for interactive use
        def add(x: float, y: float, z: float, name: Optional[str] = None):
            """Add marker at 3D position."""
            point = np.array([x, y, z])
            vertex_idx = self.find_closest_vertex(point)
            marker_name = self.add_marker(vertex_idx, name)
            self._update_viewer(mv)
            return marker_name
        
        def add_vertex(idx: int, name: Optional[str] = None):
            """Add marker at vertex index."""
            marker_name = self.add_marker(idx, name)
            self._update_viewer(mv)
            return marker_name
        
        def rename(old_name: str, new_name: str):
            """Rename a marker."""
            self.rename_marker(old_name, new_name)
            self._update_viewer(mv)
        
        def delete(name: str):
            """Delete a marker."""
            self.delete_marker(name)
            self._update_viewer(mv)
        
        def save(filepath: Optional[str] = None):
            """Save markers to file."""
            self.save_markers(filepath)
        
        def list_markers():
            """List all markers."""
            if not self.markers:
                print("No markers defined.")
                return
            print("\nCurrent markers:")
            for name, idx in self.markers.items():
                pos = self.marker_positions[name]
                print(f"  {name}: vertex {idx} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            return [name for name in self.markers]
        
        def clean_default_names():
            """Remove markers with default names."""
            to_delete = [name for name in self.markers if name.startswith("marker_")]
            for name in to_delete:
                self.delete_marker(name)
            self._update_viewer(mv)
            print(f"Deleted {len(to_delete)} default-named markers.")
        
        def quit_editor():
            """Exit the editor."""
            import sys
            sys.exit(0)
        
        # # Make functions available
        # import __main__
        # __main__.add = add
        # __main__.add_vertex = add_vertex
        # __main__.rename = rename
        # __main__.delete = delete
        # __main__.save = save
        # __main__.list = list_markers
        # __main__.quit = quit_editor
        
        # Initial viewer update
        self._update_viewer(mv)
        
        print("\nViewer opened. Use the functions above in the console.")
        print("The viewer will stay open. Close the window or type quit() to exit.")
        
        # Keep the viewer open
        import code
        code.interact(local=locals(), banner="")
    
    def _update_viewer(self, mv):
        """Update the mesh viewer with current markers."""
        from psbody.mesh.sphere import Sphere
        
        # Create marker spheres
        marker_meshes = []
        for name, idx in self.markers.items():
            pos = self.marker_positions[name]
            sphere = Sphere(center=pos, radius=0.02).to_mesh()
            sphere.vc = np.array([1.0, 0.0, 0.0])  # Red markers
            marker_meshes.append(sphere)
        
        # Update dynamic meshes
        if marker_meshes:
            mv.set_dynamic_meshes(marker_meshes)
        
        # Update title
        mv.set_titlebar(f"Marker Editor - {len(self.markers)} markers")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive Marker Placement Tool")
    parser.add_argument(
        '--model',
        type=str,
        default='../support_data/dowloads/models/smplx/neutral/model.npz',
        help='Path to SMPL model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='markers.npz',
        help='Output file for marker definitions'
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Load existing markers from file'
    )
    
    args = parser.parse_args()
    
    # Create editor
    editor = MarkerEditor(args.model, args.output)
    
    # Load existing markers if specified
    if args.load:
        editor.load_markers(args.load)
    
    # Run psbody interactive mode
    editor.run_psbody_interactive()


if __name__ == '__main__':
    main()
