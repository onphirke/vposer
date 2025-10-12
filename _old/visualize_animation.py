#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animation Renderer for SMPL Body Model

This script renders SMPL body model animations frame-by-frame from IK results.
Supports multiple output formats: video (MP4), GIF, or image sequences.

Usage:
    # Render to video
    python visualize_animation.py --input results.npz --output animation.mp4
    
    # Render to GIF
    python visualize_animation.py --input results.npz --output animation.gif
    
    # Render to image sequence
    python visualize_animation.py --input results.npz --output frames/ --format png
    
    # Custom settings
    python visualize_animation.py --input results.npz --output anim.mp4 \
        --fps 30 --resolution 1920x1080 --view-angle front
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "glx"

import sys
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Try to import rendering tools
try:
    import trimesh
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available. Using matplotlib fallback.")

try:
    from psbody.mesh import Mesh
    PSBODY_AVAILABLE = True
except ImportError:
    PSBODY_AVAILABLE = False


class BodyAnimationRenderer:
    """Renderer for SMPL body model animations."""
    
    def __init__(
        self,
        body_model_path: str,
        resolution: Tuple[int, int] = (800, 600),
        device: Optional[torch.device] = None
    ):
        """
        Initialize the renderer.
        
        Args:
            body_model_path: Path to SMPL body model file
            resolution: Output resolution (width, height)
            device: Torch device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.resolution = resolution
        
        # Load body model
        print(f"Loading body model from {body_model_path}")
        self.bm = BodyModel(body_model_path, num_betas=10, num_dmpls=None).to(device)
        self.faces = self.bm.f.cpu().numpy() if hasattr(self.bm, 'f') else self.bm.faces
        
        print(f"Renderer initialized: {resolution[0]}x{resolution[1]}")
    
    def generate_meshes(
        self,
        body_params: Dict[str, np.ndarray],
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate mesh vertices for each frame.
        
        Args:
            body_params: Dictionary of body parameters
            start_frame: Start frame index
            end_frame: End frame index (None = all frames)
        
        Returns:
            List of vertex arrays, one per frame
        """
        n_frames = body_params['betas'].shape[0]
        if end_frame is None:
            end_frame = n_frames
        
        meshes = []
        
        print(f"Generating meshes for frames {start_frame} to {end_frame-1}")
        
        with torch.no_grad():
            for frame_idx in tqdm(range(start_frame, end_frame), desc="Generating meshes"):
                body_output = self.bm(
                    betas=torch.tensor(body_params['betas'][frame_idx:frame_idx+1], device=self.device),
                    pose_body=torch.tensor(body_params['pose_body'][frame_idx:frame_idx+1], device=self.device),
                    root_orient=torch.tensor(body_params['root_orient'][frame_idx:frame_idx+1], device=self.device),
                    trans=torch.tensor(body_params['trans'][frame_idx:frame_idx+1], device=self.device)
                )
                
                vertices = body_output.v.cpu().numpy()[0]
                meshes.append(vertices)
        
        return meshes
    
    def render_frame_matplotlib(
        self,
        vertices: np.ndarray,
        view_angle: str = 'front',
        show_joints: bool = False,
        joint_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render a single frame using matplotlib (3D wireframe).
        
        Args:
            vertices: Mesh vertices (V, 3)
            view_angle: Camera view angle ('front', 'side', 'top', '3d')
            show_joints: Whether to show joint positions
            joint_positions: Joint positions if show_joints=True
        
        Returns:
            Rendered image as numpy array (H, W, 3)
        """
        fig = plt.figure(figsize=(self.resolution[0]/100, self.resolution[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=self.faces,
            color='lightgray',
            edgecolor='none',
            alpha=0.8,
            shade=True
        )
        
        # Plot joints if requested
        if show_joints and joint_positions is not None:
            ax.scatter(
                joint_positions[:, 0],
                joint_positions[:, 1],
                joint_positions[:, 2],
                c='red', s=20, alpha=0.8
            )
        
        # Set view angle
        if view_angle == 'front':
            ax.view_init(elev=0, azim=0)
        elif view_angle == 'side':
            ax.view_init(elev=0, azim=90)
        elif view_angle == 'top':
            ax.view_init(elev=90, azim=0)
        elif view_angle == '3d':
            ax.view_init(elev=20, azim=45)
        
        # Set axis properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = np.array([
            vertices[:, 0].max()-vertices[:, 0].min(),
            vertices[:, 1].max()-vertices[:, 1].min(),
            vertices[:, 2].max()-vertices[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img
    
    def render_frame_pyrender(
        self,
        vertices: np.ndarray,
        view_angle: str = 'front'
    ) -> np.ndarray:
        """
        Render a single frame using pyrender (photorealistic).
        
        Args:
            vertices: Mesh vertices (V, 3)
            view_angle: Camera view angle
        
        Returns:
            Rendered image as numpy array (H, W, 3)
        """
        if not PYRENDER_AVAILABLE:
            raise ImportError("pyrender not available")
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        
        # Create pyrender mesh
        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        
        # Create scene
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(mesh_pr)
        
        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))
        
        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        
        # Camera position based on view angle
        if view_angle == 'front':
            camera_pose = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 3],
                [0, 0, 0, 1]
            ])
        elif view_angle == 'side':
            camera_pose = np.array([
                [0, 0, 1, 3],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]
            ])
        else:  # 3d view
            camera_pose = np.array([
                [0.707, 0, 0.707, 2],
                [0, 1, 0, 0],
                [-0.707, 0, 0.707, 2],
                [0, 0, 0, 1]
            ])
        
        scene.add(camera, pose=camera_pose)
        
        # Render
        renderer = pyrender.OffscreenRenderer(self.resolution[0], self.resolution[1])
        color, depth = renderer.render(scene)
        renderer.delete()
        
        return color
    
    def save_frames(
        self,
        meshes: List[np.ndarray],
        output_dir: str,
        format: str = 'png',
        view_angle: str = 'front',
        use_pyrender: bool = False
    ):
        """
        Save animation frames as individual images.
        
        Args:
            meshes: List of mesh vertices
            output_dir: Output directory for frames
            format: Image format ('png', 'jpg')
            view_angle: Camera view angle
            use_pyrender: Use pyrender instead of matplotlib
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {len(meshes)} frames to {output_dir}")
        
        for i, vertices in enumerate(tqdm(meshes, desc="Rendering frames")):
            if use_pyrender and PYRENDER_AVAILABLE:
                img = self.render_frame_pyrender(vertices, view_angle)
            else:
                img = self.render_frame_matplotlib(vertices, view_angle)
            
            # Save image
            output_path = os.path.join(output_dir, f"frame_{i:04d}.{format}")
            plt.imsave(output_path, img)
        
        print(f"Saved frames to {output_dir}")
    
    def create_video(
        self,
        meshes: List[np.ndarray],
        output_file: str,
        fps: int = 30,
        view_angle: str = 'front',
        use_pyrender: bool = False
    ):
        """
        Create video from mesh sequence.
        
        Args:
            meshes: List of mesh vertices
            output_file: Output video file path
            fps: Frames per second
            view_angle: Camera view angle
            use_pyrender: Use pyrender instead of matplotlib
        """
        print(f"Creating video: {output_file}")
        print(f"  Frames: {len(meshes)}")
        print(f"  FPS: {fps}")
        
        # Render all frames
        frames = []
        meshes = meshes[::25]
        for vertices in tqdm(meshes, desc="Rendering frames"):
            if use_pyrender and PYRENDER_AVAILABLE:
                img = self.render_frame_pyrender(vertices, view_angle)
            else:
                img = self.render_frame_matplotlib(vertices, view_angle)
            frames.append(img)
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(self.resolution[0]/100, self.resolution[1]/100), dpi=100)
        ax.axis('off')
        
        im = ax.imshow(frames[0])
        
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            return [im]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
        
        # Save to file
        if output_file.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, bitrate=5000)
        
        anim.save(output_file, writer=writer)
        plt.close(fig)
        
        print(f"Saved video to {output_file}")
    
    def create_gif(
        self,
        meshes: List[np.ndarray],
        output_file: str,
        fps: int = 10,
        view_angle: str = 'front'
    ):
        """
        Create GIF animation from mesh sequence.
        
        Args:
            meshes: List of mesh vertices
            output_file: Output GIF file path
            fps: Frames per second
            view_angle: Camera view angle
        """
        self.create_video(meshes, output_file, fps, view_angle, use_pyrender=False)


def load_ik_results(results_file: str) -> Dict[str, np.ndarray]:
    """
    Load IK results from NPZ file.
    
    Args:
        results_file: Path to NPZ results file
    
    Returns:
        Dictionary of body parameters
    """
    print(f"Loading results from {results_file}")
    data = np.load(results_file, allow_pickle=True)
    
    # Handle different possible structures
    if 'body_params' in data:
        body_params = data['body_params'].item()
    else:
        # Assume direct format
        body_params = {
            'betas': data['betas'],
            'pose_body': data['pose_body'],
            'root_orient': data['root_orient'],
            'trans': data['trans']
        }
    
    n_frames = body_params['betas'].shape[0]
    print(f"Loaded {n_frames} frames")
    
    return body_params


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SMPL Body Animation Renderer")
    
    # I/O arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Input NPZ file with IK results')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file/directory')
    parser.add_argument('--body-model', type=str,
                        default='../support_data/dowloads/models/smplx/neutral/model.npz',
                        help='Path to SMPL body model')
    
    # Rendering arguments
    parser.add_argument('--format', type=str, choices=['video', 'gif', 'png', 'jpg'],
                        default='video',
                        help='Output format')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for video/gif')
    parser.add_argument('--resolution', type=str, default='800x600',
                        help='Output resolution (WxH)')
    parser.add_argument('--view-angle', type=str,
                        choices=['front', 'side', 'top', '3d'],
                        default='front',
                        help='Camera view angle')
    parser.add_argument('--use-pyrender', action='store_true',
                        help='Use pyrender for photorealistic rendering')
    
    # Frame selection
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Start frame')
    parser.add_argument('--end-frame', type=int, default=None,
                        help='End frame (None = all)')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Skip every N frames (for faster rendering)')
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Load results
    body_params = load_ik_results(args.input)
    
    # Apply frame selection
    if args.end_frame is None:
        args.end_frame = body_params['betas'].shape[0]
    
    if args.skip_frames > 1:
        for key in body_params:
            body_params[key] = body_params[key][::args.skip_frames]
    
    # Initialize renderer
    renderer = BodyAnimationRenderer(args.body_model, resolution=(width, height))
    
    # Generate meshes
    meshes = renderer.generate_meshes(body_params, args.start_frame, args.end_frame)
    
    # Render based on format
    if args.format in ['png', 'jpg']:
        renderer.save_frames(
            meshes,
            args.output,
            format=args.format,
            view_angle=args.view_angle,
            use_pyrender=args.use_pyrender
        )
    elif args.format == 'gif':
        renderer.create_gif(
            meshes,
            args.output,
            fps=args.fps,
            view_angle=args.view_angle
        )
    else:  # video
        renderer.create_video(
            meshes,
            args.output,
            fps=args.fps,
            view_angle=args.view_angle,
            use_pyrender=args.use_pyrender
        )
    
    print("\nRendering complete!")


if __name__ == '__main__':
    main()
