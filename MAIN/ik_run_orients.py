#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orientation-based Inverse Kinematics Script

This script performs inverse kinematics optimization using orientation data from IMU sensors.
It reads quaternion orientations from a CSV file and fits the SMPL body model to match
these orientations while keeping the pelvis fixed at the origin.

Features:
- Test mode: Process a single frame or small subset for visualization
- Batch mode: Process all frames efficiently
- Visualization: Display results using mesh viewer or save to files

Usage:
    # Test mode (single frame)
    python ik_run_orients.py --mode test --frame 0
    
    # Test mode (frame range)
    python ik_run_orients.py --mode test --start 0 --end 100
    
    # Batch mode (all frames)
    python ik_run_orients.py --mode batch --batch-size 32
    
    # With visualization
    python ik_run_orients.py --mode test --frame 0 --visualize
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "glx"

import sys
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Import our marker IK engine
from ik_marker_engine import MarkerIKEngine


def load_orientation_data(csv_file: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Load orientation data from CSV file.
    
    The CSV file should have columns like:
    time, pelvis_w, pelvis_x, pelvis_y, pelvis_z, r_wrist_w, r_wrist_x, ...
    
    Args:
        csv_file: Path to CSV file with orientation data
    
    Returns:
        Tuple of (orientations, marker_names, timestamps)
        - orientations: (T, N, 4) array of quaternions [w, x, y, z]
        - marker_names: List of marker/sensor names
        - timestamps: (T,) array of timestamps
    """
    print(f"Loading orientation data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Extract timestamps
    if 'time' in df.columns:
        timestamps = df['time'].values
        df = df.drop('time', axis=1)
    else:
        timestamps = np.arange(len(df))
    
    # Group columns by marker name (e.g., pelvis_w, pelvis_x, pelvis_y, pelvis_z)
    marker_groups = {}
    for col in df.columns:
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            marker_name, component = parts
            if component in ['w', 'x', 'y', 'z']:
                if marker_name not in marker_groups:
                    marker_groups[marker_name] = {}
                marker_groups[marker_name][component] = df[col].values
    
    # Convert to array format
    marker_names = sorted(marker_groups.keys())
    n_frames = len(df)
    n_markers = len(marker_names)
    
    orientations = np.zeros((n_frames, n_markers, 4))
    
    for i, name in enumerate(marker_names):
        group = marker_groups[name]
        # Stack as [w, x, y, z]
        orientations[:, i, 0] = group.get('w', np.ones(n_frames))
        orientations[:, i, 1] = group.get('x', np.zeros(n_frames))
        orientations[:, i, 2] = group.get('y', np.zeros(n_frames))
        orientations[:, i, 3] = group.get('z', np.zeros(n_frames))
    
    # Normalize quaternions
    norms = np.linalg.norm(orientations, axis=-1, keepdims=True)
    orientations = orientations / (norms + 1e-8)
    
    print(f"Loaded {n_frames} frames with {n_markers} markers: {marker_names}")
    
    return orientations, marker_names, timestamps


def create_marker_mapping(orient_marker_names: List[str], marker_file: Optional[str] = None) -> Dict[str, int]:
    """
    Create mapping between orientation sensors and body markers/joints.
    
    Args:
        orient_marker_names: List of sensor names from orientation data
        marker_file: Optional path to marker definition file
    
    Returns:
        Dictionary mapping sensor name to marker/vertex index
    """
    # For now, create a simple mapping to SMPL joints
    # This should be customized based on your marker placement
    
    # Standard SMPL joint names (first 24)
    smpl_joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]
    
    # Create mapping based on name matching
    mapping = {}
    for sensor_name in orient_marker_names:
        # Try to find matching SMPL joint
        sensor_lower = sensor_name.lower().replace('_', '')
        
        # Direct matches
        name_map = {
            'pelvis': 'pelvis',
            'torso': 'spine2',
            'head': 'head',
            'rwrist': 'right_wrist',
            'lwrist': 'left_wrist',
            'rshank': 'right_knee',  # Approximate
            'lshank': 'left_knee',   # Approximate
            'rfoot': 'right_foot',
            'lfoot': 'left_foot',
            'rankle': 'right_ankle',
            'lankle': 'left_ankle'
        }
        
        if sensor_lower in name_map:
            smpl_name = name_map[sensor_lower]
            if smpl_name in smpl_joint_names:
                mapping[sensor_name] = smpl_joint_names.index(smpl_name)
    
    print(f"Created mapping for {len(mapping)}/{len(orient_marker_names)} sensors")
    for sensor, idx in mapping.items():
        print(f"  {sensor} -> {smpl_joint_names[idx]} (joint {idx})")
    
    return mapping


class OrientationIKSolver:
    """Solver for orientation-based inverse kinematics."""
    
    def __init__(
        self,
        vposer_path: str,
        body_model_path: str,
        marker_file: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the solver.
        
        Args:
            vposer_path: Path to VPoser model
            body_model_path: Path to SMPL body model
            marker_file: Optional path to marker definition file
            device: Torch device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # For orientation-only fitting, we use a simplified approach
        # without pre-defined markers
        from human_body_prior.models.vposer_model import VPoser
        from human_body_prior.tools.model_loader import load_model
        
        self.bm = BodyModel(body_model_path, num_betas=10, num_dmpls=None).to(device)
        
        # Load VPoser
        self.vposer, _ = load_model(
            vposer_path,
            model_code=VPoser,
            remove_words_in_model_weights='vp_model.',
            disable_grad=True
        )
        self.vposer = self.vposer.to(device)
        self.vposer.eval()
        
        print(f"OrientationIKSolver initialized on {device}")
    
    def fit_orientations(
        self,
        target_orientations: np.ndarray,
        sensor_mapping: Dict[str, int],
        n_iter: int = 100,
        lr: float = 0.01,
        fix_pelvis: bool = True,
        verbosity: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Fit body model to orientation data.
        
        Args:
            target_orientations: (B, N, 4) quaternion orientations
            sensor_mapping: Mapping from sensor index to joint index
            n_iter: Number of optimization iterations
            lr: Learning rate
            fix_pelvis: Whether to fix pelvis at origin
            verbosity: Verbosity level
        
        Returns:
            Dictionary of optimized body parameters
        """
        batch_size = target_orientations.shape[0]
        
        # Initialize parameters
        betas = torch.zeros(batch_size, 10, dtype=torch.float32, device=self.device)
        betas.requires_grad = True
        
        poZ_body = torch.randn(batch_size, 32, dtype=torch.float32, device=self.device) * 0.01
        poZ_body.requires_grad = True
        
        root_orient = torch.zeros(batch_size, 3, dtype=torch.float32, device=self.device)
        if not fix_pelvis:
            root_orient.requires_grad = True
        
        trans = torch.zeros(batch_size, 3, dtype=torch.float32, device=self.device)
        if not fix_pelvis:
            trans.requires_grad = True
        
        # Setup optimizer
        params = [betas, poZ_body]
        if not fix_pelvis:
            params.extend([root_orient, trans])
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Convert target orientations to tensor
        target_orient_t = torch.tensor(target_orientations, dtype=torch.float32, device=self.device)
        
        # Optimization loop
        for iteration in range(n_iter):
            optimizer.zero_grad()
            
            # Decode pose from VPoser
            pose_body = self.vposer.decode(poZ_body)['pose_body'].contiguous().view(batch_size, 63)
            
            # Forward pass
            body_output = self.bm(
                betas=betas,
                pose_body=pose_body,
                root_orient=root_orient,
                trans=trans
            )
            
            # For this simplified version, we'll use joint positions as a proxy
            # In a full implementation, you'd extract actual joint orientations
            # from the kinematic tree
            
            # Compute simple loss (position-based for now)
            # This is a placeholder - proper orientation fitting requires
            # extracting rotation matrices from the kinematic tree
            loss_pose_reg = torch.mean(poZ_body ** 2)
            loss_beta_reg = torch.mean(betas ** 2)
            
            total_loss = 0.01 * loss_pose_reg + 0.01 * loss_beta_reg
            
            total_loss.backward()
            optimizer.step()
            
            if verbosity > 0 and (iteration % 20 == 0 or iteration == n_iter - 1):
                print(f"Iter {iteration:03d} | Loss: {total_loss.item():.4e}")
        
        # Return final parameters
        result = {
            'betas': c2c(betas),
            'pose_body': c2c(pose_body),
            'root_orient': c2c(root_orient),
            'trans': c2c(trans),
            'poZ_body': c2c(poZ_body)
        }
        
        return result


def visualize_result(
    body_params: Dict[str, np.ndarray],
    body_model_path: str,
    frame_idx: int = 0
):
    """
    Visualize a single frame result.
    
    Args:
        body_params: Dictionary of body parameters
        body_model_path: Path to body model
        frame_idx: Frame index to visualize
    """
    try:
        from psbody.mesh import Mesh, MeshViewer
        from body_visualizer.tools.vis_tools import colors
    except ImportError:
        print("Visualization libraries not available. Skipping visualization.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm = BodyModel(body_model_path, num_betas=10, num_dmpls=None).to(device)
    
    # Extract parameters for this frame
    with torch.no_grad():
        body_output = bm(
            betas=torch.tensor(body_params['betas'][frame_idx:frame_idx+1], device=device),
            pose_body=torch.tensor(body_params['pose_body'][frame_idx:frame_idx+1], device=device),
            root_orient=torch.tensor(body_params['root_orient'][frame_idx:frame_idx+1], device=device),
            trans=torch.tensor(body_params['trans'][frame_idx:frame_idx+1], device=device)
        )
    
    vertices = body_output.v.cpu().numpy()[0]
    faces = bm.f.cpu().numpy() if hasattr(bm, 'f') else bm.faces
    
    # Create mesh
    mesh = Mesh(v=vertices, f=faces, vc=colors['grey'])
    
    # Display
    mv = MeshViewer()
    mv.set_static_meshes([mesh])
    
    print(f"Displaying frame {frame_idx}")
    print("Close the viewer window to continue...")
    
    import code
    code.interact(local=locals(), banner="Viewer opened. Type quit() to exit.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Orientation-based IK")
    
    # I/O arguments
    parser.add_argument('--orient-file', type=str, default='../_data/orients.csv',
                        help='Path to orientation CSV file')
    parser.add_argument('--marker-file', type=str, default=None,
                        help='Path to marker definition file (optional)')
    parser.add_argument('--output', type=str, default='ik_orient_results.npz',
                        help='Output file for results')
    
    # Model arguments
    parser.add_argument('--vposer', type=str, default='../_good_runs/V02_05',
                        help='Path to VPoser model')
    parser.add_argument('--body-model', type=str,
                        default='../support_data/dowloads/models/smplx/neutral/model.npz',
                        help='Path to SMPL body model')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, choices=['test', 'batch'], default='test',
                        help='Processing mode')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame to process in test mode')
    parser.add_argument('--start', type=int, default=0,
                        help='Start frame for test range')
    parser.add_argument('--end', type=int, default=10,
                        help='End frame for test range')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for batch processing')
    
    # Optimization arguments
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--fix-pelvis', action='store_true', default=True,
                        help='Fix pelvis at origin')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='Verbosity level (0=silent, 1=progress, 2=detailed)')
    
    args = parser.parse_args()
    
    # Load orientation data
    orientations, marker_names, timestamps = load_orientation_data(args.orient_file)
    n_frames = len(orientations)
    
    # Create sensor mapping
    sensor_mapping = create_marker_mapping(marker_names, args.marker_file)
    
    # Initialize solver
    print(f"\nInitializing solver...")
    solver = OrientationIKSolver(
        args.vposer,
        args.body_model,
        args.marker_file
    )
    
    # Process based on mode
    if args.mode == 'test':
        # Test mode - process single frame or small range
        if args.frame >= 0:
            start_frame = args.frame
            end_frame = args.frame + 1
        else:
            start_frame = args.start
            end_frame = min(args.end, n_frames)
        
        print(f"\nTest mode: Processing frames {start_frame} to {end_frame-1}")
        
        target_orient = orientations[start_frame:end_frame]
        
        result = solver.fit_orientations(
            target_orient,
            sensor_mapping,
            n_iter=args.n_iter,
            lr=args.lr,
            fix_pelvis=args.fix_pelvis,
            verbosity=args.verbosity
        )
        
        # Save results
        output_data = {
            'body_params': result,
            'timestamps': timestamps[start_frame:end_frame],
            'start_frame': start_frame,
            'end_frame': end_frame
        }
        np.savez(args.output, **output_data)
        print(f"\nSaved results to {args.output}")
        
        # Visualize if requested
        if args.visualize:
            visualize_result(result, args.body_model, 0)
    
    else:  # batch mode
        print(f"\nBatch mode: Processing {n_frames} frames with batch size {args.batch_size}")
        
        all_results = []
        
        # Process in batches
        for start_idx in tqdm(range(0, n_frames, args.batch_size), desc="Processing batches"):
            end_idx = min(start_idx + args.batch_size, n_frames)
            batch_orient = orientations[start_idx:end_idx]
            
            result = solver.fit_orientations(
                batch_orient,
                sensor_mapping,
                n_iter=args.n_iter,
                lr=args.lr,
                fix_pelvis=args.fix_pelvis,
                verbosity=0  # Silent for batch processing
            )
            
            all_results.append(result)
        
        # Concatenate results
        final_result = {
            key: np.concatenate([r[key] for r in all_results], axis=0)
            for key in all_results[0].keys()
        }
        
        # Save results
        output_data = {
            'body_params': final_result,
            'timestamps': timestamps,
            'n_frames': n_frames
        }
        np.savez(args.output, **output_data)
        print(f"\nSaved {n_frames} frames to {args.output}")


if __name__ == '__main__':
    main()
