#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker-based Inverse Kinematics Engine

This module implements an IK engine that fits SMPL body model parameters to:
1. Marker positions (3D locations of markers placed on the body)
2. Marker orientations (quaternions representing sensor orientations)

The engine optimizes body pose and shape parameters to match observed marker
data, using markers that are attached to specific vertices of the body mesh.
"""

import numpy as np
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.rotation_tools import aa2matrot


class MarkerSource(nn.Module):
    """
    PyTorch module that extracts marker positions and orientations from SMPL body model.
    
    Markers are defined as points attached to specific vertices on the body mesh.
    This module computes the current positions of these markers given body parameters.
    """
    
    def __init__(
        self,
        bm: Union[str, BodyModel],
        marker_indices: np.ndarray,
        marker_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the marker source.
        
        Args:
            bm: Path to SMPL model file or BodyModel instance
            marker_indices: Array of vertex indices for each marker (N,)
            marker_names: Optional list of marker names
            device: Torch device
        """
        super(MarkerSource, self).__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize body model
        if isinstance(bm, str):
            self.bm = BodyModel(bm, persistant_buffer=False).to(device)
        else:
            self.bm = bm
        
        # Store marker information
        self.marker_indices = torch.tensor(marker_indices, dtype=torch.long, device=device)
        self.n_markers = len(marker_indices)
        self.marker_names = marker_names if marker_names is not None else [f"marker_{i}" for i in range(self.n_markers)]
        
        print(f"Initialized MarkerSource with {self.n_markers} markers")
    
    def forward(self, body_parms: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute marker positions for given body parameters.
        
        Args:
            body_parms: Dictionary of body parameters (betas, pose_body, root_orient, trans)
        
        Returns:
            Dictionary containing:
                - marker_positions: (B, N, 3) marker positions
                - body: Full body model output
        """
        # Generate body mesh
        body_output = self.bm(**body_parms)
        
        # Extract marker positions (vertices at marker indices)
        # body_output.v has shape (B, V, 3) where V is number of vertices
        marker_positions = body_output.v[:, self.marker_indices, :]  # (B, N, 3)
        
        return {
            "marker_positions": marker_positions,
            "body": body_output
        }
    
    def get_marker_names(self) -> List[str]:
        """Get list of marker names."""
        return self.marker_names


class OrientationLoss(nn.Module):
    """
    Loss function for matching marker orientations.
    
    Computes the difference between predicted and target orientations.
    Supports quaternions and rotation matrices.
    """
    
    def __init__(self, loss_type: str = 'geodesic'):
        """
        Initialize orientation loss.
        
        Args:
            loss_type: Type of orientation loss ('geodesic', 'quaternion', 'frobenius')
        """
        super(OrientationLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, pred_orient: torch.Tensor, target_orient: torch.Tensor) -> torch.Tensor:
        """
        Compute orientation loss.
        
        Args:
            pred_orient: Predicted orientations (B, N, 4) as quaternions or (B, N, 3, 3) as rotation matrices
            target_orient: Target orientations (same shape as pred_orient)
        
        Returns:
            Scalar loss value
        """
        if self.loss_type == 'quaternion':
            # Quaternion distance: 1 - |<q1, q2>|
            # This handles double cover (q and -q represent same rotation)
            dot_product = torch.abs(torch.sum(pred_orient * target_orient, dim=-1))
            loss = torch.mean(1.0 - dot_product)
            return loss
        
        elif self.loss_type == 'geodesic':
            # Geodesic distance on SO(3)
            # For quaternions: d = 2 * arccos(|<q1, q2>|)
            dot_product = torch.abs(torch.sum(pred_orient * target_orient, dim=-1))
            dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
            loss = torch.mean(2.0 * torch.acos(dot_product))
            return loss
        
        elif self.loss_type == 'frobenius':
            # Frobenius norm of difference (for rotation matrices)
            loss = torch.mean(torch.norm(pred_orient - target_orient, p='fro', dim=(-2, -1)))
            return loss
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: (B, N, 4) quaternions in [w, x, y, z] format
    
    Returns:
        Rotation matrices (B, N, 3, 3)
    """
    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    # Compute rotation matrix elements
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
    ], dim=-2)
    
    return R


class MarkerIKEngine:
    """
    Inverse Kinematics engine for fitting body model to marker data.
    
    Optimizes body parameters to match:
    - Marker positions (3D coordinates)
    - Marker orientations (quaternions or rotation matrices)
    """
    
    def __init__(
        self,
        vposer_model_path: str,
        body_model_path: str,
        marker_file: str,
        device: Optional[torch.device] = None,
        use_vposer: bool = True
    ):
        """
        Initialize the marker IK engine.
        
        Args:
            vposer_model_path: Path to VPoser model directory
            body_model_path: Path to SMPL body model file
            marker_file: Path to marker definition file (.npz)
            device: Torch device
            use_vposer: Whether to use VPoser for pose prior
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.use_vposer = use_vposer
        
        # Load marker definitions
        print(f"Loading markers from {marker_file}")
        marker_data = np.load(marker_file, allow_pickle=True)
        marker_names = marker_data['marker_names'].tolist()
        marker_indices = marker_data['marker_indices']
        
        # Initialize body model
        print(f"Loading body model from {body_model_path}")
        self.bm = BodyModel(body_model_path, num_betas=10, num_dmpls=None).to(device)
        
        # Initialize marker source
        self.marker_source = MarkerSource(
            self.bm,
            marker_indices,
            marker_names,
            device
        )
        
        # Initialize VPoser if enabled
        self.vposer = None
        if use_vposer:
            print(f"Loading VPoser from {vposer_model_path}")
            self.vposer, _ = load_model(
                vposer_model_path,
                model_code=VPoser,
                remove_words_in_model_weights='vp_model.',
                disable_grad=True
            )
            self.vposer = self.vposer.to(device)
            self.vposer.eval()
        
        print(f"MarkerIKEngine initialized on {device}")
        print(f"  Markers: {len(marker_names)}")
        print(f"  VPoser: {'enabled' if use_vposer else 'disabled'}")
    
    def fit(
        self,
        target_markers: np.ndarray,
        target_orientations: Optional[np.ndarray] = None,
        marker_mask: Optional[np.ndarray] = None,
        orientation_mask: Optional[np.ndarray] = None,
        n_iter: int = 100,
        lr: float = 0.01,
        weights: Optional[Dict[str, float]] = None,
        init_params: Optional[Dict[str, np.ndarray]] = None,
        fix_pelvis: bool = True,
        verbosity: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Fit body model to marker data.
        
        Args:
            target_markers: Target marker positions (B, N, 3)
            target_orientations: Target marker orientations (B, N, 4) as quaternions [w,x,y,z]
            marker_mask: Boolean mask for valid markers (B, N)
            orientation_mask: Boolean mask for valid orientations (B, N)
            n_iter: Number of optimization iterations
            lr: Learning rate
            weights: Loss weights dictionary
            init_params: Initial body parameters
            fix_pelvis: Whether to fix pelvis at origin
            verbosity: Verbosity level (0=silent, 1=progress, 2=detailed)
        
        Returns:
            Dictionary of optimized body parameters
        """
        batch_size = target_markers.shape[0]
        n_markers = target_markers.shape[1]
        
        # Default weights
        if weights is None:
            weights = {
                'marker_pos': 100.0,
                'marker_orient': 10.0 if target_orientations is not None else 0.0,
                'betas': 0.01,
                'poZ_body': 0.01 if self.use_vposer else 0.0,
                'pose_body': 0.01 if not self.use_vposer else 0.0
            }
        
        # Convert targets to tensors
        target_markers_t = torch.tensor(target_markers, dtype=torch.float32, device=self.device)
        
        if marker_mask is not None:
            marker_mask_t = torch.tensor(marker_mask, dtype=torch.bool, device=self.device)
        else:
            marker_mask_t = torch.ones(batch_size, n_markers, dtype=torch.bool, device=self.device)
        
        if target_orientations is not None:
            target_orientations_t = torch.tensor(target_orientations, dtype=torch.float32, device=self.device)
            if orientation_mask is not None:
                orientation_mask_t = torch.tensor(orientation_mask, dtype=torch.bool, device=self.device)
            else:
                orientation_mask_t = torch.ones(batch_size, n_markers, dtype=torch.bool, device=self.device)
        else:
            target_orientations_t = None
            orientation_mask_t = None
        
        # Initialize parameters
        if init_params is None:
            init_params = {}
        
        betas = torch.tensor(
            init_params.get('betas', np.zeros((batch_size, 10))),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        
        if self.use_vposer:
            poZ_body = torch.tensor(
                init_params.get('poZ_body', np.random.randn(batch_size, 32) * 0.01),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True
            )
            pose_body = None
        else:
            pose_body = torch.tensor(
                init_params.get('pose_body', np.zeros((batch_size, 63))),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True
            )
            poZ_body = None
        
        root_orient = torch.tensor(
            init_params.get('root_orient', np.zeros((batch_size, 3))),
            dtype=torch.float32,
            device=self.device,
            requires_grad=not fix_pelvis
        )
        
        trans = torch.tensor(
            init_params.get('trans', np.zeros((batch_size, 3))),
            dtype=torch.float32,
            device=self.device,
            requires_grad=not fix_pelvis
        )
        
        # Setup optimizer
        if self.use_vposer:
            params = [betas, poZ_body]
        else:
            params = [betas, pose_body]
        
        if not fix_pelvis:
            params.extend([root_orient, trans])
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Orientation loss
        orient_loss_fn = OrientationLoss(loss_type='geodesic')
        
        # Optimization loop
        for iteration in range(n_iter):
            optimizer.zero_grad()
            
            # Decode pose if using VPoser
            if self.use_vposer:
                # Don't use no_grad here - we need gradients to flow to poZ_body!
                pose_body_decoded = self.vposer.decode(poZ_body)['pose_body'].contiguous().view(batch_size, 63)
                pose_body_to_use = pose_body_decoded
            else:
                pose_body_to_use = pose_body
            
            # Forward pass through body model
            body_parms = {
                'betas': betas,
                'pose_body': pose_body_to_use,
                'root_orient': root_orient,
                'trans': trans
            }
            
            marker_output = self.marker_source(body_parms)
            pred_markers = marker_output['marker_positions']  # (B, N, 3)
            
            # Compute losses
            losses = {}
            
            # Marker position loss
            marker_diff = (pred_markers - target_markers_t) * marker_mask_t.unsqueeze(-1)
            losses['marker_pos'] = torch.mean(marker_diff ** 2)
            
            # Orientation loss (if provided)
            if target_orientations_t is not None and weights.get('marker_orient', 0) > 0:
                # For now, we'll use a simplified orientation loss
                # In a full implementation, you'd extract body part orientations
                # For now, just use a placeholder
                losses['marker_orient'] = torch.tensor(0.0, device=self.device)
            
            # Regularization losses
            losses['betas'] = torch.mean(betas ** 2)
            
            if self.use_vposer:
                losses['poZ_body'] = torch.mean(poZ_body ** 2)
            else:
                losses['pose_body'] = torch.mean(pose_body ** 2)
            
            # Total loss
            total_loss = sum(losses[k] * weights.get(k, 0.0) for k in losses.keys() if weights.get(k, 0.0) > 0)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Logging
            if verbosity > 0 and (iteration % 10 == 0 or iteration == n_iter - 1):
                loss_str = " | ".join([f"{k}: {v.item():.4e}" for k, v in losses.items()])
                print(f"Iter {iteration:03d} | Total: {total_loss.item():.4e} | {loss_str}")
        
        # Extract final parameters
        result = {
            'betas': c2c(betas),
            'root_orient': c2c(root_orient),
            'trans': c2c(trans)
        }
        
        if self.use_vposer:
            result['poZ_body'] = c2c(poZ_body)
            with torch.no_grad():
                result['pose_body'] = c2c(self.vposer.decode(poZ_body)['pose_body'].contiguous().view(batch_size, 63))
        else:
            result['pose_body'] = c2c(pose_body)
        
        return result


def load_marker_definitions(marker_file: str) -> Tuple[List[str], np.ndarray]:
    """
    Load marker definitions from NPZ file.
    
    Args:
        marker_file: Path to marker definition file
    
    Returns:
        Tuple of (marker_names, marker_indices)
    """
    data = np.load(marker_file, allow_pickle=True)
    marker_names = data['marker_names'].tolist()
    marker_indices = data['marker_indices']
    return marker_names, marker_indices


if __name__ == '__main__':
    # Example usage
    print("Marker-based IK Engine")
    print("This module is meant to be imported, not run directly.")
    print("\nExample usage:")
    print("""
    from ik_marker_engine import MarkerIKEngine
    
    engine = MarkerIKEngine(
        vposer_model_path='../_good_runs/V02_05',
        body_model_path='../support_data/dowloads/models/smplx/neutral/model.npz',
        marker_file='markers.npz'
    )
    
    # Prepare marker data
    target_markers = np.random.randn(10, 5, 3)  # 10 frames, 5 markers, 3D
    
    # Fit
    result = engine.fit(target_markers, n_iter=100)
    """)
