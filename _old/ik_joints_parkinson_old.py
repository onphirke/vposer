# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2021.02.12

"""
Inverse Kinematics for Parkinson's Patient Motion Data

This script performs inverse kinematics to fit SMPL body model parameters 
to joint position data from Parkinson's patient motion capture.
"""

import os
from os import path as osp
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from colour import Color
from torch import nn

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine


class SourceKeyPoints(nn.Module):
    """
    PyTorch module that extracts joint positions from SMPL body model.
    
    This class serves as a wrapper around the BodyModel to provide joint positions
    that can be used as source keypoints for inverse kinematics optimization.
    """
    
    def __init__(
        self,
        bm: Union[str, BodyModel],
        n_joints: int = 22,
        kpts_colors: Union[np.ndarray, None] = None,
    ):
        """
        Initialize the SourceKeyPoints module.
        
        Args:
            bm: Path to SMPL model file or BodyModel instance
            n_joints: Number of joints to extract (default: 22)
            kpts_colors: RGB colors for keypoints visualization
        """
        super(SourceKeyPoints, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []  # Face indices (unused)
        self.n_joints = n_joints
        self.kpts_colors = (
            np.array([Color("grey").rgb for _ in range(n_joints)])
            if kpts_colors is None
            else kpts_colors
        )

    def forward(self, body_parms: Dict) -> Dict:
        """
        Extract joint positions from body parameters.
        
        Args:
            body_parms: Dictionary of SMPL body parameters
            
        Returns:
            Dictionary containing source keypoints and full body model output
        """
        new_body = self.bm(**body_parms)
        
        print("new_body.Jtr.shape", new_body.Jtr.shape)
        return {"source_kpts": new_body.Jtr[:, : self.n_joints], "body": new_body}


def setup_paths() -> Tuple[str, str, str]:
    """
    Setup file paths for models and data.
    
    Returns:
        Tuple of (vposer_expr_dir, bm_fname, target_pts_file)
    """
    support_dir = "../support_data/dowloads"
    vposer_expr_dir = "../_data/_runs/V_me_all"  # VPoser model directory
    bm_fname = osp.join(support_dir, "models/smplx/neutral/model.npz")  # SMPL-X model
    target_pts_file = "../_data/patient_motion_full.csv"  # Patient motion data
    
    return vposer_expr_dir, bm_fname, target_pts_file


def define_body_joints() -> List[str]:
    """
    Define the list of body joint names used in the motion capture data.
    
    Returns:
        List of joint names in order
    """
    return [
        'head', 'hip_center', 'hip_R', 'hip_L', 'spine',
        'knee_L', 'knee_R', 'neck', 'shoulder_L', 'elbow_L',
        'wrist_L', 'hand_L', 'shoulder_R', 'elbow_R', 'wrist_R',
        'hand_R', 'ankle_L', 'foot_L', 'ankle_R', 'foot_R',
    ]


def load_target_points(target_pts_file: str, body_spots: List[str], 
                      frame_idx: int = 0, device: torch.device = None) -> torch.Tensor:
    """
    Load target joint positions from CSV motion capture data.
    
    Args:
        target_pts_file: Path to CSV file containing motion data
        body_spots: List of joint names to extract
        frame_idx: Frame index to use (default: 0)
        device: Torch device to move data to
        
    Returns:
        Tensor of target joint positions [1, n_joints, 3]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load motion data from CSV
    target_pts_df = pd.read_csv(target_pts_file)
    
    # Extract unique joint names from column headers (remove _x, _y, _z suffixes)
    point_names = list(
        {"_".join(s.split("_")[:-1]): None for s in target_pts_df.columns}.keys()
    )
    
    # Group columns by joint name (x, y, z coordinates)
    point_names_groups = {s: (f"{s}_x", f"{s}_y", f"{s}_z") for s in point_names}
    
    # Convert to tensors for each joint
    target_pts_map = {
        s: torch.tensor(target_pts_df[list(point_names_groups[s])].values).float()
        for s in point_names
    }
    
    # Extract specified frame for each body spot
    target_pts_list = [target_pts_map[joint_name][frame_idx] for joint_name in body_spots]
    target_pts = torch.stack(target_pts_list, dim=0).unsqueeze(0)  # Add batch dimension
    
    return target_pts.detach().to(device)


def setup_visualization(n_joints: int) -> List[List[float]]:
    """
    Setup color gradient for keypoint visualization.
    
    Args:
        n_joints: Number of joints for coloring
        
    Returns:
        List of RGB color values
    """
    # Set OpenGL platform for headless rendering
    os.environ["PYOPENGL_PLATFORM"] = "glx"
    
    # Create color gradient from red to blue
    red = Color("red")
    blue = Color("blue")
    kpts_colors = [c.rgb for c in list(red.range_to(blue, n_joints))]
    
    return kpts_colors


def setup_ik_engine(vposer_expr_dir: str, device: torch.device) -> IK_Engine:
    """
    Initialize the Inverse Kinematics optimization engine.
    
    Args:
        vposer_expr_dir: Path to trained VPoser model
        device: Torch device for computation
        
    Returns:
        Configured IK_Engine instance
    """
    # Loss function for data fitting
    data_loss = torch.nn.MSELoss(reduction="sum")
    
    # Weights for different loss terms during optimization
    stepwise_weights = [
        {"data": 10.0, "poZ_body": 0.01, "betas": 0.5},
    ]
    
    # L-BFGS optimizer configuration
    optimizer_args = {
        "type": "LBFGS",
        "max_iter": 300,
        "lr": 1,
        "tolerance_change": 1e-4,
        "history_size": 200,
    }
    
    # Initialize IK engine
    ik_engine = IK_Engine(
        vposer_expr_dir=vposer_expr_dir,
        verbosity=2,  # Enable verbose output
        display_rc=(1, 1),  # Display configuration
        data_loss=data_loss,
        stepwise_weights=stepwise_weights,
        optimizer_args=optimizer_args,
        num_betas=1,  # Number of shape parameters
    ).to(device)
    
    return ik_engine


def run_inverse_kinematics(ik_engine: IK_Engine, source_pts: SourceKeyPoints, 
                          target_pts: torch.Tensor) -> Dict:
    """
    Run inverse kinematics optimization to fit body model to target joints.
    
    Args:
        ik_engine: Configured IK optimization engine
        source_pts: Source keypoints module
        target_pts: Target joint positions to fit
        
    Returns:
        Dictionary of optimized body parameters
        
    Raises:
        ValueError: If optimization results contain NaN values
    """
    # Run IK optimization
    ik_res, ik_loss = ik_engine(source_pts, target_pts)
    
    # Detach results from computation graph
    ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
    
    # Check for NaN values in results
    nan_mask = torch.isnan(ik_res_detached["trans"]).sum(-1) != 0
    if nan_mask.sum() != 0:
        raise ValueError("Optimization results contain NaN values!")
    
    # detach the loss
    ik_loss = ik_loss.detach()
    
    return ik_res_detached, ik_loss


def main():
    """
    Main function to run the inverse kinematics pipeline.
    """
    # Setup computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup file paths
    vposer_expr_dir, bm_fname, target_pts_file = setup_paths()
    
    # Define body joints to track
    body_spots = define_body_joints()
    n_joints = len(body_spots)
    print(f"Tracking {n_joints} body joints")
    
    # Load target motion data
    frame_idx = 0  # Use first frame
    target_pts = load_target_points(target_pts_file, body_spots, frame_idx, device)
    print(f"Loaded target points with shape: {target_pts.shape}")
    
    # Setup visualization colors
    kpts_colors = setup_visualization(n_joints)
    
    # Initialize IK engine
    ik_engine = setup_ik_engine(vposer_expr_dir, device)
    
    # Create source keypoints module
    source_pts = SourceKeyPoints(
        bm=bm_fname, 
        n_joints=n_joints, 
        kpts_colors=kpts_colors
    ).to(device)
    
    # Run inverse kinematics optimization
    print("Running inverse kinematics optimization...")
    ik_results, ik_loss = run_inverse_kinematics(ik_engine, source_pts, target_pts)
    
    print("Inverse kinematics completed successfully!")
    print(f"Result keys: {list(ik_results.keys())}")
    
    return ik_results, ik_loss


if __name__ == "__main__":
    print(main())
