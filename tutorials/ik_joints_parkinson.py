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
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
from colour import Color
from torch import nn

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine
from human_body_prior.tools.omni_tools import create_list_chunks


# class SourceKeyPoints(nn.Module):
#     """
#     PyTorch module that extracts joint positions from SMPL body model.

#     This class serves as a wrapper around the BodyModel to provide joint positions
#     that can be used as source keypoints for inverse kinematics optimization.
#     """

#     def __init__(
#         self,
#         bm: Union[str, BodyModel],
#         n_joints: int = 22,
#         kpts_colors: Union[np.ndarray, None] = None,
#     ):
#         """
#         Initialize the SourceKeyPoints module.

#         Args:
#             bm: Path to SMPL model file or BodyModel instance
#             n_joints: Number of joints to extract (default: 22)
#             kpts_colors: RGB colors for keypoints visualization
#         """
#         super(SourceKeyPoints, self).__init__()

#         self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
#         self.bm_f = []  # Face indices (unused)
#         self.n_joints = n_joints
#         self.kpts_colors = (
#             np.array([Color("grey").rgb for _ in range(n_joints)])
#             if kpts_colors is None
#             else kpts_colors
#         )

#     def forward(self, body_parms: Dict) -> Dict:
#         """
#         Extract joint positions from body parameters.

#         Args:
#             body_parms: Dictionary of SMPL body parameters

#         Returns:
#             Dictionary containing source keypoints and full body model output
#         """
#         new_body = self.bm(**body_parms)

#         print("new_body.Jtr.shape", new_body.Jtr.shape)
#         return {"source_kpts": new_body.Jtr[:, : self.n_joints], "body": new_body}


def setup_paths() -> Tuple[str, str, str]:
    """
    Setup file paths for models and data.

    Returns:
        Tuple of (vposer_expr_dir, bm_fname, target_pts_file)
    """
    support_dir = "../support_data/dowloads"
    vposer_expr_dir = "../_good_runs/V02_05"  # VPoser model directory
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
        "head",
        "hip_center",
        "hip_R",
        "hip_L",
        "spine",
        "knee_L",
        "knee_R",
        "neck",
        "shoulder_L",
        "elbow_L",
        "wrist_L",
        "hand_L",
        "shoulder_R",
        "elbow_R",
        "wrist_R",
        "hand_R",
        "ankle_L",
        "foot_L",
        "ankle_R",
        "foot_R",
    ]


def get_csv_joint_names(target_pts_file: str) -> List[str]:
    """
    Extract all available joint names from the CSV file.

    Args:
        target_pts_file: Path to CSV file containing motion data

    Returns:
        List of unique joint names found in the CSV
    """
    target_pts_df = pd.read_csv(target_pts_file)

    # Extract unique joint names from column headers (remove _x, _y, _z suffixes)
    point_names = list(
        {"_".join(s.split("_")[:-1]): None for s in target_pts_df.columns}.keys()
    )

    return sorted(point_names)  # Sort for consistency


def get_smpl_joint_names() -> List[str]:
    """
    Get the standard SMPL joint names in the order they appear in Jtr.

    These correspond to the first 24 joints in the SMPL skeleton.
    Based on standard SMPL joint definitions.

    Returns:
        List of SMPL joint names in order
    """
    return [
        "pelvis",  # 0
        "left_hip",  # 1
        "right_hip",  # 2
        "spine1",  # 3
        "left_knee",  # 4
        "right_knee",  # 5
        "spine2",  # 6
        "left_ankle",  # 7
        "right_ankle",  # 8
        "spine3",  # 9
        "left_foot",  # 10
        "right_foot",  # 11
        "neck",  # 12
        "left_collar",  # 13
        "right_collar",  # 14
        "head",  # 15
        "left_shoulder",  # 16
        "right_shoulder",  # 17
        "left_elbow",  # 18
        "right_elbow",  # 19
        "left_wrist",  # 20
        "right_wrist",  # 21
        "left_hand",  # 22
        "right_hand",  # 23
    ]


def create_joint_mapping_candidates(
    csv_joints: List[str], smpl_joints: List[str]
) -> List[Dict[str, str]]:
    """
    Create candidate mappings between CSV joint names and SMPL joint names.

    Uses semantic matching based on joint name similarity.

    Args:
        csv_joints: Joint names from CSV file
        smpl_joints: SMPL joint names in order

    Returns:
        List of candidate mapping dictionaries {csv_name: smpl_name}
    """
    # Define semantic mappings based on common naming patterns
    semantic_mappings = {
        # Head and neck
        "head": ["head"],
        "neck": ["neck"],
        # Torso
        "hip_center": ["pelvis"],
        "spine": ["spine1"],
        # Left side
        "hip_L": ["left_hip"],
        "knee_L": ["left_knee"],
        "ankle_L": ["left_ankle"],
        "foot_L": ["left_foot"],
        "shoulder_L": ["left_shoulder"],
        "elbow_L": ["left_elbow"],
        "wrist_L": ["left_wrist"],
        # "hand_L": ["left_hand"],
        # Right side
        "hip_R": ["right_hip"],
        "knee_R": ["right_knee"],
        "ankle_R": ["right_ankle"],
        "foot_R": ["right_foot"],
        "shoulder_R": ["right_shoulder"],
        "elbow_R": ["right_elbow"],
        "wrist_R": ["right_wrist"],
        # "hand_R": ["right_hand"],
    }

    candidates = []

    # Create mapping based on semantic similarity
    mapping = {}
    for csv_joint in csv_joints:
        if csv_joint in semantic_mappings:
            # Try each possible SMPL joint for this CSV joint
            for smpl_candidate in semantic_mappings[csv_joint]:
                if smpl_candidate in smpl_joints:
                    mapping[csv_joint] = smpl_candidate
                    break

    if mapping:
        candidates.append(mapping)

    return candidates


def find_best_joint_mapping(
    target_pts_file: str,
    bm_fname: str,
    vposer_expr_dir: str,
    device: torch.device,
    frame_idx: int = 0,
) -> Tuple[Dict[str, int], float]:
    """
    Find the best mapping between CSV joints and SMPL joints by testing different combinations.

    Args:
        target_pts_file: Path to CSV motion data
        bm_fname: Path to SMPL model
        vposer_expr_dir: Path to VPoser model
        device: Torch device for computation
        frame_idx: Frame index to use for testing

    Returns:
        Tuple of (best_mapping, best_loss) where best_mapping maps CSV joint names to SMPL joint indices
    """
    print("Discovering joint mapping between CSV and SMPL...")

    # Get available joints from both sources
    csv_joints = get_csv_joint_names(target_pts_file)
    smpl_joints = get_smpl_joint_names()

    print(f"CSV joints found: {csv_joints}")
    print(f"SMPL joints available: {smpl_joints[:10]}...")  # Show first 10

    # Create candidate mappings
    mapping_candidates = create_joint_mapping_candidates(csv_joints, smpl_joints)

    if not mapping_candidates:
        raise ValueError("Could not create any joint mapping candidates!")

    print(f"Testing {len(mapping_candidates)} mapping candidate(s)...")

    # Load CSV data
    target_pts_df = pd.read_csv(target_pts_file)
    point_names_groups = {s: (f"{s}_x", f"{s}_y", f"{s}_z") for s in csv_joints}
    target_pts_map = {
        s: torch.tensor(target_pts_df[list(point_names_groups[s])].values).float()
        for s in csv_joints
    }

    best_mapping = None
    best_loss = float("inf")

    for i, candidate_mapping in enumerate(mapping_candidates):
        try:
            print(
                f"  Testing mapping {i + 1}/{len(mapping_candidates)}: {candidate_mapping}"
            )

            # Create joint index mapping
            joint_idx_mapping = {}
            target_joints = []

            for csv_joint, smpl_joint in candidate_mapping.items():
                if csv_joint in target_pts_map and smpl_joint in smpl_joints:
                    smpl_idx = smpl_joints.index(smpl_joint)
                    joint_idx_mapping[csv_joint] = smpl_idx
                    target_joints.append(csv_joint)

            if len(target_joints) < 5:  # Need at least 5 joints for meaningful IK
                print(f"    Skipping - only {len(target_joints)} joints matched")
                continue

            # Create target points tensor
            target_pts_list = [
                target_pts_map[joint][frame_idx] for joint in target_joints
            ]
            target_pts = torch.stack(target_pts_list, dim=0).unsqueeze(0).to(device)

            # Create source points with correct joint indices
            joint_indices = [joint_idx_mapping[joint] for joint in target_joints]
            source_pts = SourceKeyPointsMapped(bm_fname, joint_indices).to(device)

            # Setup lightweight IK engine for testing
            data_loss = torch.nn.MSELoss(reduction="sum")
            stepwise_weights = [{"data": 10.0, "poZ_body": 0.01, "betas": 0.5}]
            optimizer_args = {
                "type": "LBFGS",
                "max_iter": 50,
                "lr": 1,
                "tolerance_change": 1e-3,
            }

            ik_engine = IK_Engine(
                vposer_expr_dir=vposer_expr_dir,
                verbosity=1,  # Silent for testing
                display_rc=(1, 1),
                data_loss=data_loss,
                stepwise_weights=stepwise_weights,
                optimizer_args=optimizer_args,
                num_betas=10,
            ).to(device)

            # Run quick IK test
            ik_res, loss = ik_engine(source_pts, target_pts)
            loss_value = loss.item()

            print(f"    Loss: {loss_value:.4f} with {len(target_joints)} joints")

            if loss_value < best_loss:
                best_loss = loss_value
                best_mapping = joint_idx_mapping

        except Exception as e:
            print(f"    Failed: {str(e)}")
            continue

    if best_mapping is None:
        raise ValueError("Could not find any working joint mapping!")

    print(f"Best mapping found with loss {best_loss:.4f}")
    print(f"Joint mapping: {best_mapping}")

    return best_mapping, best_loss


class SourceKeyPointsMapped(nn.Module):
    """
    Version of SourceKeyPoints that uses specific joint indices.
    """

    def __init__(
        self,
        bm: Union[str, BodyModel],
        joint_indices: List[int],
        kpts_colors: Union[np.ndarray, None] = None,
    ):
        super(SourceKeyPointsMapped, self).__init__()

        self.bm = BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
        self.bm_f = []  # Face indices (unused)
        self.joint_indices = joint_indices
        self.kpts_colors = (
            kpts_colors
            if kpts_colors is not None
            else np.array([Color("grey").rgb for _ in range(len(joint_indices))])
        )

    def forward(self, body_parms: Dict) -> Dict:
        new_body = self.bm(**body_parms)
        selected_joints = new_body.Jtr[:, self.joint_indices]
        return {"source_kpts": selected_joints, "body": new_body}


# def load_target_points(target_pts_file: str, body_spots: List[str],
#                       frame_idx: int = 0, device: torch.device = None) -> torch.Tensor:
#     """
#     Load target joint positions from CSV motion capture data.

#     Args:
#         target_pts_file: Path to CSV file containing motion data
#         body_spots: List of joint names to extract
#         frame_idx: Frame index to use (default: 0)
#         device: Torch device to move data to

#     Returns:
#         Tensor of target joint positions [1, n_joints, 3]
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load motion data from CSV
#     target_pts_df = pd.read_csv(target_pts_file)

#     # Extract unique joint names from column headers (remove _x, _y, _z suffixes)
#     point_names = list(
#         {"_".join(s.split("_")[:-1]): None for s in target_pts_df.columns}.keys()
#     )

#     # Group columns by joint name (x, y, z coordinates)
#     point_names_groups = {s: (f"{s}_x", f"{s}_y", f"{s}_z") for s in point_names}

#     # Convert to tensors for each joint
#     target_pts_map = {
#         s: torch.tensor(target_pts_df[list(point_names_groups[s])].values).float()
#         for s in point_names
#     }

#     # Extract specified frame for each body spot
#     target_pts_list = [target_pts_map[joint_name][frame_idx] for joint_name in body_spots]
#     target_pts = torch.stack(target_pts_list, dim=0).unsqueeze(0)  # Add batch dimension

#     return target_pts.detach().to(device)


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


def setup_ik_engine(
    vposer_expr_dir: str,
    device: torch.device,
    verbosity: int = 1,
    display_rc: Tuple[int, int] = (1, 1),
) -> IK_Engine:
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
    # stepwise_weights = [
    #     {"data": 10.0, "poZ_body": 0.01, "betas": 0.5},
    # ]
    stepwise_weights = [
        # {"data": 10.0, "poZ_body": 0.00, "betas": 0.0},
        # {"data": 10.0, "poZ_body": 0.00, "betas": 0.1},
        {"data": 10.0, "poZ_body": 0.01, "betas": 0.5},
    ]

    # L-BFGS optimizer configuration
    optimizer_args = {
        "type": "LBFGS",
        "max_iter": 500,
        "lr": 0.5,
        "tolerance_change": 1e-4,
        "history_size": 200,
    }

    # Initialize IK engine
    ik_engine = IK_Engine(
        vposer_expr_dir=vposer_expr_dir,
        verbosity=verbosity,  # Reduced verbosity to avoid visualizations
        display_rc=display_rc,  # Display configuration
        data_loss=data_loss,
        stepwise_weights=stepwise_weights,
        optimizer_args=optimizer_args,
        num_betas=10,  # Number of shape parameters
    ).to(device)

    return ik_engine


def run_inverse_kinematics(
    ik_engine: IK_Engine, source_pts: SourceKeyPointsMapped, target_pts: torch.Tensor
) -> Dict:
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
    print(source_pts, target_pts.shape)
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


def run_ik_for_frames(
    frame_ids: List[int],
    joint_mapping: Dict[str, int],
    target_pts_file: str,
    bm_fname: str,
    vposer_expr_dir: str,
    device: torch.device,
    verbosity: int = 1,
    display_rc: Tuple[int, int] = (1, 1),
) -> Tuple[Dict, torch.Tensor]:
    """
    Run inverse kinematics for a single frame using the discovered joint mapping.

    Args:
        frame_idx: Frame index to process
        joint_mapping: Mapping from CSV joint names to SMPL joint indices
        target_pts_file: Path to CSV motion data
        bm_fname: Path to SMPL model file
        vposer_expr_dir: Path to VPoser model
        device: Torch device for computation

    Returns:
        Tuple of (ik_results, ik_loss)
    """
    # Load target points using discovered mapping
    csv_joints = get_csv_joint_names(target_pts_file)
    mapped_joints = [joint for joint in csv_joints if joint in joint_mapping]

    # Load target motion data with mapped joints
    target_pts_df = pd.read_csv(target_pts_file)
    point_names_groups = {s: (f"{s}_x", f"{s}_y", f"{s}_z") for s in csv_joints}
    target_pts_map = {
        s: torch.tensor(target_pts_df[list(point_names_groups[s])].values).float()
        for s in csv_joints
    }

    # Extract frames for the batch [batch_size, n_joints, 3]
    target_pts_list = []
    for frame_i in frame_ids:
        frame_pts = [target_pts_map[joint][frame_i] for joint in mapped_joints]
        target_pts_list.append(torch.stack(frame_pts, dim=0))

    target_pts = torch.stack(target_pts_list, dim=0).to(
        device
    )  # [batch_size, n_joints, 3]
    target_pts = target_pts.type(torch.float)
    print(f"Loaded target points with shape: {target_pts.shape}")

    # Create joint indices for SMPL model
    joint_indices = [joint_mapping[joint] for joint in mapped_joints]

    # Setup visualization colors
    n_joints = len(mapped_joints)
    kpts_colors = setup_visualization(n_joints)

    # Initialize IK engine with full optimization settings
    # ik_engine = setup_ik_engine(vposer_expr_dir, device)
    ik_engine = setup_ik_engine(
        vposer_expr_dir,
        device,
        verbosity=verbosity,
        display_rc=display_rc,
    )

    # Create source keypoints module with mapped joint indices
    source_pts = SourceKeyPointsMapped(
        bm_fname, joint_indices, kpts_colors=kpts_colors
    ).to(device)

    # Run inverse kinematics optimization
    ik_results, ik_loss = run_inverse_kinematics(ik_engine, source_pts, target_pts)

    return ik_results, ik_loss


def main():
    """
    Main function to run the inverse kinematics pipeline with automatic joint mapping discovery.
    """
    # Setup computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup file paths
    vposer_expr_dir, bm_fname, target_pts_file = setup_paths()

    joint_mapping, mapping_loss = find_best_joint_mapping(
        target_pts_file, bm_fname, vposer_expr_dir, device, 0
    )

    # =================================================
    # testing
    # =================================================

    # run_ik_for_frames(
    #     list(range(115, 150, 4)),
    #     joint_mapping,
    #     target_pts_file,
    #     bm_fname,
    #     vposer_expr_dir,
    #     device,
    #     verbosity=2,
    #     display_rc=(3, 3),
    # )
    # exit()

    # =================================================

    # Discover the best joint mapping between CSV and SMPL
    results = []
    batch_size = 300
    num_frames = 300

    # Process frames in batches
    for frame_ids in create_list_chunks(
        list(range(0, num_frames)),
        batch_size,
        overlap_size=0,
        cut_smaller_batches=False,
    ):
        start_frame = frame_ids[0]
        end_frame = frame_ids[-1] + 1
        print(f"\nProcessing batch: frames {start_frame} to {end_frame - 1}")
        ik_res, _ = run_ik_for_frames(
            frame_ids,
            joint_mapping,
            target_pts_file,
            bm_fname,
            vposer_expr_dir,
            device,
        )
        ik_res = {k: v.cpu().numpy() for k, v in ik_res.items()}
        ik_res_unbatched = [
            {k: np.array([ik_res[k][i]]) for k in ik_res} for i in range(len(frame_ids))
        ]
        results.extend(ik_res_unbatched)

    print(f"\nProcessed {len(results)} frames successfully!")

    # Save results to a file
    np.savez("ik_results_parkinson.npz", results=results, joint_mapping=joint_mapping)


if __name__ == "__main__":
    main()
