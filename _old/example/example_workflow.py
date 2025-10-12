#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example workflow demonstrating the marker-based IK pipeline.

This script shows how to:
1. Create synthetic marker data
2. Define markers programmatically
3. Run IK optimization
4. Visualize results

Usage:
    python example_workflow.py
"""

import os
import sys
import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from human_body_prior.body_model.body_model import BodyModel


def create_example_markers():
    """Create example marker definitions."""
    
    # Define some key body vertices as markers
    # These are approximate indices for common marker locations
    marker_definitions = {
        'head_top': 411,      # Top of head
        'neck': 3050,         # Neck
        'left_shoulder': 3023,   # Left shoulder
        'right_shoulder': 6470,  # Right shoulder
        'left_elbow': 1666,   # Left elbow
        'right_elbow': 5111,  # Right elbow
        'left_wrist': 2112,   # Left wrist
        'right_wrist': 5559,  # Right wrist
        'chest': 3500,        # Chest center
        'pelvis': 3143,       # Pelvis
    }
    
    marker_names = list(marker_definitions.keys())
    marker_indices = np.array([marker_definitions[name] for name in marker_names], dtype=np.int32)
    
    return marker_names, marker_indices


def generate_synthetic_motion(n_frames=50):
    """
    Generate synthetic motion data for testing.
    
    Args:
        n_frames: Number of frames to generate
    
    Returns:
        Dictionary of body parameters
    """
    print(f"Generating {n_frames} frames of synthetic motion...")
    
    # Simple animation: raise arms over time
    t = np.linspace(0, 2*np.pi, n_frames)
    
    betas = np.zeros((n_frames, 10))  # Neutral shape
    pose_body = np.zeros((n_frames, 63))
    
    # Animate left arm (joint 16 = left shoulder)
    # Joint indices in pose_body: joint_i * 3 to joint_i * 3 + 3
    left_shoulder_idx = 16 * 3
    pose_body[:, left_shoulder_idx] = np.sin(t) * 1.5  # Rotate around X axis
    
    # Animate right arm (joint 17 = right shoulder)
    right_shoulder_idx = 17 * 3
    pose_body[:, right_shoulder_idx] = np.sin(t) * 1.5
    
    root_orient = np.zeros((n_frames, 3))
    trans = np.zeros((n_frames, 3))
    
    return {
        'betas': betas,
        'pose_body': pose_body,
        'root_orient': root_orient,
        'trans': trans
    }


def extract_marker_positions(body_params, marker_indices, body_model_path):
    """
    Extract marker positions from body parameters.
    
    Args:
        body_params: Dictionary of body parameters
        marker_indices: Array of marker vertex indices
        body_model_path: Path to body model
    
    Returns:
        Array of marker positions (T, N, 3)
    """
    print("Extracting marker positions from body model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm = BodyModel(body_model_path, num_betas=10, num_dmpls=None).to(device)
    
    n_frames = body_params['betas'].shape[0]
    n_markers = len(marker_indices)
    marker_positions = np.zeros((n_frames, n_markers, 3))
    
    with torch.no_grad():
        for i in range(n_frames):
            body_output = bm(
                betas=torch.tensor(body_params['betas'][i:i+1], dtype=torch.float32, device=device),
                pose_body=torch.tensor(body_params['pose_body'][i:i+1], dtype=torch.float32, device=device),
                root_orient=torch.tensor(body_params['root_orient'][i:i+1], dtype=torch.float32, device=device),
                trans=torch.tensor(body_params['trans'][i:i+1], dtype=torch.float32, device=device)
            )
            
            vertices = body_output.v.cpu().numpy()[0]
            marker_positions[i] = vertices[marker_indices]
    
    return marker_positions


def run_example_workflow():
    """Run the complete example workflow."""
    
    print("="*70)
    print("Example Marker-Based IK Workflow")
    print("="*70)
    
    # Paths
    body_model_path = '../support_data/dowloads/models/smplx/neutral/model.npz'
    vposer_path = '../_good_runs/V02_05'
    marker_file = 'example_markers.npz'
    results_file = 'example_results.npz'
    
    # Check if body model exists
    if not os.path.exists(body_model_path):
        print(f"\nError: Body model not found at {body_model_path}")
        print("Please update the path to your SMPL model.")
        return
    
    # Step 1: Create marker definitions
    print("\n" + "-"*70)
    print("Step 1: Creating marker definitions")
    print("-"*70)
    
    marker_names, marker_indices = create_example_markers()
    print(f"Created {len(marker_names)} markers:")
    for name, idx in zip(marker_names, marker_indices):
        print(f"  {name}: vertex {idx}")
    
    # Save marker definitions
    # First, get marker positions from default pose
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm = BodyModel(body_model_path, num_betas=10, num_dmpls=None).to(device)
    
    with torch.no_grad():
        default_body = bm(
            betas=torch.zeros(1, 10, dtype=torch.float32, device=device),
            pose_body=torch.zeros(1, 63, dtype=torch.float32, device=device),
            root_orient=torch.zeros(1, 3, dtype=torch.float32, device=device),
            trans=torch.zeros(1, 3, dtype=torch.float32, device=device)
        )
        vertices = default_body.v.cpu().numpy()[0]
        marker_positions = vertices[marker_indices]
    
    np.savez(
        marker_file,
        marker_names=marker_names,
        marker_indices=marker_indices,
        marker_positions=marker_positions,
        model_path=body_model_path
    )
    print(f"\nSaved marker definitions to {marker_file}")
    
    # Step 2: Generate synthetic motion
    print("\n" + "-"*70)
    print("Step 2: Generating synthetic motion")
    print("-"*70)
    
    body_params_gt = generate_synthetic_motion(n_frames=30)
    
    # Step 3: Extract marker positions (simulating mocap data)
    print("\n" + "-"*70)
    print("Step 3: Extracting marker positions (simulating mocap)")
    print("-"*70)
    
    target_markers = extract_marker_positions(body_params_gt, marker_indices, body_model_path)
    print(f"Extracted marker positions: {target_markers.shape}")
    
    # Add some noise to simulate real mocap data
    noise_level = 0.005  # 5mm noise
    target_markers += np.random.randn(*target_markers.shape) * noise_level
    print(f"Added Gaussian noise (σ={noise_level*1000:.1f}mm)")
    
    # Step 4: Run IK optimization
    print("\n" + "-"*70)
    print("Step 4: Running IK optimization")
    print("-"*70)
    
    try:
        from ik_marker_engine import MarkerIKEngine
        
        engine = MarkerIKEngine(
            vposer_model_path=vposer_path,
            body_model_path=body_model_path,
            marker_file=marker_file,
            use_vposer=True
        )
        
        print("\nOptimizing body parameters to fit markers...")
        result = engine.fit(
            target_markers=target_markers,
            n_iter=100,
            lr=0.01,
            fix_pelvis=True,
            verbosity=1
        )
        
        print("\nOptimization complete!")
        
        # Step 5: Save results
        print("\n" + "-"*70)
        print("Step 5: Saving results")
        print("-"*70)
        
        np.savez(
            results_file,
            body_params=result,
            target_markers=target_markers,
            marker_names=marker_names
        )
        print(f"Saved results to {results_file}")
        
        # Step 6: Compute reconstruction error
        print("\n" + "-"*70)
        print("Step 6: Evaluation")
        print("-"*70)
        
        # Re-extract markers from fitted body
        fitted_markers = extract_marker_positions(result, marker_indices, body_model_path)
        
        # Compute per-marker error
        errors = np.linalg.norm(fitted_markers - target_markers, axis=-1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"\nReconstruction Error:")
        print(f"  Mean: {mean_error*1000:.2f}mm")
        print(f"  Max:  {max_error*1000:.2f}mm")
        print(f"  Std:  {np.std(errors)*1000:.2f}mm")
        
        print("\nPer-marker error (mm):")
        for i, name in enumerate(marker_names):
            marker_error = np.mean(errors[:, i])
            print(f"  {name:20s}: {marker_error*1000:.2f}mm")
        
        # Step 7: Suggest visualization
        print("\n" + "-"*70)
        print("Step 7: Visualization")
        print("-"*70)
        
        print("\nTo visualize the results, run:")
        print(f"  python visualize_animation.py --input {results_file} --output example_animation.mp4")
        print(f"  python visualize_animation.py --input {results_file} --output example_animation.gif --fps 10")
        
    except Exception as e:
        print(f"\nError during IK optimization: {e}")
        print("Make sure VPoser model is available and dependencies are installed.")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Example workflow complete!")
    print("="*70)


if __name__ == '__main__':
    run_example_workflow()
