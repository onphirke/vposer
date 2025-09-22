#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for Inverse Kinematics Results from Parkinson's Patient Motion Data

This script loads the saved IK results and creates a video visualization 
showing the reconstructed body motion over time.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path as osp
from typing import List, Tuple, Dict

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Try to import visualization tools - these may not be available
try:
    from body_visualizer.tools.vis_tools import render_smpl_params
    from body_visualizer.tools.vis_tools import imagearray2file
    BODY_VISUALIZER_AVAILABLE = True
except ImportError:
    BODY_VISUALIZER_AVAILABLE = False
    print("Warning: body_visualizer not available. Using basic matplotlib visualization.")

# Try to import video encoding tools
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def setup_paths() -> Tuple[str, str, str]:
    """
    Setup file paths for models and results.
    
    Returns:
        Tuple of (bm_fname, results_file, output_dir)
    """
    support_dir = "../support_data/dowloads"
    bm_fname = osp.join(support_dir, "models/smplx/neutral/model.npz")
    results_file = "ik_results_parkinson.npz"
    output_dir = "visualization_output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return bm_fname, results_file, output_dir


def load_ik_results(results_file: str) -> Tuple[List[Tuple[Dict, float]], Dict]:
    """
    Load the saved inverse kinematics results.
    
    Args:
        results_file: Path to the results NPZ file
        
    Returns:
        Tuple of (results_list, joint_mapping)
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    data = np.load(results_file, allow_pickle=True)
    results = data['results']
    joint_mapping = data['joint_mapping'].item() if 'joint_mapping' in data else {}
    
    print(f"Loaded {len(results)} frames of IK results")
    print(f"Joint mapping: {joint_mapping}")
    
    return results, joint_mapping


def create_body_meshes(results: List[Tuple[Dict, float]], bm: BodyModel, device: torch.device) -> List[Dict]:
    """
    Generate body meshes from IK results using the SMPL model.
    
    Args:
        results: List of (ik_params, loss) tuples
        bm: Body model instance
        device: Torch device
        
    Returns:
        List of body mesh dictionaries
    """
    body_meshes = []

    for i, ik_params in enumerate(results):
        print(f"Generating mesh for frame {i+1}/{len(results)}")

        # Convert numpy arrays back to tensors
        body_params = {}
        for key, value in ik_params.items():
            if isinstance(value, np.ndarray):
                body_params[key] = torch.from_numpy(value).float().to(device)
            else:
                body_params[key] = value
        
        # Generate body mesh
        with torch.no_grad():
            body_output = bm(**body_params)
            
        # Store mesh data
        mesh_data = {
            'vertices': c2c(body_output.v)[0],  # Shape: [6890, 3]
            'faces': c2c(body_output.f),        # Face indices
            'joints': c2c(body_output.Jtr)[0],  # Joint positions [24, 3]
            'params': ik_params
        }
        
        body_meshes.append(mesh_data)
    
    return body_meshes


def create_joint_video_matplotlib(body_meshes: List[Dict], output_path: str, joint_mapping: Dict) -> None:
    """
    Create a video showing joint trajectories using matplotlib.
    
    Args:
        body_meshes: List of body mesh data
        output_path: Output video file path
        joint_mapping: Joint name to index mapping
    """
    print("Creating joint trajectory video with matplotlib...")
    
    # Extract joint positions for all frames
    n_frames = len(body_meshes)
    n_joints = body_meshes[0]['joints'].shape[0]
    
    # Stack all joint positions [n_frames, n_joints, 3]
    all_joints = np.array([mesh['joints'] for mesh in body_meshes])

    # switch y and z for visualization
    all_joints = all_joints[:, :, [0, 2, 1]]
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # SMPL skeleton connections (parent-child relationships)
    skeleton_connections = [
        (0, 1), (0, 2), (0, 3),        # pelvis to hips and spine
        (1, 4), (2, 5),                # hips to knees
        (4, 7), (5, 8),                # knees to ankles
        (7, 10), (8, 11),              # ankles to feet
        (3, 6), (6, 9),                # spine chain
        (9, 12),                       # spine to neck
        (12, 15),                      # neck to head
        (9, 13), (9, 14),              # spine to collars
        (13, 16), (14, 17),            # collars to shoulders
        (16, 18), (17, 19),            # shoulders to elbows
        (18, 20), (19, 21),            # elbows to wrists
        (20, 22), (21, 23),            # wrists to hands
    ]
    
    # Set up the animation function
    def animate(frame_idx):
        ax.clear()
        
        # Get joint positions for current frame
        joints = all_joints[frame_idx]
        
        # Plot joint points
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                  c='red', s=50, alpha=0.8, label='Joints')
        
        # Plot skeleton connections
        for connection in skeleton_connections:
            start_idx, end_idx = connection
            if start_idx < len(joints) and end_idx < len(joints):
                ax.plot([joints[start_idx, 0], joints[end_idx, 0]],
                       [joints[start_idx, 1], joints[end_idx, 1]],
                       [joints[start_idx, 2], joints[end_idx, 2]], 
                       'b-', linewidth=2, alpha=0.7)
        
        # Plot trajectory for key joints (show last 10 frames)
        start_frame = max(0, frame_idx - 10)
        for joint_idx in [0, 12, 15, 20, 21]:  # pelvis, neck, head, wrists
            if joint_idx < n_joints:
                trajectory = all_joints[start_frame:frame_idx+1, joint_idx]
                if len(trajectory) > 1:
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                           'g--', alpha=0.5, linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(f'Parkinson\'s Patient Motion - Frame {frame_idx+1}/{n_frames}')
        
        # Set consistent axis limits
        all_coords = all_joints.reshape(-1, 3)
        margin = 0.2
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
        ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)
        
        # Set viewing angle
        ax.view_init(elev=10, azim=frame_idx * 2)  # Slowly rotate view
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                 interval=200, repeat=True, blit=False)
    
    # Save animation
    print(f"Saving video to {output_path}...")
    anim.save(output_path, writer='pillow', fps=5)
    print("Video saved successfully!")
    
    plt.close()


def create_mesh_video_advanced(body_meshes: List[Dict], bm: BodyModel, output_path: str, device: torch.device) -> None:
    """
    Create a high-quality mesh rendering video (requires body_visualizer).
    
    Args:
        body_meshes: List of body mesh data
        bm: Body model instance
        output_path: Output video file path
        device: Torch device for computation
    """
    if not BODY_VISUALIZER_AVAILABLE:
        print("body_visualizer not available. Skipping advanced mesh rendering.")
        return
    
    print("Creating high-quality mesh video...")
    
    try:
        # Prepare all body parameters on the correct device
        all_body_params = {}
        param_keys = list(body_meshes[0]['params'].keys())
        
        print(f"Processing {len(param_keys)} parameter types: {param_keys}")
        
        for key in param_keys:
            param_values = []
            for i, mesh in enumerate(body_meshes):
                # Convert numpy to tensor and ensure correct device
                param_tensor = torch.from_numpy(mesh['params'][key]).float().to(device)
                param_values.append(param_tensor)
                print(f"Frame {i+1}, {key}: shape {param_tensor.shape}, device {param_tensor.device}")
            
            # Stack all frames for this parameter
            all_body_params[key] = torch.stack(param_values)
            print(f"Stacked {key}: shape {all_body_params[key].shape}")
        
        # Ensure body model is on correct device
        bm = bm.to(device)
        
        # Render all frames
        print("Rendering frames...")
        
        # Render frames individually to avoid memory issues
        rendered_images = []
        for frame_idx in range(len(body_meshes)):
            # Extract single frame parameters and ensure correct dimensionality
            frame_params = {}
            for key, val in all_body_params.items():
                # Extract single frame and ensure it has batch dimension
                frame_tensor = val[frame_idx]  # Remove frame dimension
                if frame_tensor.dim() == 1:
                    frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension [1, ...]
                elif frame_tensor.dim() == 0:
                    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and feature dim [1, 1]
                
                frame_params[key] = frame_tensor
                
            print(f"Rendering frame {frame_idx+1}/{len(body_meshes)}")
            print(f"  Frame params shapes: {[(k, v.shape) for k, v in frame_params.items()]}")
            
            # Render single frame
            frame_image = render_smpl_params(bm, frame_params, rot_body=[0, 180, 0])
            
            # Convert to numpy and store
            if isinstance(frame_image, torch.Tensor):
                frame_image = frame_image.cpu().numpy()
            
            rendered_images.append(frame_image)
        
        print(f"Successfully rendered {len(rendered_images)} frames")
        
        FPS = 5  # Frames per second for video
        
        # Save as video using imageio
        if IMAGEIO_AVAILABLE:
            # Stack and reshape images for video
            video_frames = []
            for img in rendered_images:
                if img.ndim == 5:  # [1, 1, 1, H, W, 3]
                    img = img[0, 0, 0]  # Remove batch dimensions
                elif img.ndim == 4:  # [1, H, W, 3]
                    img = img[0]
                
                # Ensure image is in correct format [H, W, 3]
                if img.shape[-1] == 3:
                    # Convert to uint8 if needed
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)
                    video_frames.append(img)
            
            if video_frames:
                print(f"Saving video to {output_path}...")
                if output_path.lower().endswith('.mp4'):
                    # MP4 format doesn't support loop parameter
                    imageio.mimsave(output_path, video_frames, fps=FPS)
                else:
                    # GIF format supports loop parameter
                    imageio.mimsave(output_path, video_frames, fps=FPS, loop=0)
                print("Advanced mesh video saved successfully!")
            else:
                print("No valid frames to save")
        else:
            # Fallback: save individual frames
            print("imageio not available, saving individual frames...")
            for i, img in enumerate(rendered_images):
                frame_path = output_path.replace('.mp4', f'_frame_{i:03d}.png')
                if img.ndim == 5:
                    img = img[0, 0, 0]
                elif img.ndim == 4:
                    img = img[0]
                
                # Save using matplotlib
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(frame_path, bbox_inches='tight', dpi=150)
                plt.close()
            
            print(f"Saved {len(rendered_images)} individual frame images")
        
    except Exception as e:
        print(f"Advanced rendering failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def analyze_joint_movement(body_meshes: List[Dict], joint_mapping: Dict, output_dir: str) -> None:
    """
    Analyze and visualize joint movement patterns.
    
    Args:
        body_meshes: List of body mesh data  
        joint_mapping: Joint name to index mapping
        output_dir: Output directory
    """
    print("Analyzing joint movement patterns...")
    
    # Extract joint positions
    all_joints = np.array([mesh['joints'] for mesh in body_meshes])
    
    # Calculate joint displacement over time
    joint_velocities = np.diff(all_joints, axis=0)
    joint_speeds = np.linalg.norm(joint_velocities, axis=2)  # [n_frames-1, n_joints]
    
    # Plot joint speed over time for key joints
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Key joints to analyze
    key_joints = {
        'Head': 15,
        'Left Wrist': 20, 
        'Right Wrist': 21,
        'Pelvis': 0
    }
    
    for i, (joint_name, joint_idx) in enumerate(key_joints.items()):
        if i < len(axes) and joint_idx < joint_speeds.shape[1]:
            axes[i].plot(joint_speeds[:, joint_idx], linewidth=2)
            axes[i].set_title(f'{joint_name} Speed Over Time')
            axes[i].set_xlabel('Frame')
            axes[i].set_ylabel('Speed (m/frame)')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    analysis_path = osp.join(output_dir, 'joint_movement_analysis.png')
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"Joint movement analysis saved to: {analysis_path}")
    plt.close()
    
    # Calculate and print movement statistics
    mean_speeds = np.mean(joint_speeds, axis=0)
    print("\nJoint Movement Statistics:")
    print("-" * 40)
    for joint_name, joint_idx in key_joints.items():
        if joint_idx < len(mean_speeds):
            print(f"{joint_name:12}: {mean_speeds[joint_idx]:.6f} m/frame")


def main():
    """
    Main function to create visualizations from saved IK results.
    """
    print("Starting IK Results Visualization...")
    
    # Setup paths
    bm_fname, results_file, output_dir = setup_paths()
    
    # Load results
    results, joint_mapping = load_ik_results(results_file)
    
    if len(results) == 0:
        print("No results found to visualize!")
        return
    
    # Setup device and body model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    bm = BodyModel(bm_fname=bm_fname, persistant_buffer=False).to(device)
    
    # Generate body meshes
    print("Generating body meshes...")
    body_meshes = create_body_meshes(results, bm, device)
    
    # Create visualizations
    video_path = osp.join(output_dir, 'parkinson_motion_joints.gif')
    create_joint_video_matplotlib(body_meshes, video_path, joint_mapping)
    
    # Analyze joint movement
    analyze_joint_movement(body_meshes, joint_mapping, output_dir)
    
    # Try advanced mesh rendering if available
    if BODY_VISUALIZER_AVAILABLE:
        mesh_video_path = osp.join(output_dir, 'parkinson_motion_mesh.mp4')
        create_mesh_video_advanced(body_meshes, bm, mesh_video_path, device)
    
    print(f"\nVisualization complete! Check the '{output_dir}' directory for outputs:")
    print(f"  - Joint motion video: {video_path}")
    print(f"  - Loss trajectory: {osp.join(output_dir, 'loss_trajectory.png')}")
    print(f"  - Movement analysis: {osp.join(output_dir, 'joint_movement_analysis.png')}")
    if BODY_VISUALIZER_AVAILABLE:
        print(f"  - Advanced mesh video: {osp.join(output_dir, 'parkinson_motion_mesh.mp4')}")


if __name__ == "__main__":
    main()
