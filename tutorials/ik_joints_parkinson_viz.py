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
import cv2
from os import path as osp
from typing import List, Tuple, Dict

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Configuration
DEFAULT_BATCH_SIZE = 128  # Adjust based on GPU memory
MAX_BATCH_SIZE = DEFAULT_BATCH_SIZE     # Maximum batch size to prevent memory issues

# Try to import visualization tools - these may not be available
try:
    from body_visualizer.tools.vis_tools import render_smpl_params
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


def determine_optimal_batch_size(device: torch.device, n_frames: int) -> int:
    """
    Determine optimal batch size based on available GPU memory and number of frames.
    
    Args:
        device: Torch device
        n_frames: Total number of frames to render
        
    Returns:
        Optimal batch size
    """
    
    # if device.type == 'cpu':
    #     return min(DEFAULT_BATCH_SIZE, n_frames)
    
    # try:
    #     # Get GPU memory info
    #     if hasattr(torch.cuda, 'get_device_properties'):
    #         gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
    #         # Estimate batch size based on memory
    #         # Rough estimate: each frame needs ~50MB for rendering
    #         estimated_batch_size = max(1, int(gpu_memory_gb * 0.3 / 0.05))  # Use 30% of memory
    #         batch_size = min(estimated_batch_size, MAX_BATCH_SIZE, n_frames)
    #     else:
    #         batch_size = min(DEFAULT_BATCH_SIZE, n_frames)
            
    #     print(f"Determined optimal batch size: {batch_size}")
    #     return batch_size
        
    # except Exception:
    #     return min(DEFAULT_BATCH_SIZE, n_frames)
    
    return min(DEFAULT_BATCH_SIZE, n_frames)


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


def create_mesh_video_advanced(body_meshes: List[Dict], bm: BodyModel, output_path: str, device: torch.device, batch_size: int = 8) -> None:
    """
    Create a high-quality mesh rendering video using batch rendering (requires body_visualizer).
    
    Args:
        body_meshes: List of body mesh data
        bm: Body model instance
        output_path: Output video file path
        device: Torch device for computation
        batch_size: Number of frames to render in each batch
    """
    if not BODY_VISUALIZER_AVAILABLE:
        print("body_visualizer not available. Skipping advanced mesh rendering.")
        return
    
    print(f"Creating high-quality mesh video with batch rendering (batch_size={batch_size})...")
    
    try:
        # Prepare all body parameters on the correct device
        all_body_params = {}
        param_keys = list(body_meshes[0]['params'].keys())
        
        print(f"Processing {len(param_keys)} parameter types: {param_keys}")
        
        for key in param_keys:
            param_values = []
            for mesh in body_meshes:
                # Convert numpy to tensor and ensure correct device
                param_tensor = torch.from_numpy(mesh['params'][key]).float().to(device)
                # Ensure single frame has batch dimension
                if param_tensor.dim() == 1:
                    param_tensor = param_tensor.unsqueeze(0)
                elif param_tensor.dim() == 0:
                    param_tensor = param_tensor.unsqueeze(0)
                param_values.append(param_tensor)
            
            # Stack all frames for this parameter [n_frames, ...]
            all_body_params[key] = torch.cat(param_values, dim=0)
            print(f"Prepared {key}: shape {all_body_params[key].shape}")
        
        # Ensure body model is on correct device
        bm = bm.to(device)
        
        # Initialize video writer for streaming
        FPS = 30  # Frames per second for video
        # Get image dimensions from first render to initialize video writer
        print("Initializing video writer...")
        
        # Render first frame to get dimensions
        first_batch_params = {}
        for key, val in all_body_params.items():
            first_batch_params[key] = val[0:1]  # Just first frame

        rotation = [-110, 160, 0]

        first_frame = render_smpl_params(bm, first_batch_params, rot_body=rotation)
        if isinstance(first_frame, torch.Tensor):
            first_frame = first_frame.cpu().numpy()
        
        # Get image dimensions
        if first_frame.ndim == 4:  # [1, H, W, 3]
            img_h, img_w = first_frame.shape[1], first_frame.shape[2]
            sample_frame = first_frame[0]
        else:
            img_h, img_w = first_frame.shape[0], first_frame.shape[1]
            sample_frame = first_frame
        
        print(f"Video dimensions: {img_w}x{img_h}")
        
        # Initialize video writer
        video_writer = None
        if output_path.lower().endswith('.mp4'):
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (img_w, img_h))
            
            # Write the first frame
            frame = sample_frame
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            print("Video writer initialized successfully")
        
        # Render remaining frames in batches and stream to file
        print("Rendering and streaming frames in batches...")
        n_frames = len(body_meshes)
        n_batches = (n_frames + batch_size - 1) // batch_size
        frames_written = 1  # Already wrote first frame
        
        for batch_idx, batch_start in enumerate(range(batch_size, n_frames, batch_size)):
            batch_end = min(batch_start + batch_size, n_frames)
            current_batch_size = batch_end - batch_start
            
            print(f"Rendering batch {batch_idx + 2}/{n_batches}: frames {batch_start+1}-{batch_end} ({current_batch_size} frames)")
            
            # Extract batch parameters
            batch_params = {}
            for key, val in all_body_params.items():
                batch_params[key] = val[batch_start:batch_end]
            
            # Render entire batch at once
            batch_images = render_smpl_params(bm, batch_params, rot_body=rotation)
            
            # Convert to numpy if needed
            if isinstance(batch_images, torch.Tensor):
                batch_images = batch_images.cpu().numpy()
            
            # Stream frames directly to video file
            if batch_images.ndim == 4:  # [batch_size, H, W, 3]
                for i in range(current_batch_size):
                    frame = batch_images[i]
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    if video_writer is not None:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                        frames_written += 1
            else:
                print(f"Unexpected batch image shape: {batch_images.shape}")
            
            # Clear memory after each batch
            del batch_images
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            print(f"  Completed batch {batch_idx + 2}/{n_batches} ({frames_written} total frames written)")
        
        # Close video writer
        if video_writer is not None:
            video_writer.release()
            print(f"Successfully streamed {frames_written} frames to video file")
        else:
            # Fallback for non-MP4 formats - use imageio but still stream
            print("Using imageio fallback for non-MP4 format...")
            if IMAGEIO_AVAILABLE:
                with imageio.get_writer(output_path, mode='I', fps=FPS) as writer:
                    # Re-render frames and write directly to avoid memory accumulation
                    for batch_idx, batch_start in enumerate(range(0, n_frames, batch_size)):
                        batch_end = min(batch_start + batch_size, n_frames)
                        
                        batch_params = {}
                        for key, val in all_body_params.items():
                            batch_params[key] = val[batch_start:batch_end]
                        
                        batch_images = render_smpl_params(bm, batch_params, rot_body=[-90,0,0])
                        if isinstance(batch_images, torch.Tensor):
                            batch_images = batch_images.cpu().numpy()
                        
                        if batch_images.ndim == 4:
                            for i in range(batch_end - batch_start):
                                frame = batch_images[i]
                                if frame.dtype != np.uint8:
                                    frame = (frame * 255).astype(np.uint8)
                                writer.append_data(frame)
                        
                        del batch_images
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        print(f"Streamed batch {batch_idx + 1}/{n_batches}")
                
                print("Video saved successfully with imageio!")
            else:
                print("Neither OpenCV nor imageio available for video writing")
        
        print("Advanced mesh video created with streaming - minimal memory usage!")
        
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
    # video_path = osp.join(output_dir, 'parkinson_motion_joints.gif')
    # create_joint_video_matplotlib(body_meshes, video_path, joint_mapping)
    
    # Analyze joint movement
    analyze_joint_movement(body_meshes, joint_mapping, output_dir)
    
    # Try advanced mesh rendering if available
    if BODY_VISUALIZER_AVAILABLE:
        mesh_video_path = osp.join(output_dir, 'parkinson_motion_mesh.mp4')
        # Determine optimal batch size
        batch_size = determine_optimal_batch_size(device, len(body_meshes))
        create_mesh_video_advanced(body_meshes, bm, mesh_video_path, device, batch_size)
    
    print(f"\nVisualization complete! Check the '{output_dir}' directory for outputs:")
    # print(f"  - Joint motion video: {video_path}")
    print(f"  - Loss trajectory: {osp.join(output_dir, 'loss_trajectory.png')}")
    print(f"  - Movement analysis: {osp.join(output_dir, 'joint_movement_analysis.png')}")
    if BODY_VISUALIZER_AVAILABLE:
        print(f"  - Advanced mesh video: {osp.join(output_dir, 'parkinson_motion_mesh.mp4')}")


if __name__ == "__main__":
    main()
