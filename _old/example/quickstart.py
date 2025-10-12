#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Guide for MAIN IK Pipeline

This script helps you get started with the marker-based IK pipeline.
It provides a simple menu-driven interface to:
1. Check dependencies
2. Create example markers
3. Run test IK on orientation data
4. Visualize results

Usage:
    python quickstart.py
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "glx"

import sys
import subprocess


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70)


def print_section(text):
    """Print a formatted section."""
    print("\n" + "-"*70)
    print(text)
    print("-"*70)


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm'
    }
    
    optional = {
        'psbody.mesh': 'psbody.mesh (for interactive marker editor)',
        'pyrender': 'PyRender (for photorealistic rendering)',
        'trimesh': 'Trimesh (for mesh processing)',
        'imageio': 'imageio (for video creation)'
    }
    
    print("\nRequired dependencies:")
    all_required = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_required = False
    
    print("\nOptional dependencies:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} - not installed (optional)")
    
    if not all_required:
        print("\n⚠ Some required dependencies are missing!")
        print("Install with: pip install torch numpy pandas matplotlib tqdm")
        return False
    
    print("\n✓ All required dependencies are installed!")
    return True


def check_models():
    """Check if required models are available."""
    print_header("Checking Models")
    
    models = {
        'SMPL Body Model': '../support_data/dowloads/models/smplx/neutral/model.npz',
        'VPoser V02_05': '../_good_runs/V02_05',
        'VPoser V_me_all': '../_good_runs/V_me_all'
    }
    
    available = {}
    for name, path in models.items():
        exists = os.path.exists(path)
        available[name] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
    
    if not available['SMPL Body Model']:
        print("\n⚠ SMPL body model not found!")
        print("Please download SMPL-X model and place it in the correct location.")
        return False
    
    if not (available['VPoser V02_05'] or available['VPoser V_me_all']):
        print("\n⚠ VPoser model not found!")
        print("At least one VPoser model is required.")
        return False
    
    print("\n✓ Required models are available!")
    return True


def check_data():
    """Check if orientation data is available."""
    print_header("Checking Data")
    
    orient_file = '../_data/orients.csv'
    
    if os.path.exists(orient_file):
        print(f"  ✓ Orientation data: {orient_file}")
        
        # Try to read and show info
        try:
            import pandas as pd
            df = pd.read_csv(orient_file)
            print(f"    - {len(df)} frames")
            print(f"    - {len(df.columns)} columns")
            return True
        except Exception as e:
            print(f"  ⚠ Could not read file: {e}")
            return False
    else:
        print(f"  ✗ Orientation data not found: {orient_file}")
        print("\n  You can still run the example workflow without this file.")
        return False


def run_example():
    """Run the example workflow."""
    print_header("Running Example Workflow")
    
    print("\nThis will:")
    print("  1. Create synthetic marker definitions")
    print("  2. Generate synthetic motion")
    print("  3. Run IK optimization")
    print("  4. Evaluate results")
    print("\nThis takes about 1-2 minutes...")
    
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return
    
    print("\nRunning example_workflow.py...")
    subprocess.run([sys.executable, 'example_workflow.py'])


def run_test_ik():
    """Run test IK on real orientation data."""
    print_header("Running Test IK on Orientation Data")
    
    orient_file = '../_data/orients.csv'
    if not os.path.exists(orient_file):
        print(f"\n⚠ Orientation data not found: {orient_file}")
        print("Cannot run this test without orientation data.")
        return
    
    print("\nThis will process a single frame from the orientation data.")
    print("Takes about 30 seconds...")
    
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Skipped.")
        return
    
    print("\nRunning IK on frame 0...")
    subprocess.run([
        sys.executable, 'ik_run_orients.py',
        '--mode', 'test',
        '--frame', '0',
        '--n-iter', '100',
        '--output', 'test_ik_result.npz',
        '--verbosity', '1'
    ])
    
    print("\n✓ Test IK complete! Results saved to test_ik_result.npz")
    print("\nTo visualize:")
    print("  python visualize_animation.py --input test_ik_result.npz --output test.mp4")


def create_markers():
    """Launch the marker editor."""
    print_header("Marker Editor")
    
    print("\nThe marker editor allows you to define custom markers on the body model.")
    print("\nTwo modes available:")
    print("  1. Console mode (no graphics, works everywhere)")
    print("  2. Interactive mode (requires psbody.mesh)")
    
    try:
        import psbody.mesh
        print("\n✓ psbody.mesh is available - interactive mode is possible")
        default_mode = '2'
    except ImportError:
        print("\n○ psbody.mesh not available - only console mode available")
        default_mode = '1'
    
    mode = input(f"\nSelect mode [1=console, 2=interactive, default={default_mode}]: ").strip()
    if not mode:
        mode = default_mode
    
    if mode == '1':
        print("\nLaunching console-only marker editor...")
        subprocess.run([sys.executable, 'marker_editor.py', '--console-only'])
    else:
        print("\nLaunching interactive marker editor...")
        subprocess.run([sys.executable, 'marker_editor.py'])


def visualize_results():
    """Visualize existing results."""
    print_header("Visualize Results")
    
    # Find NPZ files
    npz_files = [f for f in os.listdir('.') if f.endswith('.npz') and 'result' in f.lower()]
    
    if not npz_files:
        print("\n⚠ No result files found in current directory.")
        print("Run IK optimization first to generate results.")
        return
    
    print("\nAvailable result files:")
    for i, f in enumerate(npz_files, 1):
        print(f"  {i}. {f}")
    
    choice = input(f"\nSelect file [1-{len(npz_files)}]: ").strip()
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(npz_files):
            raise ValueError()
        input_file = npz_files[idx]
    except:
        print("Invalid choice.")
        return
    
    output_file = input("Output filename (e.g., animation.mp4): ").strip()
    if not output_file:
        output_file = 'animation.mp4'
    
    print(f"\nRendering {input_file} to {output_file}...")
    subprocess.run([
        sys.executable, 'visualize_animation.py',
        '--input', input_file,
        '--output', output_file,
        '--fps', '30'
    ])


def show_menu():
    """Show the main menu."""
    while True:
        print_header("MAIN IK Pipeline - Quick Start")
        
        print("\nOptions:")
        print("  1. Check dependencies and models")
        print("  2. Run example workflow (synthetic data)")
        print("  3. Create/edit markers")
        print("  4. Test IK on orientation data")
        print("  5. Visualize results")
        print("  6. View README documentation")
        print("  0. Exit")
        
        choice = input("\nSelect option [0-6]: ").strip()
        
        if choice == '0':
            print("\nGoodbye!")
            break
        elif choice == '1':
            check_dependencies()
            check_models()
            check_data()
        elif choice == '2':
            run_example()
        elif choice == '3':
            create_markers()
        elif choice == '4':
            run_test_ik()
        elif choice == '5':
            visualize_results()
        elif choice == '6':
            print_header("README Documentation")
            if os.path.exists('README.md'):
                with open('README.md', 'r') as f:
                    print(f.read())
            else:
                print("README.md not found.")
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n" + "="*70)
    print("MAIN IK Pipeline - Quick Start")
    print("="*70)
    print("\nWelcome! This wizard will help you get started with the")
    print("marker-based inverse kinematics pipeline.")
    
    # Quick check
    print("\nRunning quick check...")
    deps_ok = check_dependencies()
    models_ok = check_models()
    
    if not (deps_ok and models_ok):
        print("\n⚠ Setup incomplete. Please install missing dependencies and models.")
        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print("Exiting.")
            return
    
    # Show menu
    show_menu()


if __name__ == '__main__':
    main()
