#!/usr/bin/env python3
"""
NPZ File Inspector - A tool to analyze and inspect NumPy .npz files.
"""
import argparse
import sys
import numpy as np
from pathlib import Path



def load_npz_file(filepath):
    """Load and return the contents of an NPZ file."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        sys.exit(1)


def print_file_info(data, filepath):
    """Print basic information about the NPZ file."""
    print(f"\nFile: {filepath}")
    print(f"Number of arrays: {len(data.files)}")
    print(f"Array names: {list(data.files)}")
    print("-" * 50)
    
    for name in data.files:
        array = data[name]
        print(f"Array '{name}':")
        print(f"  Shape: {array.shape}")
        print(f"  Dtype: {array.dtype}")
        print(f"  Size: {array.size}")
        print(f"  Memory usage: {array.nbytes / 1024:.2f} KB")
        if array.size > 0 and np.issubdtype(array.dtype, np.number):
            print(f"  Min: {np.min(array)}")
            print(f"  Max: {np.max(array)}")
            print(f"  Mean: {np.mean(array):.4f}")
        elif array.size > 0:
            print(f"  Sample values: {array.flat[:3]}")
        print()


def interactive_mode(data):
    """Enter interactive mode for exploring the NPZ file."""
    print("\nInteractive mode - Available commands:")
    print("  list - Show all array names")
    print("  info <name> - Show detailed info for array")
    print("  show <name> - Display array contents")
    print("  shape <name> - Show array shape")
    print("  stats <name> - Show array statistics")
    print("  exit/quit - Exit interactive mode")
    print()
    
    while True:
        try:
            command = input("npz> ").strip().split()
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd in ['exit', 'quit']:
                break
            elif cmd == 'list':
                print(f"Arrays: {list(data.files)}")
            elif cmd == 'info' and len(command) > 1:
                name = command[1]
                if name in data.files:
                    array = data[name]
                    print(f"Array '{name}':")
                    print(f"  Shape: {array.shape}")
                    print(f"  Dtype: {array.dtype}")
                    print(f"  Size: {array.size}")
                    print(f"  Memory: {array.nbytes / 1024:.2f} KB")
                else:
                    print(f"Array '{name}' not found")
            elif cmd == 'show' and len(command) > 1:
                name = command[1]
                if name in data.files:
                    array = data[name]
                    if array.size <= 100:  # Only show small arrays
                        print(f"Array '{name}':\n{array}")
                    else:
                        print(f"Array '{name}' is too large to display ({array.size} elements)")
                        print(f"First 10 elements: {array.flat[:10]}")
                else:
                    print(f"Array '{name}' not found")
            elif cmd == 'shape' and len(command) > 1:
                name = command[1]
                if name in data.files:
                    print(f"Shape of '{name}': {data[name].shape}")
                else:
                    print(f"Array '{name}' not found")
            elif cmd == 'stats' and len(command) > 1:
                name = command[1]
                if name in data.files:
                    array = data[name]
                    if array.size > 0:
                        if np.issubdtype(array.dtype, np.number):
                            print(f"Statistics for '{name}':")
                            print(f"  Min: {np.min(array)}")
                            print(f"  Max: {np.max(array)}")
                            print(f"  Mean: {np.mean(array):.4f}")
                            print(f"  Std: {np.std(array):.4f}")
                        else:
                            print(f"Array '{name}' is not numeric (dtype: {array.dtype})")
                            print(f"  Sample values: {array.flat[:5]}")
                    else:
                        print(f"Array '{name}' is empty")
                else:
                    print(f"Array '{name}' not found")
            else:
                print("Unknown command or missing argument")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze and inspect NumPy .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'file',
        type=str,
        help='Path to the .npz file to analyze'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Enter interactive mode for detailed exploration'
    )
    
    args = parser.parse_args()
    
    # Validate file path
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File '{filepath}' does not exist")
        sys.exit(1)
    
    if not filepath.suffix.lower() == '.npz':
        print(f"Warning: File '{filepath}' does not have .npz extension")
    
    # Load and analyze the file
    data = load_npz_file(filepath)
    print_file_info(data, filepath)
    
    # Enter interactive mode if requested
    if args.interactive:
        interactive_mode(data)
    
    data.close()


if __name__ == "__main__":
    main()