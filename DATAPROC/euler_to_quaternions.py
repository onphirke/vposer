#!/usr/bin/env python3
"""
Convert Roll/Pitch/Yaw angles from IMU CSV to quaternions.
Outputs quaternions with rw, rx, ry, rz suffixes for each sensor.
"""

import argparse
import numpy as np
import pandas as pd


def euler_to_quaternion(roll, pitch, yaw, convention='ZYX', degrees=True):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).
    
    Args:
        roll: Rotation around X-axis
        pitch: Rotation around Y-axis  
        yaw: Rotation around Z-axis
        convention: Rotation order (default 'ZYX' is common for IMUs)
        degrees: If True, angles are in degrees (default), else radians
    
    Returns:
        Quaternion [w, x, y, z] (scalar-first convention)
    """
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    
    # Compute half angles
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # ZYX convention (yaw, then pitch, then roll)
    if convention == 'ZYX':
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    # XYZ convention (roll, then pitch, then yaw)
    elif convention == 'XYZ':
        w = cr * cp * cy - sr * sp * sy
        x = sr * cp * cy + cr * sp * sy
        y = cr * sp * cy - sr * cp * sy
        z = cr * cp * sy + sr * sp * cy
    # YXZ convention
    elif convention == 'YXZ':
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy + cr * sp * sy
        y = cr * sp * cy - sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
    else:
        raise ValueError(f"Unknown convention: {convention}")
    
    # Ensure w is positive (canonical form)
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    
    # Normalize
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    return np.array([w, x, y, z])


def convert_euler_to_quaternions(roll_array, pitch_array, yaw_array, convention='ZYX', degrees=True):
    """
    Convert arrays of Euler angles to quaternions.
    
    Args:
        roll_array: Array of roll angles
        pitch_array: Array of pitch angles
        yaw_array: Array of yaw angles
        convention: Euler angle rotation order
        degrees: If True, angles are in degrees
    
    Returns:
        Nx4 array of quaternions [w, x, y, z]
    """
    n_samples = len(roll_array)
    quats = np.zeros((n_samples, 4))
    
    for i in range(n_samples):
        quats[i] = euler_to_quaternion(
            roll_array[i], 
            pitch_array[i], 
            yaw_array[i], 
            convention, 
            degrees
        )
    
    return quats


def parse_time_to_seconds(val):
    """Convert time values to seconds (handles 'X sec' format)."""
    if isinstance(val, str):
        s = val.strip()
        for tok in ("seconds", "second", "secs", "sec"):
            s = s.replace(tok, "")
        s = s.strip()
        try:
            return float(s)
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser(
        description='Convert Roll/Pitch/Yaw angles to quaternions in IMU CSV'
    )
    parser.add_argument('input_csv', type=str, help='Input CSV file with IMU data')
    parser.add_argument('output_csv', type=str, help='Output CSV file for quaternions')
    parser.add_argument('--time-col', type=str, default='Time', 
                        help='Name of time column (default: Time)')
    parser.add_argument('--convention', type=str, default='ZYX',
                        choices=['ZYX', 'XYZ', 'YXZ'],
                        help='Euler angle rotation order (default: ZYX)')
    parser.add_argument('--sensors', type=str, nargs='+',
                        help='List of sensor prefixes (e.g., LowerBack R_Wrist). If not provided, auto-detect.')
    
    args = parser.parse_args()
    
    # Read CSV
    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df.columns = [c.strip() for c in df.columns]
    
    # Parse time
    if args.time_col not in df.columns:
        raise ValueError(f"Time column '{args.time_col}' not found in CSV")
    
    time = df[args.time_col].apply(parse_time_to_seconds).values
    
    # Auto-detect sensors if not provided
    if args.sensors:
        sensor_prefixes = args.sensors
    else:
        # Find all columns ending with _Roll, _Pitch, _Yaw
        roll_cols = [c for c in df.columns if c.endswith('_Roll')]
        sensor_prefixes = [c.replace('_Roll', '') for c in roll_cols]
        print(f"Auto-detected {len(sensor_prefixes)} sensors: {', '.join(sensor_prefixes)}")
    
    # Convert Euler angles to quaternions for each sensor
    output_data = {'time': time}
    
    for sensor in sensor_prefixes:
        roll_col = f'{sensor}_Roll'
        pitch_col = f'{sensor}_Pitch'
        yaw_col = f'{sensor}_Yaw'
        
        # Check if columns exist
        missing = []
        if roll_col not in df.columns:
            missing.append(roll_col)
        if pitch_col not in df.columns:
            missing.append(pitch_col)
        if yaw_col not in df.columns:
            missing.append(yaw_col)
        
        if missing:
            print(f"Warning: Skipping {sensor} - missing columns: {', '.join(missing)}")
            continue
        
        # Get Euler angles
        roll = df[roll_col].values
        pitch = df[pitch_col].values
        yaw = df[yaw_col].values
        
        # Convert to quaternions
        quats = convert_euler_to_quaternions(roll, pitch, yaw, args.convention, degrees=True)
        
        # Add to output (lowercase sensor name)
        sensor_lower = sensor.lower()
        output_data[f'{sensor_lower}_rw'] = quats[:, 0]  # w component
        output_data[f'{sensor_lower}_rx'] = quats[:, 1]  # x component
        output_data[f'{sensor_lower}_ry'] = quats[:, 2]  # y component
        output_data[f'{sensor_lower}_rz'] = quats[:, 3]  # z component
        
        print(f"  Processed {sensor}")
    
    # Write output
    out_df = pd.DataFrame(output_data)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nWrote quaternions to {args.output_csv}")
    print(f"  {len(time)} samples")
    print(f"  {len([k for k in output_data.keys() if k.endswith('_rw')])} sensors")


if __name__ == '__main__':
    main()
