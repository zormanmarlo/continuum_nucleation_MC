#!/usr/bin/env python3
"""
Lightweight XYZ energy testing driver.

Reads XYZ trajectory files, calculates energies using PMF, and outputs results.
Supports single-frame and multi-frame trajectories.

Usage:
    python test_energy.py <xyz_file> <ff_path> <output_file>

Example:
    python test_energy.py trajectory.xyz potentials/PR_NaCl.txt energies.dat
"""

import sys
import numpy as np
from config import Config
from system import System


def parse_xyz_file(filename):
    """
    Parse XYZ file and extract frames with positions and types.

    Handles both standard XYZ format and non-standard format where
    all frames are concatenated without individual headers.

    Returns:
        frames: list of dicts, each containing:
            - 'positions': np.array of shape (N, 3)
            - 'types': np.array of shape (N,)
            - 'box_length': float
            - 'num_atoms': int
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    frames = []
    i = 0

    # Parse first frame header
    try:
        num_atoms_per_frame = int(lines[0].strip())
    except (ValueError, IndexError):
        print("Error: Cannot parse number of atoms from first line")
        return frames

    comment = lines[1].strip()
    box_length = parse_box_length(comment)

    # Count total atom lines
    total_lines = len(lines) - 2  # Exclude header lines

    # Check if this is a multi-frame trajectory without individual headers
    if total_lines > num_atoms_per_frame and total_lines % num_atoms_per_frame == 0:
        # Special format: one header for all frames
        num_frames = total_lines // num_atoms_per_frame
        print(f"Detected non-standard multi-frame format: {num_frames} frames")

        atom_type_map = {}
        next_type_id = 0

        for frame_idx in range(num_frames):
            positions = []
            types = []

            for j in range(num_atoms_per_frame):
                line_idx = 2 + frame_idx * num_atoms_per_frame + j
                parts = lines[line_idx].split()

                if len(parts) < 4:
                    continue

                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                # Map element to type (consistent across all frames)
                if element not in atom_type_map:
                    atom_type_map[element] = next_type_id
                    next_type_id += 1

                positions.append([x, y, z])
                types.append(atom_type_map[element])

            if len(positions) == num_atoms_per_frame:
                frames.append({
                    'positions': np.array(positions),
                    'types': np.array(types, dtype=int),
                    'box_length': box_length,
                    'num_atoms': num_atoms_per_frame
                })

    else:
        # Standard format: each frame has its own header
        i = 0
        while i < len(lines):
            # Read number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue

            # Read comment line (contains box info)
            if i + 1 >= len(lines):
                break

            comment = lines[i + 1].strip()
            box_length = parse_box_length(comment)

            # Read atom data
            positions = []
            types = []
            atom_type_map = {}  # Map element symbols to numeric types
            next_type_id = 0

            for j in range(num_atoms):
                line_idx = i + 2 + j
                if line_idx >= len(lines):
                    break

                parts = lines[line_idx].split()
                if len(parts) < 4:
                    continue

                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                # Map element to type (first element → 0, second element → 1)
                if element not in atom_type_map:
                    atom_type_map[element] = next_type_id
                    next_type_id += 1

                positions.append([x, y, z])
                types.append(atom_type_map[element])

            if len(positions) == num_atoms:
                frames.append({
                    'positions': np.array(positions),
                    'types': np.array(types, dtype=int),
                    'box_length': box_length,
                    'num_atoms': num_atoms
                })

            # Move to next frame
            i += num_atoms + 2

    return frames


def parse_box_length(comment_line):
    """
    Extract box length from XYZ comment line.

    Supports formats like:
        Lattice="99.0 0.0 0.0 0.0 99.0 0.0 0.0 0.0 99.0"

    Returns cubic box length (assumes cubic box).
    """
    if 'Lattice=' in comment_line:
        try:
            # Extract the lattice string
            start = comment_line.find('Lattice="') + 9
            end = comment_line.find('"', start)
            lattice_str = comment_line[start:end]

            # Parse lattice values (9 values for 3x3 matrix)
            values = [float(x) for x in lattice_str.split()]

            if len(values) >= 9:
                # For cubic box, use the diagonal elements
                box_x = values[0]
                box_y = values[4]
                box_z = values[8]

                # Check if cubic (all diagonal elements equal)
                if abs(box_x - box_y) < 1e-6 and abs(box_y - box_z) < 1e-6:
                    return box_x
                else:
                    print(f"Warning: Non-cubic box detected ({box_x}, {box_y}, {box_z}). Using x-dimension.")
                    return box_x
        except Exception as e:
            print(f"Warning: Could not parse Lattice from comment line: {e}")

    # Default box length if parsing fails
    print("Warning: Using default box length 99.0 Å")
    return 99.0


def create_mock_config(box_length, num_particles, ff_path):
    # Create empty Config object (bypass __init__)
    config = object.__new__(Config)

    # Set all required attributes (hardcoded for testing)
    config.box_length = box_length
    config.num_particles = num_particles
    config.ff_path = ff_path
    config.kT = 0.592
    config.ratio_type1 = 1
    config.ratio_type2 = 1
    config.total_ratio = 2
    config.lower_energy_cutoff = 1.5
    config.energy_cutoff = 15.0
    config.clust_cutoff = 3.5 # shouldnt matter
    config.seed = 0
    config.bias_type = None
    config.bias = None
    config.active_moves = []
    config.move_probabilities = []

    return config


def main():
    if len(sys.argv) != 4:
        print("Usage: python test_energy.py <xyz_file> <ff_path> <output_file>")
        print("\nExample:")
        print("  python test_energy.py trajectory.xyz potentials/PR_NaCl.txt energies.dat")
        sys.exit(1)

    xyz_file = sys.argv[1]
    ff_path = sys.argv[2]
    output_file = sys.argv[3]

    print(f"Reading XYZ file: {xyz_file}")
    frames = parse_xyz_file(xyz_file)
    print(f"Found {len(frames)} frame(s)")

    if not frames:
        print("Error: No valid frames found in XYZ file")
        sys.exit(1)

    # Extract basename from ff_path for config (e.g., "potentials/PR_NaCl.txt" -> "PR_NaCl.txt")
    import os
    ff_basename = os.path.basename(ff_path)

    # Create mock config and system using actual Config and System classes
    print(f"Loading force field: {ff_path}")
    config = create_mock_config(
        frames[0]['box_length'],
        frames[0]['num_atoms'],
        ff_basename
    )
    system = System(config, id=0)

    print(f"Calculating energies using System.calc_full_energy()...")

    with open(output_file, 'w') as f:
        f.write("# Frame  Potential\n")

        for idx, frame in enumerate(frames):
            # Set positions and types directly on system
            system.positions = frame['positions']
            system.types = frame['types']
            system.num_particles = frame['num_atoms']
            system.box_length = frame['box_length']

            # Use actual System.calc_full_energy() method (sets system.energy)
            system.calc_full_energy()
            energy = system.energy

            f.write(f"{idx}  {energy:.6f}\n")

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  Frame {idx}: E = {energy:.6f} kT")

    print(f"\nDone! Energies written to {output_file}")
    print(f"Total frames processed: {len(frames)}")


if __name__ == "__main__":
    main()
