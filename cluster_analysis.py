import MDAnalysis as mda
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree

## ------------------------------------------------------------ ##
## Script to generate average cluster population data from singular simulations ##
## ------------------------------------------------------------ ##

# Function to find box size based on concentration and number of ions
def calc_boxsize(conc_M, N_ions):
    density_per_A3 = conc_M * 6.022e-4
    volume_per_molecule = 1.0 / density_per_A3
    total_volume = N_ions * volume_per_molecule
    return total_volume**(1/3)

# Function to find clusters based on a distance cutoff
def find_clusters(positions, cutoff, box=40):
    if box is not None:
        tree = cKDTree(positions, boxsize=box+0.1)
    else:
        tree = cKDTree(positions)
    num_atoms = len(positions)
    visited = np.zeros(num_atoms, dtype=bool)
    clusters = []
    for i in range(num_atoms):
        if not visited[i]:
            cluster = set()
            stack = [i]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    cluster.add(current)
                    neighbors = tree.query_ball_point(positions[current], cutoff)
                    stack.extend([n for n in neighbors if not visited[n]])
            clusters.append(cluster)
    return clusters

# concentrations = ["10mM", "20mM", "50mM", "100mM", "2000mM"]
# concentrations = ["250mM", "500mM", "750mM", "1000mM", "1250mM", "1500mM", "1750mM", "2000mM", "2250mM", "2500mM", "2750mM", "3000mM", "3250mM", "3500mM", "3750mM", "4000mM", "4250mM", "4500mM", "4750mM", "5000mM"]
concentrations = ["4000mM"]
ff = "JC"

volumes = []
for conc in concentrations:
    N_ions = 1000/2
    # print(calc_boxsize(float(conc[:-2])/1000, N_ions))
    volumes.append(calc_boxsize(float(conc[:-2])/1000, N_ions)+0.5)


def calc_conc(n, l, spacing=1):
    return n/((l*spacing)**3)*(1/(1E-8))**3*1000/(6.022E23)

for i, conc in enumerate(concentrations):
    # Parameters
    traj_file = f"./{conc}_{ff}/traj-00.xyz"
    top_file = traj_file
    filename = f"./cluster_distribution_{conc}_{ff}_mc.txt"

    # Load the trajectory
    u = mda.Universe(top_file, traj_file)
    cutoff_distance = 3.5

    # Select Na and Cl ions
    ions = u.select_atoms("name H or name O")
    traj_skip = 10
    skip = 1

    # Analyze the trajectory
    cluster_counts = []
    times = []

    # Analyze the trajectory
    cluster_compositions = []

    # Create visualization trajectory file for H-H dimers
    viz_filename = f"/Users/mdog/Desktop/hh_dimers_{conc}_{ff}_visualization.xyz"

    for frame_idx, ts in enumerate(u.trajectory[traj_skip::skip]):
        positions = ions.positions
        clusters = find_clusters(positions, cutoff_distance, box=volumes[i])

        # Identify which particles are in H-H dimers (n20 clusters)
        hh_dimer_particles = set()
        for cluster in clusters:
            h_indices = [idx for idx in cluster if ions[idx].name == 'H']
            o_indices = [idx for idx in cluster if ions[idx].name == 'O']

            # Check if this is an H-H dimer (2 H atoms, 0 O atoms)
            if len(h_indices) == 2 and len(o_indices) == 0:
                hh_dimer_particles.update(h_indices)
                # Debug: print the positions of the H-H dimer particles
                if frame_idx == 0:  # Only print for first frame to avoid spam
                    pos1 = positions[h_indices[0]]
                    pos2 = positions[h_indices[1]]
                    dist = np.linalg.norm(pos1 - pos2)
                    print(f"H-H dimer found: indices {h_indices[0]}, {h_indices[1]}")
                    print(f"  Position 1: {pos1}")
                    print(f"  Position 2: {pos2}")
                    print(f"  Distance: {dist:.3f}")
                    print()

        # Write visualization frame
        with open(viz_filename, 'a' if frame_idx > 0 else 'w') as viz_file:
            viz_file.write(f"{len(ions)}\n")
            viz_file.write(f"Frame {frame_idx}: H-H dimers as Na, everything else as H\n")

            for idx, ion in enumerate(ions):
                x, y, z = positions[idx]
                # Mark H-H dimer particles as Na, everything else as H
                atom_type = 'Na' if idx in hh_dimer_particles else 'H'
                viz_file.write(f"{atom_type} {x:>8.3f} {y:>8.3f} {z:>8.3f}\n")

        # For each cluster, count the number of H and O atoms
        composition_counts = Counter()
        for cluster in clusters:
            h_count = sum(1 for idx in cluster if ions[idx].name == 'H')
            o_count = sum(1 for idx in cluster if ions[idx].name == 'O')
            composition_counts[(h_count, o_count)] += 1

        cluster_compositions.append(composition_counts)

    # Aggregate results over all frames
    total_counts = Counter()
    for comp_count in cluster_compositions:
        total_counts.update(comp_count)

    # Average over frames
    num_frames = len(cluster_compositions)
    avg_counts = {key: val / num_frames for key, val in total_counts.items()}

    # Save to file in the format: nXY = Z
    with open(filename, "w") as f:
        for (h, o), count in sorted(avg_counts.items()):
            f.write(f"n{h}{o} = {calc_conc(count, volumes[i])}\n")

    print(f"Visualization trajectory saved to: {viz_filename}")
    print(f"H-H dimers are shown as Na atoms, everything else as H atoms")
