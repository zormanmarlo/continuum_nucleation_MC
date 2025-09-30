import numpy as np
import random

def generate_amorphous_cluster(N, X, N_extra, boxsize,
                                     output_file="./amorphous_LJ.xyz", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
    lattice = {}
    max_cells = int(boxsize // X)

    # Build FCC lattice with single atom type H
    for i in range(max_cells):
        for j in range(max_cells):
            for k in range(max_cells):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x, y, z = (i + dx) * X, (j + dy) * X, (k + dz) * X
                    if x < boxsize and y < boxsize and z < boxsize:
                        lattice[(i + dx, j + dy, k + dz)] = ("H", x, y, z)

    # Central lattice site as seed
    center_idx = tuple(int(round((boxsize / 2) / X)) for _ in range(3))
    if center_idx not in lattice:
        for dx, dy, dz in [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]:
            neighbor = (center_idx[0]+dx, center_idx[1]+dy, center_idx[2]+dz)
            if neighbor in lattice:
                center_idx = neighbor
                break

    # Amorphous cluster growth (biased random walk with sparse connectivity)
    cluster = set()
    visited = set()
    frontier = [center_idx]
    max_neighbors_per_step = 2

    while len(cluster) < N and frontier:
        current = frontier.pop(random.randint(0, len(frontier) - 1))
        if current not in lattice or current in cluster:
            continue

        cluster.add(current)
        visited.add(current)

        neighbors = [(current[0]+dx, current[1]+dy, current[2]+dz)
                     for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]]
        random.shuffle(neighbors)

        # Filter neighbors: avoid dense areas
        filtered = []
        for neighbor in neighbors:
            if neighbor not in lattice or neighbor in visited:
                continue
            count = sum(1 for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
                        if (neighbor[0]+dx, neighbor[1]+dy, neighbor[2]+dz) in cluster)
            if count <= 1:  # avoid compact regions
                filtered.append(neighbor)

        for neighbor in filtered[:random.randint(1, max_neighbors_per_step)]:
            frontier.append(neighbor)

    atoms = [lattice[idx] for idx in cluster]

    # Random bath atoms (not on lattice), spacing constraints
    attempts, max_attempts = 0, 10000
    extra_atoms = []
    while len(extra_atoms) < N_extra and attempts < max_attempts:
        x, y, z = np.random.uniform(0, boxsize, 3)
        too_close = False
        for atom in atoms + extra_atoms:
            if np.linalg.norm([x - atom[1], y - atom[2], z - atom[3]]) < 3:
                too_close = True
                break
        if not too_close:
            extra_atoms.append(("H", x, y, z))
        attempts += 1

    if attempts >= max_attempts:
        print(f"Warning: Only placed {len(extra_atoms)} of {N_extra} extra atoms")

    all_atoms = atoms + extra_atoms

    with open(output_file, "w") as f:
        f.write(f"{len(all_atoms)}\n")
        f.write("Highly Amorphous H Cluster with Extra Atoms\n")
        for atom in all_atoms:
            f.write(f"{atom[0]} {atom[1]:.3f} {atom[2]:.3f} {atom[3]:.3f}\n")

    print(f"Amorphous cluster saved to {output_file}")

# Calculate box size from concentration
def calc_boxsize(conc_M, N_ions):
    """Calculate box size in Angstroms for given concentration and number of ion pairs"""
    # Convert M to molecules/Angstrom^3: M * 6.022e23 / 1e27 = M * 6.022e-4
    density_per_A3 = conc_M * 6.022e-4
    volume_per_molecule = 1.0 / density_per_A3
    total_volume = N_ions * volume_per_molecule
    return total_volume**(1/3)


# Example usage
X = 3.8
boxsize = 130
n_total = 512
for N in range(2,100,2):
    print(N)
    N_extra = n_total - N
    for i in range(0,5):
        i_str = str(i).zfill(2)
        generate_amorphous_cluster(N, X, N_extra, boxsize, output_file=f"./S6_512mono/{N}mer_{i_str}.xyz", seed=i)