import numpy as np
import random

def generate_amorphous_nacl_cluster(N, X, N_extra, boxsize,
                                     output_file="/Users/mdog/Desktop/amorphous_cluster_nacl.xyz", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
    lattice = {}
    max_cells = int(boxsize // X)

    # Build NaCl FCC lattice
    for i in range(max_cells):
        for j in range(max_cells):
            for k in range(max_cells):
                atom_type = "Na" if (i + j + k) % 2 == 0 else "Cl"
                x, y, z = i * X, j * X, k * X
                lattice[(i, j, k)] = (atom_type, x, y, z)

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
            atom_type = "Na" if len(extra_atoms) % 2 == 0 else "Cl"
            extra_atoms.append((atom_type, x, y, z))
        attempts += 1

    if attempts >= max_attempts:
        print(f"Warning: Only placed {len(extra_atoms)} of {N_extra} extra atoms")

    all_atoms = atoms + extra_atoms
    na_atoms = [a for a in all_atoms if a[0] == "Na"]
    cl_atoms = [a for a in all_atoms if a[0] == "Cl"]
    sorted_atoms = na_atoms + cl_atoms

    with open(output_file, "w") as f:
        f.write(f"{len(sorted_atoms)}\n")
        f.write("Highly Amorphous NaCl Cluster with Extra Atoms\n")
        for atom in sorted_atoms:
            f.write(f"{atom[0]} {atom[1]:.3f} {atom[2]:.3f} {atom[3]:.3f}\n")

    print(f"Amorphous cluster saved to {output_file}")

# Example usage
X = 2.75
boxsize = 127 # 50 mM
for N in range(2,44,2):
    N_extra = 5000 - N
    for i in range(0,5):
        i_str = str(i).zfill(2)
        generate_amorphous_nacl_cluster(N, X, N_extra, boxsize, output_file=f"./inputs/nacl_us/2M_nacl_{N}mer_large_random_{i_str}.xyz", seed=i)


