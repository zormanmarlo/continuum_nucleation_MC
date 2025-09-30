import numpy as np

## --------------------------------------------------------------- ## 
## Script to generate a spherical H FCC crystal with bath atoms   ##
## --------------------------------------------------------------- ## 

def generate_sphere(N, X, N_extra, boxsize, output_file="/Users/mdog/Desktop/50mer.xyz", seed=0):
    np.random.seed(seed)
    atoms = []
    
    cube_size = int(np.ceil((N / 0.5) ** (1/3)))
    center = boxsize / 2
    radius = (N / (4/3 * np.pi)) ** (1/3) * X
    
    for i in range(cube_size):
        for j in range(cube_size):
            for k in range(cube_size):
                x, y, z = i * X + center - (cube_size * X / 2), j * X + center - (cube_size * X / 2), k * X + center - (cube_size * X / 2)
                distance = np.linalg.norm([x - center, y - center, z - center])
                if distance <= radius:
                    atom_type = "H"
                    atoms.append((atom_type, x, y, z, distance))
    
    atoms = sorted(atoms, key=lambda atom: atom[4])
    if len(atoms) > N:
        atoms = atoms[:N]
    elif len(atoms) < N:
        N_extra += (N - len(atoms))
    
    max_sphere_radius = max(atom[4] for atom in atoms)
    
    extra_atoms = []
    max_attempts = 10000
    attempts = 0
    
    while len(extra_atoms) < N_extra and attempts < max_attempts:
        x, y, z = np.random.uniform(0, boxsize, 3)
        pos = np.array([x, y, z])
        distance = np.linalg.norm([x - center, y - center, z - center])
        if distance <= (max_sphere_radius + 2):
            attempts += 1
            continue
            
        too_close = False
        for atom in atoms:
            if np.linalg.norm([x - atom[1], y - atom[2], z - atom[3]]) < 3:
                too_close = True
                break
        if not too_close:
            for atom in extra_atoms:
                if np.linalg.norm([x - atom[1], y - atom[2], z - atom[3]]) < 3:
                    too_close = True
                    break
        if not too_close:
            atom_type = "H"
            extra_atoms.append((atom_type, x, y, z))
        attempts += 1
    
    all_atoms = atoms + extra_atoms
    
    with open(output_file, "w") as f:
        f.write(f"{len(all_atoms)}\n")
        f.write("H Spherical Lattice with Extra Atoms\n")
        for atom in all_atoms:
            f.write(f"{atom[0]} {atom[1]:.3f} {atom[2]:.3f} {atom[3]:.3f}\n")

X = 3.8
boxsize = 130
n_total = 512

for N in range(2,100,2):
    print(N)
    N_extra = n_total - N
    for i in range(0,5):
        i_str = str(i).zfill(2)
        generate_sphere(N, X, N_extra, boxsize, output_file=f"./S6_512mono_sphere/{N}mer_{i_str}.xyz", seed=i)