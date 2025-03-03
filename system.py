import random
import numpy as np
from numba import njit, prange
from time import time

# from scipy.spatial.distance import pdist, squareform
# from scipy.sparse.csgraph import connected_components
# from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from utils import PMF, Particle, Bias, generate_energy_table

@njit
def find_neighbors_numba(positions, pos1, cutoff, box_length):
    cutoff_squared = cutoff**2
    neighbors = []
    for i in range(positions.shape[0]):  # number of particles
        dx = positions[i, 0] - pos1[0]
        dy = positions[i, 1] - pos1[1]
        dz = positions[i, 2] - pos1[2]

        # Apply periodic boundary conditions
        dx -= box_length * round(dx / box_length)
        dy -= box_length * round(dy / box_length)
        dz -= box_length * round(dz / box_length)

        dist_squared = dx**2 + dy**2 + dz**2
        if dist_squared < cutoff_squared:
            neighbors.append(i)
    return neighbors

def calc_energy(particle_idx, positions, types, box_length, cutoff, energy_table, clust_cutoff, counter_cutoff, concentration, pmf=None):
       concentration = 1
       # Using NUMBA provides a significant speed up for low concentration systems
       # However there is a bug, so for now we just default to using Scipy to calculate energy
       if concentration <= 0.5:
        # Extract particle1 information
            pos1 = positions[particle_idx]
            type1 = types[particle_idx]
            
            neighbors = find_neighbors_numba(positions, pos1, cutoff, box_length)
            clust_neighbors = find_neighbors_numba(positions, pos1, clust_cutoff, box_length)

            # Remove particle1 from the neighbor list
            neighbors = [idx for idx in neighbors if idx != particle_idx]
            precomputed_distances = energy_table[:, 0]
            precomputed_energies = energy_table[:, 1:]

            # Initialize energy
            energy = 0.0

            # Loop over neighbors to calculate energy contributions
            screened_candidates =  []
            for neighbor_idx in neighbors:
                pos2 = positions[neighbor_idx]
                type2 = types[neighbor_idx]

                # Calculate distance using minimum image convention
                delta = pos2 - pos1
                delta = delta - box_length * np.round(delta / box_length)
                distance = np.linalg.norm(delta)
                # Check if within cut off
                if distance <= 20 and distance >= 0.0:
                    # Add energy contribution based on particle types
                    if type1 == 0 and type2 == 0:
                        interaction_type = 0
                        if distance < counter_cutoff:
                            screened_candidates.append(neighbor_idx)
                    elif type1 == 1 and type2 == 1:
                        interaction_type = 1
                        if distance < counter_cutoff:
                            screened_candidates.append(neighbor_idx)
                    else:
                        interaction_type = 2
                    
                    bin_idx = np.searchsorted(precomputed_distances, distance) - 1
                    if bin_idx >= 0 and bin_idx < len(precomputed_distances) - 1:
                        # Linear interpolation
                        dist1 = precomputed_distances[bin_idx]
                        dist2 = precomputed_distances[bin_idx + 1]
                        energy1 = precomputed_energies[bin_idx, interaction_type]
                        energy2 = precomputed_energies[bin_idx + 1, interaction_type]
                        weight = (distance - dist1) / (dist2 - dist1)
                        energy += energy1 * (1 - weight) + energy2 * weight
                
            # Check candidates for screening
            for screened_idx in screened_candidates:
                screened_neighbors = find_neighbors_numba(positions, positions[screened_idx], clust_cutoff, box_length)
                if any(neighbor in clust_neighbors for neighbor in screened_neighbors):
                    pos2 = positions[screened_idx]
                    type2 = types[screened_idx]
                    
                    delta = pos2 - pos1
                    delta = delta - box_length * np.round(delta / box_length)
                    distance = np.linalg.norm(delta)

                    if type1 == 0 and type2 == 0:
                        interaction_type = 3
                        sub_interaction_type = 0
                    elif type1 == 1 and type2 == 1:
                        interaction_type = 4
                        sub_interaction_type = 1
                    else:
                        raise ValueError("Screened interaction must be like-like")

                    bin_idx = np.searchsorted(precomputed_distances, distance) - 1
                    if bin_idx >= 0 and bin_idx < len(precomputed_distances) - 1:
                        # Linear interpolation
                        dist1 = precomputed_distances[bin_idx]
                        dist2 = precomputed_distances[bin_idx + 1]
                        energy1 = precomputed_energies[bin_idx, interaction_type]
                        energy2 = precomputed_energies[bin_idx + 1, interaction_type]
                        sub_energy1 = precomputed_energies[bin_idx, sub_interaction_type]
                        sub_energy2 = precomputed_energies[bin_idx + 1, sub_interaction_type]
                        weight = (distance - dist1) / (dist2 - dist1)
                        energy -= sub_energy1 * (1 - weight) + sub_energy2 * weight
                        energy += energy1 * (1 - weight) + energy2 * weight
            return energy
       
       else:
            particle1 = particle_idx
            pos1 = positions[particle1]
            type1 = types[particle1]

            # Calculate distance vector with periodic boundary conditions
            distances = positions - pos1
            distances = distances - box_length * np.round(distances / box_length)
            distances = np.linalg.norm(distances, axis=1)

            # Mask for particles within cut-off distance (excluding particle1 itself)
            within_cutoff = (distances < 20) & (distances > 0.0)

            # Apply the cutoff mask
            cutoff_distances = distances[within_cutoff]
            cutoff_positions = positions[within_cutoff]
            cutoff_types = types[within_cutoff]

            # Find particles to use screened potential for
            unlike_bonded_mask = (cutoff_distances < clust_cutoff) & (cutoff_types != type1)
            like_candidates_mask = (cutoff_distances < counter_cutoff) & (cutoff_types == type1)
            
            # For these candidates, check if they are bonded to an unlike particle
            bonded_to_unlike = np.any(
                np.linalg.norm(
                    cutoff_positions[like_candidates_mask, None, :] -
                    cutoff_positions[unlike_bonded_mask, :],
                    axis=2
                ) < clust_cutoff,
                axis=1
            )
            
            # Final mask for "like-bonded" particles
            like_bonded_mask = like_candidates_mask.copy()
            like_bonded_mask[like_candidates_mask] = bonded_to_unlike

            # Step 3: Override the original like-like interaction mask
            # Separate 0-0 and 1-1 masks for original interactions
            type_0_0_mask = (cutoff_types == 0) & (type1 == 0)
            type_1_1_mask = (cutoff_types == 1) & (type1 == 1)
            type_0_1_mask = (cutoff_types != type1)

            # Modified like-bonded masks for 0-0 and 1-1 interactions
            bonded_0_0_mask = (cutoff_types == 0) & (type1 == 0) & like_bonded_mask
            bonded_1_1_mask = (cutoff_types == 1) & (type1 == 1) & like_bonded_mask

            type_0_0_mask = type_0_0_mask & ~like_bonded_mask
            type_1_1_mask = type_1_1_mask & ~like_bonded_mask

            # Combined mask for type-specific interactions, now respecting 0-0, 1-1, and modified like-like interactions
            type_masks = [
                type_0_0_mask,  # Type 0-0 interactions
                type_1_1_mask,  # Type 1-1 interactions
                type_0_1_mask,  # Type 0-1 interactions
                bonded_0_0_mask,  # Modified "0-0" interactions
                bonded_1_1_mask   # Modified "1-1" interactions
            ]

            # Pre-compute energies for each type of interaction
            energy = sum(
                np.sum(pmf.energies(i, cutoff_distances[mask]))
                for i, mask in enumerate(type_masks)
                if np.any(mask)
            )
            return energy

class System:
    def __init__(self, l, n, s=11, pmf_path="gale.txt", bias_path=None, upper_cutoff=3.9, lower_cutoff=2.7, clust_cutoff=3.9, counter_cutoff=6.0, max_trans=4.2, target_max=100):
        self.box_length = l
        self.num_particles = n
        self.positions = []
        self.types = []

        self.seed = s
        np.random.seed(self.seed)

        self.cut_off = 20.0
        self.pmf = PMF("potentials/"+pmf_path)
        self.energy_table = generate_energy_table("potentials/"+pmf_path)
        if bias_path is not None:
            self.bias = Bias(path="bias/"+bias_path, target=target_max)
        else:
            self.bias = Bias(target=target_max)

        self.target_clust_idx = []
        self.cluster_sizes = []
        self.tmp_target_clust_idx = self.target_clust_idx.copy()

        self.energy = 0.0
        self.bias_energy = 0.0
        self.rejections = [0,0,0,0,0]
        self.attempts = [0,0,0,0,0]

        self.kT = 0.6
        self.clust_cutoff = clust_cutoff
        self.high_cutoff = upper_cutoff
        self.low_cutoff = lower_cutoff
        self.counter_cutoff = counter_cutoff
        self.Vin = 4.0/3.0 * np.pi * (self.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.low_cutoff**3)
        self.Vout = self.box_length**3

        self.concentration = self.num_particles/2/self.Vout * 1.66054E3
        if self.concentration > 0.5:
            print(f"Concentration is greater than 0.5 M ({self.concentration:.3f} M), using scipy to calculate energy")
        else:
            print(f"Concentration is less than 0.5 M ({self.concentration:.3f} M), using numba to calculate energy")
        print("OVERRULLING -- DUE TO A BUG IN NUMBA-BASED ENERGY CALCULATION, SCIPY WILL BE USED FOR ALL SYSTEMS")

        self.max_displacement = max_trans

    def init(self, input_path=None):
        if input_path is not None:
            with open("inputs/"+input_path, 'r') as f:
                for i, line in enumerate(f.readlines()[2:]):
                    tmp = line.split()
                    x, y, z = float(tmp[1]), float(tmp[2]), float(tmp[3])
                    assign_value = lambda atom: 0 if atom == "H" else 1 if atom == "O" else None
                    part_type = assign_value(str(tmp[0]))
                    self.positions.append(np.array([x, y, z]))
                    self.types.append(part_type)
        else:
            n_species = int(self.num_particles/2)
            # populate first species
            for i in range(n_species):
                position = np.round(np.random.rand(3) * self.box_length, 3)
                self.positions.append(position)
                self.types.append(0)
            # populate second species
            for i in range(n_species):
                i += n_species
                position = np.round(np.random.rand(3) * self.box_length, 3)
                self.positions.append(position)
                self.types.append(1)

        self.positions = np.array(self.positions)
        self.types = np.array(self.types)
        self.energy = self.calc_full_energy()
    
    def find_clusters(self):
        # pos = np.array([p.position for p in self.particles])
        # tree = cKDTree(self.positions, boxsize=self.box_length+0.001)
        neighbors = find_neighbors_numba(self.positions, self.positions[0], self.clust_cutoff, self.box_length)
        visited = set()
        clusters = []
        for i in range(self.num_particles):
            if i not in visited:
                cluster = []
                queue = [i]
                while queue:
                    current = queue.pop(0)
                    if current not in visited:
                        visited.add(current)
                        cluster.append(current)
                        neighbors = find_neighbors_numba(self.positions, self.positions[current], self.clust_cutoff, self.box_length)
                        # neighbors = tree.query_ball_point(self.positions[current], self.clust_cutoff)
                        queue.extend([n for n in neighbors if n not in visited])
                clusters.append(cluster)
        
        self.cluster_sizes = [len(cluster) for cluster in clusters]
        self.target_clust_idx = clusters[0]
        
        return self.cluster_sizes, self.target_clust_idx
    
    def find_cluster_around_target(self, target_idx=0):        
        # Start with the target particle
        # tree = cKDTree(self.positions, boxsize=self.box_length+0.001)
        visited = set()
        queue = [target_idx]
        cluster = []
        
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                cluster.append(current)
                # neighbors = tree.query_ball_point(self.positions[current], self.clust_cutoff)
                # for neighbor in neighbors:
                    # if neighbor not in visited:
                        # queue.append(neighbor)
                neighbors = find_neighbors_numba(self.positions, self.positions[current], self.clust_cutoff, self.box_length)
                queue.extend([n for n in neighbors if n not in visited])
        
        # Cluster around target particle found
        self.target_clust_idx = cluster
        return self.target_clust_idx
        
    def check_in(self, particle_idx):
        for clust_idx in self.target_clust_idx:
            if clust_idx != particle_idx:
                distance = self.calc_dist(self.positions[particle_idx], self.positions[clust_idx])
                if distance < self.clust_cutoff:
                    return True
        return False
        
    def calc_in(self, particle_idx):
        particle_position = self.positions[particle_idx]
        
        distances = self.positions - particle_position
        distances = distances - self.box_length * np.round(distances / self.box_length)
        distances = np.linalg.norm(distances, axis=1)      
        close_indices = np.where((distances < self.clust_cutoff))[0]
        close_indices = [i for i in close_indices if i != particle_idx]

        Nin = len(close_indices)
        Nin_idx = close_indices
        
        return Nin, Nin_idx

    def calc_full_energy(self):
        self.energy = 0.0
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if i < j:
                    distance = self.calc_dist(self.positions[i], self.positions[j])
                    if distance < self.cut_off:
                        dist_idx = np.searchsorted(self.energy_table[:, 0], distance) - 1
                        if self.types[i] == 0 and self.types[j] == 0:
                            # self.energy += self.energy_table[dist_idx, 0]
                            self.energy += self.pmf.energy(0, distance)
                        if self.types[i] == 1 and self.types[j] == 1:
                            self.energy += self.pmf.energy(1, distance)
                            # self.energy += self.energy_table[dist_idx, 1]
                        if (self.types[i] == 0 and self.types[j] == 1) or (self.types[i] == 1 and self.types[j] == 0):
                            self.energy += self.pmf.energy(2, distance)
                            # self.energy += self.energy_table[dist_idx, 2]
                        # Specialized potential for like particles in the bonded region
                        if (self.types[i] == 0 and self.types[j] == 0) or (self.types[i] == 1 and self.types[j] == 1):
                            # dist = self.calc_dist(particle1.position, particle2.position)
                            if distance < self.counter_cutoff:
                                # Check for a common bonding particle of a different type
                                for bonding_particle in range(self.num_particles):
                                    if self.types[bonding_particle] != self.types[i]:
                                        dist1 = self.calc_dist(self.positions[i], self.positions[bonding_particle])
                                        dist2 = self.calc_dist(self.positions[i], self.positions[bonding_particle])

                                        # Both particles must be within the bonded region of the bonding particle
                                        if dist1 < self.clust_cutoff and dist2 < self.clust_cutoff:
                                            if self.types[i] == 0:
                                                self.energy -= self.pmf.energy(0, distance)
                                                self.energy += self.pmf.energy(3, distance)
                                            elif self.types[j] == 1:
                                                self.energy -= self.pmf.energy(1, distance)
                                                self.energy += self.pmf.energy(4, distance)
                                            break  # Exit the loop once a valid bonding particle is found
        return self.energy

    def calc_dist(self, pos1, pos2):
        dist_vec = np.abs(pos1 - pos2)
        dist_vec = dist_vec - self.box_length * np.round(dist_vec / self.box_length)
        return np.linalg.norm(dist_vec)
    
    def translation(self, particle_idx):
        self.attempts[0] += 1

        displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)
        while (np.sum(displacement**2) > self.max_displacement**2):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)

        old_pos = self.positions[particle_idx].copy()
        new_pos = (old_pos + displacement) % (self.box_length)

        # old_energy = calc_energy(particle_idx)
        old_energy = calc_energy(particle_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        self.positions[particle_idx] = new_pos

        self.tmp_target_clust_idx = self.target_clust_idx.copy()
        self.target_clust_idx = self.find_cluster_around_target()
        bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))    
        
        # new_energy = calc_energy(particle_idx)
        new_energy = calc_energy(particle_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)

        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy

        acc_prob = min(1, np.exp(-(delta_energy+bias_energy)/self.kT))
        if np.random.rand() >= acc_prob:
            self.positions[particle_idx] = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            self.target_clust_idx = self.tmp_target_clust_idx
            self.rejections[0] += 1

    # in -> out AVBMC move
    def inout_AVBMC(self, anchor_idx):
        self.attempts[1] += 1

        Nin, Nin_idx = self.calc_in(anchor_idx)
        if Nin == 0:
            self.rejections[1] += 1 
            return
        target_idx = np.random.choice(Nin_idx)

        old_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        old_pos = self.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
        while self.calc_dist(old_pos, new_pos) <= self.high_cutoff:
            new_pos = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length

        # self.particles[target_idx].position = new_pos
        self.positions[target_idx] = new_pos

        self.tmp_target_clust_idx = self.target_clust_idx.copy()
        self.target_clust_idx = self.find_cluster_around_target()
        bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))
            
        new_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy
        avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.kT)*self.Vout/self.Vin*(Nin)/(self.num_particles-Nin+1)

        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.positions[target_idx] = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            self.target_clust_idx = self.tmp_target_clust_idx
            self.rejections[1] += 1


    # out -> in AVBMC move
    def outin_AVBMC(self, anchor_idx):
        self.attempts[2] += 1

        Nin, Nin_idx = self.calc_in(anchor_idx)
        target_idx = np.random.randint(self.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0):
            target_idx = np.random.randint(self.num_particles)

        old_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        old_pos = self.positions[target_idx].copy()
        target_clust = self.find_cluster_around_target()

        # Calculate wnew for the new configuration
        nrb = 32  # Number of Rosenbluth trials
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):            
            r = np.cbrt(np.random.rand() * (self.clust_cutoff**3 - self.low_cutoff**3) + self.low_cutoff**3)
            # Uniform sampling on the sphere for direction
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian coordinates
            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_pos = (self.positions[anchor_idx] + np.array([x, y, z])) % (self.box_length)
            
            self.positions[target_idx] = new_pos
            new_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
            w = np.exp(-new_energy / self.kT)
            wnew += w

            rosenbluth_weights.append((w, new_energy, new_pos))
            self.positions[target_idx] = old_pos

        # Select one configuration based on Rosenbluth weights
        rosenbluth_weights_norm = [(weight / wnew, d, pos) for weight, d, pos in rosenbluth_weights]
        _, new_energy, selected_pos = rosenbluth_weights[np.random.choice(range(len(rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.positions[target_idx] = selected_pos

        # Calculate wold for the original configuration
        wold = np.exp(-(old_energy) / self.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            old_energy_out = new_energy
            
            old_pos_out = self.positions[target_idx_out].copy()
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
            while self.calc_dist(old_pos_out, new_pos_out) <= self.high_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
            
            self.positions[target_idx_out] = new_pos_out.copy()
            new_energy_out = calc_energy(target_idx_out, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
            w = np.exp(-new_energy_out / self.kT)

            wold += w
            self.positions[target_idx_out] = old_pos_out

        self.tmp_target_clust_idx = self.target_clust_idx.copy()
        self.target_clust_idx = self.find_cluster_around_target()
        bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))

        # print(new_energy, old_energy)
        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy

        avbmc_energy = np.exp(-bias_energy/self.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.num_particles - Nin) / (Nin + 1))
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.positions[target_idx] = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            self.target_clust_idx = self.tmp_target_clust_idx
            self.rejections[2] += 1
    
    # NVT Nucleation in -> out AVBMC move
    def nvt_inout_AVBMC(self, anchor_idx, Nin_idx):
        self.attempts[3] += 1
        Nin = len(Nin_idx)

        Nin, Nin_idx = self.calc_in(anchor_idx)
        if Nin == 0:
            self.rejections[1] += 1 
            return
        target_idx = np.random.choice(Nin_idx)

        # old_energy = calc_energy(target_idx)
        old_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        old_pos = self.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
        regen = True
        clust_pos = np.asarray([self.positions[i] for i in self.target_clust_idx])
        while regen:
            distances = np.linalg.norm(clust_pos - new_pos, axis=1)
            if np.all(distances > self.high_cutoff):
                regen = False
            else:
                new_pos = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length

        # self.particles[target_idx].position = new_pos
        self.positions[target_idx] = new_pos

        self.tmp_target_clust_idx = self.target_clust_idx.copy()
        self.target_clust_idx = self.find_cluster_around_target()
        bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))
            
        # new_energy = calc_energy(target_idx)
        new_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy

        try:
            avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.kT)*self.Vout/self.Vin*(Nin)/(self.num_particles-len(self.tmp_target_clust_idx)+1)*(len(self.tmp_target_clust_idx)/(len(self.tmp_target_clust_idx)-1))
        except:
            print("ZeroDivisionError")
            avbmc_energy = 0
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.positions[target_idx] = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            self.target_clust_idx = self.tmp_target_clust_idx
            self.rejections[3] += 1

    # NVT Nucleation out -> in AVBMC move
    def nvt_outin_AVBMC(self, anchor_idx, Nin_idx):
        self.attempts[4] += 1
        Nin = len(Nin_idx)

        # Nin, Nin_idx = self.calc_in(self.particles[anchor_idx])
        target_idx = np.random.randint(self.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0) or (target_idx in self.target_clust_idx):
            target_idx = np.random.randint(self.num_particles)

        old_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
        old_pos = self.positions[target_idx].copy()
        _ = self.find_cluster_around_target() 

        # Calculate wnew for the new configuration
        nrb = 32  # Number of Rosenbluth trials
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.high_cutoff * 2), 3)
            while ((sum(displacement**2) > self.high_cutoff**2) or (sum(displacement**2) < self.low_cutoff**2)):
                displacement = np.round(((np.random.rand(3) - 0.5) * self.high_cutoff * 2), 3)
            
            new_pos = (self.positions[anchor_idx] + displacement) % (self.box_length)
            self.positions[target_idx] = new_pos
            new_energy = calc_energy(target_idx, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
            w = np.exp(-new_energy / self.kT)

            wnew += w
            rosenbluth_weights.append((w, new_energy, new_pos))
            self.positions[target_idx] = old_pos

        # Select one configuration based on Rosenbluth weights
        rosenbluth_weights_norm = [(weight / wnew, d, pos) for weight, d, pos in rosenbluth_weights]
        _, new_energy, selected_pos = rosenbluth_weights[np.random.choice(range(len(rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.positions[target_idx] = selected_pos

        # Calculate wold for the original configuration
        wold = np.exp(-(old_energy) / self.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            old_energy_out = new_energy
            
            old_pos_out = self.positions[target_idx_out].copy()
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
            while self.calc_dist(old_pos_out, new_pos_out) <= self.high_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
            
            self.positions[target_idx_out] = new_pos_out
            new_energy_out = calc_energy(target_idx_out, self.positions, self.types, self.box_length, self.cut_off, self.energy_table, self.clust_cutoff, self.counter_cutoff, self.concentration, self.pmf)
            w = np.exp(-new_energy_out / self.kT)

            wold += w
            self.positions[target_idx_out] = old_pos_out

        self.tmp_target_clust_idx = self.target_clust_idx.copy()
        self.target_clust_idx = self.find_cluster_around_target()
        bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))

        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy

        avbmc_energy = np.exp(-bias_energy/self.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.num_particles - len(self.tmp_target_clust_idx)) / (Nin + 1)) * ((len(self.tmp_target_clust_idx)) / (len(self.tmp_target_clust_idx)+1))
        
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.positions[target_idx] = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            self.rejections[4] += 1
            self.target_clust_idx = self.tmp_target_clust_idx
