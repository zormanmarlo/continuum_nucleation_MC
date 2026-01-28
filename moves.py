import numpy as np

class Move:
    def __init__(self, system):
        '''Initialize base move class with system reference and statistics tracking'''
        self.system = system
        self.attempts = 0
        self.rejections = 0
    
    def get_acceptance_rate(self):
        '''Calculate and return current acceptance rate for this move type'''
        if self.attempts == 0:
            return 0.0
        return 1.0 - (self.rejections / self.attempts)
    
    def reset_stats(self):
        '''Reset move attempt and rejection counters to zero'''
        self.attempts = 0
        self.rejections = 0
    
    def attempt_move(self, particle_idx):
        '''Abstract method for attempting a Monte Carlo move - must be implemented by subclasses'''
        raise NotImplementedError("Subclasses must implement attempt_move")


class TranslationMove(Move):
    def __init__(self, system):
        '''Initialize translation move with maximum displacement parameter'''
        super().__init__(system)
        self.max_displacement = system.config.max_displacement
    
    def attempt_move(self, particle_idx):
        '''Attempt random translation move within spherical displacement constraint'''
        self.attempts += 1

        displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)
        while (np.sum(displacement**2) > self.max_displacement**2):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)

        old_pos = self.system.positions[particle_idx].copy()
        new_pos = (old_pos + displacement) % (self.system.box_length)

        delta_energy, bias_energy, new_energy, old_energy = self.system.calc_energy_delta(particle_idx, new_pos, old_pos)
        acc_prob = min(1, np.exp(np.clip((-(delta_energy+bias_energy)/self.system.kT), -500, 500)))
        move_rand = np.random.rand()

        if self.system.config.parameters['output_detailed_balance']:
            prob_new = np.exp(-new_energy / self.system.kT)
            prob_old = np.exp(-old_energy / self.system.kT)
            acc_new = min(1, np.exp(np.clip((-(delta_energy)/self.system.kT), -500, 500)))
            acc_old = min(1, np.exp(np.clip(((delta_energy)/self.system.kT), -500, 500)))
            acc_new = 1 if acc_new >= move_rand else 0
            acc_old = 1 if acc_old >= np.random.rand() else 0
            self.system.detailed_balance_data['translation']['fwd'] += (prob_old * acc_new)
            self.system.detailed_balance_data['translation']['rvr'] += (prob_new * acc_old) 

        if np.random.rand() >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[particle_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class SwapMove(Move):
    def __init__(self, system):
        '''Initialize swap move for random particle repositioning'''
        super().__init__(system)
    
    def attempt_move(self, particle_idx):
        '''Attempt to swap particle to random position in simulation box'''
        self.attempts += 1

        old_pos = self.system.positions[particle_idx].copy()
        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        delta_energy, bias_energy, _, _ = self.system.calc_energy_delta(particle_idx, new_pos, old_pos)
        
        acc_prob = min(1, np.exp(np.clip((-(delta_energy+bias_energy)/self.system.kT), -500, 500)))

        ## TODO: Detailed balance for swap move (we generally dont use this right now)
        if np.random.rand() >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[particle_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class InOutAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC in-out move with volume calculations for bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3 # subtract Vin for accuracy? seems neglible for large boxes

    def attempt_move(self, anchor_idx):
        '''Attempt AVBMC move to remove particle from cluster to bulk solution'''
        self.attempts += 1

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)
        old_pos = self.system.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
        while self.system.calc_dist(old_pos, new_pos) <= self.system.config.upper_cutoff:
            new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        delta_energy, bias_energy, new_energy, old_energy = self.system.calc_energy_delta(target_idx, new_pos, old_pos)
        # avbmc_energy = np.exp(np.clip((-(delta_energy+bias_energy)/self.system.kT)*self.Vout/self.Vin*(Nin)/(self.system.num_particles-Nin+1), -500, 500))
        avbmc_energy = np.exp(np.clip(-(delta_energy+bias_energy)/self.system.kT,
                                -500, 500)) * self.Vout/self.Vin * (Nin)/(self.system.num_particles-Nin+1)
        acc_prob = min(1, avbmc_energy)
        move_rand = np.random.rand()

        if self.system.config.parameters['output_detailed_balance']:
            prob_old = np.exp(-old_energy / self.system.kT)
            prob_new = np.exp(-new_energy / self.system.kT)
            acc_inout = min(1, np.exp(np.clip(-(delta_energy)/self.system.kT, -500, 500)) * self.Vout/self.Vin * Nin/(self.system.num_particles-Nin+1))
            acc_outin =  min(1, np.exp(np.clip((delta_energy)/self.system.kT, -500, 500)) * self.Vin/self.Vout * (self.system.num_particles-(Nin-1))/(Nin-1+1))
            acc_inout = 1 if acc_inout >= move_rand else 0
            # acc_outin = 1 if acc_outin < move_rand else 0
            acc_outin = 1 if acc_outin >= np.random.rand() else 0

            alpha_inout = 1 / (self.Vout * Nin)
            alpha_outin = 1 / (self.Vin * (self.system.num_particles-(Nin-1)))
            self.system.detailed_balance_data['inout_avbmc']['fwd'] += (prob_old * alpha_inout * acc_inout)
            self.system.detailed_balance_data['inout_avbmc']['rvr'] += (prob_new * alpha_outin * acc_outin)

        if move_rand >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[target_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class OutInAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC out-in move with volume calculations and Rosenbluth sampling'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3
    
    def attempt_move(self, anchor_idx):
        '''Attempt AVBMC move to insert bulk particle into cluster using Rosenbluth weighting'''
        self.attempts += 1

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        target_idx = np.random.randint(self.system.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0):
            target_idx = np.random.randint(self.system.num_particles)

        old_energy = self.system.calc_energy(target_idx)
        old_pos = self.system.positions[target_idx].copy()
        self.system.target_clust_idx = self.system.find_target_cluster()

        # Calculate wnew for the new configuration
        nrb = self.system.config.parameters['rosenbluth_trials']
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):            
            # Uniform sampling on the sphere for direction
            r = np.cbrt(np.random.rand() * (self.system.clust_cutoff**3 - self.system.config.lower_cutoff**3) + self.system.config.lower_cutoff**3)
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian coordinates
            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_pos = (self.system.positions[anchor_idx] + np.array([x, y, z])) % (self.system.box_length)
            
            self.system.positions[target_idx] = new_pos
            new_energy = self.system.calc_energy(target_idx)
            w = np.exp(-new_energy / self.system.kT)
            if np.isnan(w) or np.isinf(w):
                w = 0
                wnew += 0
            else:
                wnew += w

            rosenbluth_weights.append((w, new_energy, new_pos))
            self.system.positions[target_idx] = old_pos

        if wnew == 0:
            self.rejections += 1
            return

        # Select one configuration based on Rosenbluth weights
        valid_rosenbluth_weights = [(weight, d, pos) for weight, d, pos in rosenbluth_weights if np.isfinite(weight) and weight > 0]
        wnew_valid = sum(weight for weight, _, _ in valid_rosenbluth_weights)

        rosenbluth_weights_norm = [(weight / wnew_valid, d, pos) for weight, d, pos in valid_rosenbluth_weights]
        _, new_energy, selected_pos = valid_rosenbluth_weights[np.random.choice(range(len(valid_rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.system.positions[target_idx] = selected_pos

        wold = np.exp(-(old_energy) / self.system.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            old_energy_out = new_energy
            
            old_pos_out = self.system.positions[target_idx_out].copy()
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_pos_out, new_pos_out) <= self.system.config.upper_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            
            self.system.positions[target_idx_out] = new_pos_out.copy()
            new_energy_out = self.system.calc_energy(target_idx_out)
            w = np.exp(-new_energy_out / self.system.kT)

            wold += w
            self.system.positions[target_idx_out] = old_pos_out
            
        if self.system.bias is not None:
            # self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
            self.system.tmp_target_clust_idx = self.system.find_target_cluster(target_idx=target_idx)
            self.system.target_clust_idx = self.system.find_target_cluster(target_idx=target_idx)
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy
        self.system.energy += delta_energy
        self.system.bias_energy += bias_energy

        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_particles - Nin) / (Nin + 1))
        acc_prob = min(1, avbmc_energy)
        move_rand = np.random.rand()

        if self.system.config.parameters['output_detailed_balance']:
            prob_old = np.exp(-old_energy / self.system.kT)
            prob_new = np.exp(-new_energy / self.system.kT)
            acc_inout = (wold/wnew) * (self.Vout / self.Vin) * ((Nin+1) / (self.system.num_particles - Nin))
            acc_inout = min(1, acc_inout)
            acc_outin = (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_particles - Nin) / (Nin + 1))
            acc_outin = min(1, acc_outin)
            acc_outin = 1 if acc_outin >= move_rand else 0
            acc_inout = 1 if acc_inout >= np.random.rand() else 0

            alpha_inout = 1 / (self.Vout * (Nin+1))
            alpha_outin = 1 / (self.Vin * (self.system.num_particles-(Nin)))
            self.system.detailed_balance_data['outin_avbmc']['fwd'] += (prob_old * alpha_outin * acc_outin)
            self.system.detailed_balance_data['outin_avbmc']['rvr'] += (prob_new * alpha_inout * acc_inout)

        if move_rand >= acc_prob:
            self.system.positions[target_idx] = old_pos
            self.system.energy -= delta_energy
            self.system.bias_energy -= bias_energy
            self.rejections += 1

class NVTInOutMove(Move):
    def __init__(self, system):
        '''Initialize NVT in-out move for cluster nucleation with AVBMC bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3
    
    def attempt_move(self, anchor_idx, Nin_idx):
        '''Attempt NVT move to remove particle from target cluster to bulk with nucleation bias'''
        self.attempts += 1
        Nin = len(Nin_idx)

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)
        old_pos = self.system.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
        regen = True
        # Ensure target cluster is current
        self.system.target_clust_idx = self.system.find_target_cluster()
        clust_pos = np.asarray([self.system.positions[i] for i in self.system.target_clust_idx])
        while regen:
            distances = np.linalg.norm(clust_pos - new_pos, axis=1)
            if np.all(distances > self.system.config.upper_cutoff):
                regen = False
            else:
                new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        # Store old cluster size for AVBMC calculation
        old_cluster_size = len(self.system.target_clust_idx)
        delta_energy, bias_energy, new_energy, old_energy = self.system.calc_energy_delta(target_idx, new_pos, old_pos)

        try:
            avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.system.kT)*self.Vout/self.Vin*(Nin)/(self.system.num_particles-old_cluster_size+1)*(old_cluster_size/(old_cluster_size-1))
        except:
            # print("ZeroDivisionError")
            # print(self.Vin, self.Vout, Nin, self.system.num_particles - old_cluster_size + 1, self.system.kT, old_cluster_size)
            avbmc_energy = 0
        acc_prob = min(1, avbmc_energy)
        move_rand = np.random.rand()

        if self.system.config.parameters['output_detailed_balance']:
            prob_old = np.exp(-old_energy / self.system.kT)
            prob_new = np.exp(-new_energy / self.system.kT)
            # Forward: deletion (in→out) from cluster of size n
            n = old_cluster_size
            if n > 1:
                acc_deletion = min(1, np.exp(np.clip(-(delta_energy)/self.system.kT, -500, 500))
                                 * (n/(n-1)) * (Nin/(self.system.num_particles-n+1)) * (self.Vout/self.Vin))
                acc_insertion = min(1, np.exp(np.clip((delta_energy)/self.system.kT, -500, 500))
                                  * ((n-1)/n) * ((self.system.num_particles-(n-1))/Nin) * (self.Vin/self.Vout))
                acc_deletion = 1 if acc_deletion >= move_rand else 0
                acc_insertion = 1 if acc_insertion >= np.random.rand() else 0
                alpha_deletion = 1 / (n * Nin * self.Vout)
                alpha_insertion = 1 / ((n-1) * (self.system.num_particles-(n-1)) * self.Vin)
                self.system.detailed_balance_data['nvt_inout']['fwd'] += (prob_old * alpha_deletion * acc_deletion)
                self.system.detailed_balance_data['nvt_inout']['rvr'] += (prob_new * alpha_insertion * acc_insertion)

        if move_rand >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[target_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class NVTOutInMove(Move):
    def __init__(self, system):
        '''Initialize NVT out-in move for cluster growth with Rosenbluth sampling and nucleation bias'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3
    
    def attempt_move(self, anchor_idx, Nin_idx):
        '''Attempt NVT move to insert bulk particle into target cluster using Rosenbluth weighting'''
        self.attempts += 1
        Nin = len(Nin_idx)

        target_idx = np.random.randint(self.system.num_particles)
        # Ensure target cluster is current
        self.system.target_clust_idx = self.system.find_target_cluster()
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0) or (target_idx in self.system.target_clust_idx):
            target_idx = np.random.randint(self.system.num_particles)

        old_energy = self.system.calc_energy(target_idx)
        old_pos = self.system.positions[target_idx].copy()
        self.system.target_clust_idx = self.system.find_target_cluster()

        # Calculate wnew for the new configuration
        nrb = self.system.config.parameters['rosenbluth_trials']  # Number of Rosenbluth trials
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):
            r = np.cbrt(np.random.rand() * (self.system.clust_cutoff**3 - self.system.config.lower_cutoff**3) + self.system.config.lower_cutoff**3)
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian coordinates
            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_pos = (self.system.positions[anchor_idx] + np.array([x, y, z])) % (self.system.box_length)
            
            self.system.positions[target_idx] = new_pos
            new_energy = self.system.calc_energy(target_idx)
            w = np.exp(-new_energy / self.system.kT)
            if np.isnan(w):
                wnew += 0
            else:
                wnew += w
            rosenbluth_weights.append((w, new_energy, new_pos))
            self.system.positions[target_idx] = old_pos

        if wnew == 0:
            self.rejections += 1
            return

        # Select one configuration based on Rosenbluth weights
        rosenbluth_weights_norm = [(weight / wnew, d, pos) for weight, d, pos in rosenbluth_weights]
        _, new_energy, selected_pos = rosenbluth_weights[np.random.choice(range(len(rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.system.positions[target_idx] = selected_pos

        # Calculate wold for the original configuration
        wold = np.exp(-(old_energy) / self.system.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            
            old_pos_out = self.system.positions[target_idx_out].copy()
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_pos_out, new_pos_out) <= self.system.config.upper_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            
            self.system.positions[target_idx_out] = new_pos_out
            new_energy_out = self.system.calc_energy(target_idx_out)
            w = np.exp(-new_energy_out / self.system.kT)

            wold += w
            self.system.positions[target_idx_out] = old_pos_out


        self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
        if self.system.bias is not None:
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy
        self.system.energy += delta_energy
        self.system.bias_energy += bias_energy

        # Use old cluster size (before insertion) for acceptance calculation
        n = len(self.system.tmp_target_clust_idx)
        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_particles - n) / (Nin + 1)) * (n / (n+1))

        acc_prob = min(1, avbmc_energy)
        move_rand = np.random.rand() # want to use the same random number for detailed balance fwd move

        if self.system.config.parameters['output_detailed_balance']:
            prob_old = np.exp(-old_energy / self.system.kT)
            prob_new = np.exp(-new_energy / self.system.kT)
            acc_insertion = min(1, (wnew/wold) * (self.Vin/self.Vout)
                              * ((self.system.num_particles-n)/(Nin+1)) * (n/(n+1)))
            acc_deletion = min(1, (wold/wnew) * (self.Vout/self.Vin)
                             * ((n+1)/n) * ((Nin+1)/(self.system.num_particles-(n+1)+1)))
            acc_insertion = 1 if acc_insertion >= move_rand else 0
            acc_deletion = 1 if acc_deletion >= np.random.rand() else 0
            alpha_insertion = 1 / (n * (self.system.num_particles-n) * self.Vin)
            alpha_deletion = 1 / ((n+1) * (Nin+1) * self.Vout)
            self.system.detailed_balance_data['nvt_outin']['fwd'] += (prob_old * alpha_insertion * acc_insertion)
            self.system.detailed_balance_data['nvt_outin']['rvr'] += (prob_new * alpha_deletion * acc_deletion)

        if move_rand >= acc_prob:
            self.system.positions[target_idx] = old_pos
            self.system.energy -= delta_energy
            self.system.bias_energy -= bias_energy
            self.rejections += 1
