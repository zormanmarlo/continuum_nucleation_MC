from scipy.interpolate import interp1d
import numpy as np
import random

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def generate_energy_table(path, num_bins=1000, cutoff=20.0):
    data = np.loadtxt(path)
    distances = data[:, 0]
    precomputed_distances = np.linspace(0, cutoff, num_bins)
    precomputed_energies = np.zeros((5, num_bins))

    # Interpolate energies for each interaction type
    for col in range(1, 4):
        interp_func = interp1d(distances, data[:, col], kind='linear', fill_value="extrapolate")
        precomputed_energies[col - 1] = interp_func(precomputed_distances)

    combined = np.vstack((precomputed_distances, precomputed_energies)).T
    return combined

class Particle:
    def __init__(self, index, type, xyz):
        self.idx = index
        self.type = type
        self.position = np.array(xyz)
        self.cluster = None

class PMF:
    def __init__(self, path):
        self.pmf_function = np.loadtxt(path)
        distances = self.pmf_function[:,0]
        # X = 0.5
        # N = int(X / (distances[1] - distances[0]))
        sigma = 3
        self.pmf_function[:,4] = (sigma/self.pmf_function[:,0])**12
        self.pmf_function[:,5] = (sigma/self.pmf_function[:,0])**12
        
    
        self.cut_off = 20.0

    def energies(self, type, distances):
        distances = np.asarray(distances)
        sorted_distances = self.pmf_function[:, 0]  # First column contains sorted distances

        # Binary search to find indices of the upper bound
        indices = np.searchsorted(sorted_distances, distances)

        # Clip indices to valid bounds to avoid overflow
        indices = np.clip(indices, 1, len(sorted_distances) - 1)

        # Get the left and right indices
        left_indices = indices - 1
        right_indices = indices

        # Extract distances and energies for interpolation
        left_distances = sorted_distances[left_indices]
        right_distances = sorted_distances[right_indices]

        left_energies = self.pmf_function[left_indices, type + 1]
        right_energies = self.pmf_function[right_indices, type + 1]

        # Compute weights for linear interpolation
        weights = (distances - left_distances) / (right_distances - left_distances)

        # Perform interpolation
        interpolated_energies = left_energies + weights * (right_energies - left_energies)

        return interpolated_energies

    def energy(self, type, distance):
        index = np.argmin(np.abs(self.pmf_function[:, 0] - distance))
        return self.pmf_function[index, type+1]
    
class Bias:
    def __init__(self, target=100, path=None):
        if path is None:
            self.bias = np.zeros(target)
        else:
            self.bias = np.loadtxt(path)

    def denergy(self, new, old):
        try: 
            old_bias = self.bias[old-1]
            new_bias = self.bias[new-1]
            dE_bias = new_bias - old_bias
        except:
            dE_bias = 0 
        return dE_bias
    
    def energy(self, size):
        try:
            return self.bias[size-1]
        except:
            return 0
        
    def update(self, distribution):
        new_potential = np.zeros_like(self.bias) # will likely need to change this to account for new size
        pivot_bin = np.argmax(distribution)
        n_star = distribution[pivot_bin]
        n_star_m = 1 / n_star
        
        for i in range(len(distribution)):
            if distribution[i] > 0:
                new_potential[i] = self.bias[i]  + 0.6*np.log(distribution[i] / n_star)
            else:
                new_potential[i] = self.bias[pivot_bin] + 0.6*np.log(n_star_m)
        
        # Re-shift potentials to ensure the reference state is 0kBT.
        new_potential -= new_potential[1]
        self.bias = new_potential
        return self.bias
