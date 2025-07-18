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
        self.cut_off = 20.0
        # clean up the bulk potentials
        for i in range(4, self.pmf_function.shape[1]):
            col = self.pmf_function[:, i]
            col = np.where(col > 50, 900, col)
            col = np.where(np.isnan(col) | np.isinf(col), 900, col)
            self.pmf_function[:, i] = col

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
    def __init__(self, max_size=200, path=None, center=0, type="harmonic", force_constant=0.0):
        self.max_size = max_size
        self.type = type
        self.center = center
        if type == "linear":
            if path is None:
                self.bias = np.zeros(max_size)
            else:
                self.bias = np.loadtxt(path)
        elif type == "harmonic":
            self.center = center
            self.force_constant = force_constant
        else:
            raise ValueError("Invalid bias type")

    def denergy(self, new, old):
        # Hard coding massive bias for moves that lead to clusters larger than max_size
        # Might need to move this to acceptance criteria in order to avoid overflow errors
        if new >= self.max_size:
            bias = 100000
        else:
            if self.type == "harmonic":
                bias = self.force_constant/2*(new-self.center)**2 - self.force_constant/2*(old-self.center)**2
            elif self.type == "linear":
                bias = self.bias[new-1] - self.bias[old-1]
        return bias
    
    def energy(self, size):
        if size >= self.max_size:
            bias = 100000
        else:
            if self.type == "harmonic":
                bias = self.force_constant/2*(size-self.center)**2
            elif self.type == "linear":
                bias = self.bias[size-1]
        return bias
    
    # Update bias for adaptive US
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
