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

        self.pmf_function[:,4] = np.where(self.pmf_function[:,4] > 50, 900, self.pmf_function[:,4])
        self.pmf_function[:,4] = np.where(np.isnan(self.pmf_function[:,4]) | np.isinf(self.pmf_function[:,4]), 900, self.pmf_function[:,4])
        self.pmf_function[:,5] = np.where(self.pmf_function[:,5] > 50, 900, self.pmf_function[:,5])
        self.pmf_function[:,5] = np.where(np.isnan(self.pmf_function[:,5]) | np.isinf(self.pmf_function[:,5]), 900, self.pmf_function[:,5])

        if "JC" in path:
            self.ff_type = "JC"
            self.pmf_function[:,6] = np.where(self.pmf_function[:,6] > 50, 900, self.pmf_function[:,6])
            self.pmf_function[:,6] = np.where(np.isnan(self.pmf_function[:,6]) | np.isinf(self.pmf_function[:,6]), 900, self.pmf_function[:,6])
        else:
            self.ff_type = "dang"
            
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
        # Might need to move this to acc