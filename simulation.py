import os
import sys
import shutil
import argparse
import time

import cProfile
import pstats

import numpy as np
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from system import System
from utils import *

class Simulation:
    def __init__(self, config_file, jobname, ID=0, path="."):
        self.parameters = self.read_config(config_file)
        self.ID = str(ID).zfill(2)
        self.path = path

        self.parameters["seed"] = self.parameters["seed"] + ID
        print(f"Using seed from from config file: {self.parameters['seed']}")
        np.random.seed(self.parameters["seed"])
        
        self.system = System(self.parameters["box_length"], self.parameters["num_particles"], s=self.parameters["seed"], target_max=self.parameters["max_target"], 
                             pmf_path=self.parameters["ff_file"], bias_path=self.parameters["bias_file"], upper_cutoff=self.parameters["upper_cutoff"], 
                             lower_cutoff=self.parameters["lower_cutoff"], clust_cutoff=self.parameters["clust_cutoff"], max_trans=self.parameters["max_trans"],
                             counter_cutoff=self.parameters["counter_cutoff"], center=self.parameters["center"], k=self.parameters["k"], id=self.ID)
        
        self.system.init(input_path=self.parameters["input_file"])
        self.clust_sizes, self.target_clust = self.system.find_clusters()
        self.target_sizes = [len(self.target_clust)]

        self.jobname = jobname
        self.avbmc_rate = self.parameters["avbmc_rate"]
        self.translation_rate = self.parameters["translation_rate"]
        self.nvt_rate = self.parameters["nvt_rate"]
        self.swap_rate = self.parameters["swap_rate"]
        if self.parameters["nvt_rate"] + self.parameters["avbmc_rate"] + self.parameters["translation_rate"] + self.parameters["swap_rate"] < 0.99:
            print(self.parameters["nvt_rate"] + self.parameters["avbmc_rate"] + self.parameters["translation_rate"] + self.parameters["swap_rate"])
            raise ValueError("Rates do not sum to one")
        with open(f'{self.path}/{self.jobname}/stats-{self.ID}.log', 'a') as f:
            f.write(f'# step translations in-out out-in target-in-out target-out-in\n')

    def read_config(self, config_file):
        with open(config_file, 'r') as f:
            parameters = {}
            for line in f:
                if "#" not in line:
                    tmp = line.split()
                    if tmp == []:
                        continue
                    key, value = tmp[0], tmp[-1]
                    if value.isdigit():
                        parameters[key] = int(value)
                    elif is_float(value):
                        parameters[key] = float(value)
                    else:
                        if value.lower() == "none":
                            parameters[key] = None
                        else:
                            parameters[key] = value
        return parameters
    
    def clean_dir(self):
        if os.path.exists(f'{self.path}/{self.jobname}/colvar_{self.system.bias.center}.out'):
            os.remove(f'{self.path}/{self.jobname}/colvar_{self.system.bias.center}.out')
        if os.path.exists(f'{self.path}/{self.jobname}/E-{self.ID}.log'):
            os.remove(f'{self.path}/{self.jobname}/E-{self.ID}.log')
        if os.path.exists(f'{self.path}/{self.jobname}/traj-{self.ID}.xyz'):
            os.remove(f'{self.path}/{self.jobname}/traj-{self.ID}.xyz')
        if os.path.exists(f'{self.path}/{self.jobname}/clusters-{self.ID}.out'):
            os.remove(f'{self.path}/{self.jobname}/clusters-{self.ID}.out')
        if os.path.exists(f'{self.path}/{self.jobname}/target_cluster-{self.ID}.out'):
            os.remove(f'{self.path}/{self.jobname}/target_cluster-{self.ID}.out')
        if os.path.exists(f'{self.path}/{self.jobname}/stats-{self.ID}.log'):
            os.remove(f'{self.path}/{self.jobname}/stats-{self.ID}.log')

    def write_output(self, step):
        if self.system.bias.center != 0:
            with open(f"{self.path}/{self.jobname}/colvar_{self.system.bias.center}.out", "a") as f:
                f.write(f"{step} {len(self.target_clust)} {self.system.bias_energy}\n")
        with open(f'{self.path}/{self.jobname}/E-{self.ID}.log', 'a') as f:
            f.write(f'{step} {self.system.energy} {self.system.bias_energy}\n')
        with open(f'{self.path}/{self.jobname}/traj-{self.ID}.xyz', 'a') as f:
            f.write(f'  {self.parameters["num_particles"]}\n')
            f.write(f'  Step: {step}\n')
            for i, particle in enumerate(self.system.positions):
                if self.system.types[i] == 0:
                    f.write(f'H {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')
                else:
                    f.write(f'O {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')
        with open(f'{self.path}/{self.jobname}/clusters-{self.ID}.out', 'a') as f:
            clust_size_dist = np.histogram(self.clust_sizes, bins=np.arange(1, np.max(self.clust_sizes)+2))[0]
            f.write(f'{step} {clust_size_dist}\n')
        with open(f'{self.path}/{self.jobname}/target_cluster-{self.ID}.out', 'a') as f:
            f.write(f'{step} {len(self.target_clust)} {self.target_clust}\n')
        with open(f'{self.path}/{self.jobname}/stats-{self.ID}.log', 'a') as f:
            f.write(f'{step} {self.system.rejections[0]/(self.system.attempts[0]+1)} {self.system.rejections[1]/(self.system.attempts[1]+1)} {self.system.rejections[2]/(self.system.attempts[2]+1)} {self.system.rejections[3]/(self.system.attempts[3]+1)} {self.system.rejections[4]/(self.system.attempts[4]+1)}\n')
        
        for i in range(len(self.system.rejections)):
            self.system.rejections[i] = 0
            self.system.attempts[i] = 0
        
    def step(self, step):
        tmp = np.random.rand()
        init_coords = self.system.positions[0].copy()
        # Translation move
        if tmp <= self.parameters["translation_rate"]:
            # for particle in range(1, self.parameters["num_particles"]):
            #     self.system.translation(particle)
            particle = np.random.randint(self.parameters["num_particles"])
            # # while particle == 0:
            # #     particle = np.random.randint(self.parameters["num_particles"])
            self.system.translation(particle)
        # AVBMC move
        elif tmp <= self.parameters["translation_rate"] + self.parameters["swap_rate"]:
            particle = np.random.randint(self.parameters["num_particles"])
            # while particle == 0:
            #     particle = np.random.randint(self.parameters["num_particles"])
            self.system.swap(particle)
        elif tmp <= self.parameters["translation_rate"] + self.parameters["swap_rate"] + self.parameters["avbmc_rate"]:
            tmp2 = np.random.rand()
            particle = np.random.randint(self.parameters["num_particles"])
            Nin, Nin_idx = self.system.calc_in(particle)
            if (Nin == 1 and (0 in Nin_idx)):
                tmp = 1
            # in -> out move
            if (tmp2 <= 0.5 and Nin >= 1):
                self.system.inout_AVBMC(particle)
            # out -> in move
            else:
                self.system.outin_AVBMC(particle)
        # NVT Nucleation move
        else:
            self.target_clust = self.system.find_cluster_around_target()
            particle = np.random.choice(self.system.target_clust_idx)
            Nin, Nin_idx = self.system.calc_in(particle)
            tmp2 = np.random.rand()
            if (Nin == 1 and (0 in Nin_idx)):
                tmp = 1
            # in -> out move
            if (tmp2 <= 0.5 and Nin >= 1):
                self.system.nvt_inout_AVBMC(particle, Nin_idx)
            # out -> in move
            else:
                self.system.nvt_outin_AVBMC(particle, Nin_idx)

def production_run(sim):
    current_min = 1
    for s in range(1, sim.parameters["prod_steps"]):
        sim.step(s)
        # Niave translation acceptance rate adjustment
        if s % sim.parameters["internal_interval"] == 0:
            # for some reason system trends to infintely large max_displacement for dilute systems desptie these checks
            # for now we only adjust max_displacement for concentrated systems, since acceptance rates are fine for dilute systems anyway
            if sim.system.concentration > 0.5:
                current_max = sim.system.max_displacement
                trans_rate = sim.system.rejections[0]/(sim.system.attempts[0]+1)
                if trans_rate < 0.55:
                    sim.system.max_displacement *= 1.1
                if trans_rate > 0.65:
                    sim.system.max_displacement *= 0.9
                if trans_rate > 10:
                    sim.system.max_displacement = current_max
                elif trans_rate < 0.01:
                    sim.system.max_displacement = current_max
        if s % sim.parameters["output_interval"] == 0:
            sim.clust_sizes, sim.target_clust = sim.system.find_clusters()
            sim.write_output(s)
    return sim

def equilibration_run(sim):
    for s in range(1, sim.parameters["equil_steps"]):
        sim.step(s)
        # Niave translation acceptance rate adjustment
        if s % sim.parameters["internal_interval"] == 0:
            # for some reason system trends to infintely large max_displacement for dilute systems desptie these checks
            # for now we only adjust max_displacement for concentrated systems, since acceptance rates are fine for dilute systems anyway
            if sim.system.concentration > 0.5:
                current_max = sim.system.max_displacement
                trans_rate = sim.system.rejections[0]/(sim.system.attempts[0]+1)
                if trans_rate < 0.55:
                    sim.system.max_displacement *= 1.1
                if trans_rate > 0.65:
                    sim.system.max_displacement *= 0.9
                if trans_rate > 10:
                    sim.system.max_displacement = current_max
                elif trans_rate < 0.01:
                    sim.system.max_displacement = current_max
    return sim

if __name__ == "__main__":
    # Parse command line for settings
    parser = argparse.ArgumentParser(description='S-I-M-U-L-A-T-E')
    parser.add_argument('-np', type=int, default=1, help='Number of processors')
    parser.add_argument('-jobname', type=str, default='JOB', help='Name of the job')
    parser.add_argument('-config', type=str, default="config.txt", help="Configuration file")
    parser.add_argument('-path', type=str, default=".", help="Path to save output")
    args = parser.parse_args()

    print("Starting job: "+str(args.jobname))

    # Set up directory and simulations
    if os.path.exists(args.path+"/"+args.jobname) and os.path.isdir(args.path+"/"+args.jobname):
        shutil.rmtree(args.path+"/"+args.jobname)
    try:
        os.makedirs(args.path+"/"+args.jobname, exist_ok=True)
        shutil.copy(args.config, f"{args.path}/{args.jobname}/config.txt")

    except OSError as error:
        print(f'error creating directory -- exiting {args.path+"/"+args.jobname}: {error}')
    simulations = [Simulation(args.config, args.jobname, ID=i, path=args.path) for i in range(args.np)]
    
    # Run simple simulation
    # Only pool if running more than one markov chain
    if args.np == 1:
        # pr = cProfile.Profile()
        # pr.enable()
        equilibration_run(simulations[0])
        production_run(simulations[0])
        # pr.disable()
        # ps = pstats.Stats(pr).sort_stats('cumulative')
        # ps.print_stats(10)
    else:
        with mp.Pool(processes=args.np) as pool:
            print(f"Running {np} markov chains")
            print("Running equilibration")
            simulations = pool.map(equilibration_run, simulati