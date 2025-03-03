import os
# import sys
import shutil
import argparse
# import time

# import cProfile
# import pstats

import numpy as np
import pickle as pkl
import multiprocessing as mp
# import matplotlib.pyplot as plt
# from scipy.stats import ks_2samp

from system import System
from utils import *

class Simulation:
    def __init__(self, config_file, jobname, ID=0):
        self.parameters = self.read_config(config_file)
        self.ID = str(ID).zfill(2)

        self.parameters["seed"] = self.parameters["seed"] + ID
        print(f"SEED from config: {self.parameters['seed']}")
        np.random.seed(self.parameters["seed"])
        
        self.system = System(self.parameters["box_length"], self.parameters["num_particles"], s=self.parameters["seed"], target_max=self.parameters["max_target"], 
                             pmf_path=self.parameters["ff_file"], bias_path=self.parameters["bias_file"], upper_cutoff=self.parameters["upper_cutoff"], 
                             lower_cutoff=self.parameters["lower_cutoff"], clust_cutoff=self.parameters["clust_cutoff"], max_trans=self.parameters["max_trans"],
                             counter_cutoff=self.parameters["counter_cutoff"])
        
        self.system.init(input_path=self.parameters["input_file"])
        self.clust_sizes, self.target_clust = self.system.find_clusters()
        self.target_sizes = [len(self.target_clust)]

        self.jobname = jobname
        self.avbmc_rate = self.parameters["avbmc_rate"]
        self.translation_rate = self.parameters["translation_rate"]
        self.nvt_rate = self.parameters["nvt_rate"]
        if self.parameters["nvt_rate"] + self.parameters["avbmc_rate"] + self.parameters["translation_rate"] != 1:
            raise ValueError("Rates do not sum to one")
        with open(f'{self.jobname}/stats-{self.ID}.log', 'a') as f:
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
        if os.path.exists(f'{self.jobname}/E-{self.ID}.log'):
            os.remove(f'{self.jobname}/E-{self.ID}.log')
        if os.path.exists(f'{self.jobname}/traj-{self.ID}.xyz'):
            os.remove(f'{self.jobname}/traj-{self.ID}.xyz')
        if os.path.exists(f'{self.jobname}/clusters-{self.ID}.out'):
            os.remove(f'{self.jobname}/clusters-{self.ID}.out')
        if os.path.exists(f'{self.jobname}/target_cluster-{self.ID}.out'):
            os.remove(f'{self.jobname}/target_cluster-{self.ID}.out')
        if os.path.exists(f'{self.jobname}/stats-{self.ID}.log'):
            os.remove(f'{self.jobname}/stats-{self.ID}.log')

    def write_output(self, step):
        with open(f'{self.jobname}/E-{self.ID}.log', 'a') as f:
            f.write(f'{step} {self.system.energy} {self.system.bias_energy}\n')
        with open(f'{self.jobname}/traj-{self.ID}.xyz', 'a') as f:
            f.write(f'  {self.parameters["num_particles"]}\n')
            f.write(f'  Step: {step}\n')
            for i, particle in enumerate(self.system.positions):
                if self.system.types[i] == 0:
                    f.write(f'H {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')
                else:
                    f.write(f'O {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')
        with open(f'{self.jobname}/clusters-{self.ID}.out', 'a') as f:
            clust_size_dist = np.histogram(self.clust_sizes, bins=np.arange(1, np.max(self.clust_sizes)+2))[0]
            f.write(f'{step} {clust_size_dist}\n')
        with open(f'{self.jobname}/target_cluster-{self.ID}.out', 'a') as f:
            f.write(f'{step} {len(self.target_clust)} {self.target_clust}\n')
        with open(f'{self.jobname}/stats-{self.ID}.log', 'a') as f:
            f.write(f'{step} {self.system.rejections[0]/(self.system.attempts[0]+1)} {self.system.rejections[1]/(self.system.attempts[1]+1)} {self.system.rejections[2]/(self.system.attempts[2]+1)} {self.system.rejections[3]/(self.system.attempts[3]+1)} {self.system.rejections[4]/(self.system.attempts[4]+1)}\n')
        
        for i in range(len(self.system.rejections)):
            self.system.rejections[i] = 0
            self.system.attempts[i] = 0
        
    def step(self):
        tmp = np.random.rand()
        # Translation move
        if tmp <= self.parameters["translation_rate"]:
            particle = np.random.randint(self.parameters["num_particles"])
            self.system.translation(particle)
        # AVBMC move
        elif tmp <= self.parameters["translation_rate"] + self.parameters["avbmc_rate"]:
            tmp2 = np.random.rand()
            particle = np.random.randint(self.parameters["num_particles"])
            Nin, Nin_idx = self.system.calc_in(particle)

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
            # in -> out move
            if (tmp2 <= 0.5 and Nin >= 1):
                self.system.nvt_inout_AVBMC(particle, Nin_idx)
            # out -> in move
            else:
                self.system.nvt_outin_AVBMC(particle, Nin_idx)

def equal_hist(dist):
    max_diff = np.max(np.abs(np.diff(dist)))
    if max_diff <= 0.05 * np.mean(dist):
        return True
    else:
        return False

def production_run(sim):
    for s in range(1, sim.parameters["prod_steps"]):
        sim.step()
        # Niave translation acceptance rate adjustment
        if s % sim.parameters["internal_interval"] == 0:
            trans_rate = sim.system.rejections[0]/(sim.system.attempts[0]+1)
            if trans_rate < 0.55:
                sim.system.max_displacement *= 1.1
            if trans_rate > 0.65:
                sim.system.max_displacement *= 0.9
        if s % sim.parameters["output_interval"] == 0:
            sim.clust_sizes, sim.target_clust = sim.system.find_clusters()
            sim.write_output(s)
    return sim

# Equilibration runs do not output to log and trajectory files
def equilibration_run(sim):
    for s in range(1, sim.parameters["equil_steps"]):
        sim.step()
        # Niave translation acceptance rate adjustment
        if s % sim.parameters["internal_interval"] == 0:
            trans_rate = sim.system.rejections[0]/(sim.system.attempts[0]+1)
            if trans_rate < 0.55:
                sim.system.max_displacement *= 1.1
            if trans_rate > 0.65:
                sim.system.max_displacement *= 0.9
    return sim

if __name__ == "__main__":
    # Parse command line for settings
    parser = argparse.ArgumentParser(description='S-I-M-U-L-A-T-E')
    parser.add_argument('-np', type=int, default=1, help='Number of processors')
    parser.add_argument('-jobname', type=str, default='JOB', help='Name of the job')
    parser.add_argument('-config', type=str, default="config.txt", help="Configuration file")
    parser.add_argument('-US', type=bool, default=False, help="Run adaptive US or not")
    args = parser.parse_args()

    print("Starting job: "+str(args.jobname))

    # Set up directory and simulations
    if os.path.exists(args.jobname) and os.path.isdir(args.jobname):
        shutil.rmtree(args.jobname)
    try:
        os.makedirs(args.jobname, exist_ok=True)
        shutil.copy(args.config, f"{args.jobname}/config.txt")

    except OSError as error:
        print(f'error creating directory -- exiting {args.jobname}: {error}')
    simulations = [Simulation(args.config, args.jobname, ID=i) for i in range(args.np)]
    
    # Run simple simulation
    if not args.US:
        # Only pool if running more than one markov chain
        if args.np == 1:
            # uncomment profiling code to run cProfile
            # pr = cProfile.Profile()
            # pr.enable()
            equilibration_run(simulations[0])
            production_run(simulations[0])
            # pr.disable()
            # ps = pstats.Stats(pr).sort_stats('cumulative')
            # ps.print_stats(10)
        else:
            with mp.Pool(processe=args.np) as pool:
                print(f"Running {np} markov chains")
                print("Running equilibration")
                simulations = pool.map(equilibration_run, simulations)
                print("Running production")
                simulations = pool.map(production_run, simulations)

    # Run until potentials are converged
    else:
        # Run unbiased simulation
        with mp.Pool(processes=args.np) as pool:
            simulations = pool.map(equilibration_run, simulations, True)
            simulations = pool.map(production_run, simulations)

        # Run biased simulations, iteratively updating bias
        print("Generating initial bias")
        cont = True
        orig_prod_steps = simulations[0].parameters["prod_steps"]
        current_it = 0
        while cont:
            current_it += 1
            pkl.dump(simulations, open(f"{args.jobname}/system.pkl", "wb"))

            cluster_counts = np.concatenate([sim.target_sizes for sim in simulations])
            dist = np.histogram(cluster_counts, bins=np.arange(1, simulations[0].parameters["max_target"]+2))[0]
            with open(f"{args.jobname}/histograms.out", "a") as file:
                file.write(f"{dist}\n")
            
            for sim in simulations:
                sim.system.bias.update(dist)
                # Reset data after each iteration, unclear if this is the best way to do this
                sim.system.target_sizes = []
                sim.target_sizes = []
                if all(dist > 0):
                    sim.system.parameters.prod_steps = sim.system.parameters.prod_steps + orig_prod_steps*0.2
            potential = simulations[0].system.bias.bias
            with open(f"{args.jobname}/potentials.out", "a") as file:
                file.write(f"{potential}\n")
            
            with mp.Pool(processes=args.np) as pool:
                simulations = pool.map(production_run, simulations, True)
            print("updating bias")

            if equal_hist(dist):
                print(f"potential converged in {current_it} iterations -- ending run")
                potential = simulations[0].system.bias.bias
                
                # Save final system
                sizes = np.concatenate([sim.target_sizes[-2 * sim.parameters["num_steps"] // sim.parameters["output_interval"]:] for sim in simulations])
                dist = np.histogram(sizes, bins=np.arange(1, simulations[0].parameters["max_target"]+2))[0]
                with open(f"{args.jobname}/histograms.out", "a") as file:
                    file.write(f"{dist}\n")
                with open(f"{args.jobname}/potentials.out", "a") as file:
                    file.write(f"{potential[-1]}\n")
                
                pkl.dump(simulations, open(f"{args.jobname}/system.pkl", "wb"))
                cont = False