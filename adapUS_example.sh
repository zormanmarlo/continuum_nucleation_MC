# Command to run an adaptive US simulation of 1 M NaCl with the Dang forcefield
# Maximum cluster size is a 30mer
# Each iteration lasts for 100000 steps
# Using 5 markov chains
python simulation.py -jobname 1M_adapUS -config configs/adapUS_config.txt -adapUS -np 5
