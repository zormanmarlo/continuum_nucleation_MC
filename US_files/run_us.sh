#!/bin/bash

for i in $(seq 2 2 44); do
	sed "s/CENTER/${i}/g" configs/nacl_us/100mM_nacl_template_large_random_JC.txt > configs/nacl_us/100mM_nacl_${i}mer_large_random_noswitch_JC.txt
    	sed "s/CENTER/${i}/g" submit_template.sh > submit_tmp.sh
	sbatch submit_tmp.sh
	rm submit_tmp.sh
done
