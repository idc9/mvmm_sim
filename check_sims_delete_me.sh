#!/bin/sh

# This script runs all the simulations and data analyses from (Carmichael, 2020). It assumes you first run git clone www.github.com/idc9/mvmm_sim and cd into the cloned directory.


#########
# setup #
#########

# conda environment
# conda create -n repro_mvmm python=3.6
# source activate repro_mvmm

# pip install .

# install required packages
# pip install mvmm==0.0.2
# pip install -r requirements.txt
# TODO: specify tag
# pip install git+https://github.com/idc9/explore.git@sometag
# pip install git+https://github.com/pca/pca.git@sometag


# python simulation_scripts/submit_full_sim.py --pi_name beads_2_5 --mini --sim_name test_run
# exit

# 5 2x2 blocks, uneven signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name beads_2_5 --clust_mean_std_v1 1 --clust_mean_std_v2 .5 --n_feats 10 --n_init 20 --bd_start_max_n_steps 10 --bd_final_max_n_steps 200 --exclude_log_pen_mvmm --n_mc_reps 3 --n_tr_samples_seq exp --sim_name beads_2_5__1__.5__exp --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'

