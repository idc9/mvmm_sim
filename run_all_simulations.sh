#!/bin/sh

# This script runs all the simulations and data analyses from (Carmichael, 2020). It assumes you first run git clone www.github.com/idc9/mvmm_sim and cd into the cloned directory.

#########
# setup #
#########

# conda environment
conda create -n repro_mvmm python=3.6
conda activate repro_mvmm

pip install .

# install required packages
# pip install mvmm==0.0.2
pip install -r requirements.txt

# TODO: specify tag
# Manually install a couple packages
# pip install git+https://github.com/idc9/explore.git@sometag
# pip install git+https://github.com/idc9/ya_pca.git@sometag


# uncomment this line to test the code with a small scale simulation
# python simulation_scripts/submit_full_sim.py --pi_name beads_2_5 --mini --sim_name test_run

##############################
# Synthetic data simulations #
##############################

# the commented out lines code is used for the cluster Iain ran
# the simulations on. The --submit_sep_sims command is useful if you
# want to parallelize the simulations over multiple cluster nodes,
# granted you will have to write some code to do so yourself

# 5 2x2 blocks, uneven signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name beads_2_5 --clust_mean_std_v1 1 --clust_mean_std_v2 .5 --n_feats 10 --n_init 20 --bd_start_max_n_steps 10 --bd_final_max_n_steps 200 --log_pen_start_max_n_steps 10 --log_pen_final_max_n_steps 200 --n_mc_reps 20 --sim_name beads_2_5__1__.5 # --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'


# 5 2x2 blocks, even signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name beads_2_5 --clust_mean_std_v1 1 --clust_mean_std_v2 1 --n_feats 10 --n_init 20 --bd_start_max_n_steps 10 --bd_final_max_n_steps 200 --log_pen_start_max_n_steps 10 --log_pen_final_max_n_steps 200 --n_mc_reps 20 --sim_name beads_2_5__1__1 # --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'


# lollipop with one 5x5 and 5 1x1 blocks, uneven signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name lollipop_5_5 --clust_mean_std_v1 1 --clust_mean_std_v2 .5 --n_feats 10 --n_init 20 --bd_start_max_n_steps 10 --bd_final_max_n_steps 200 --log_pen_start_max_n_steps 10 --log_pen_final_max_n_steps 200 --n_mc_reps 20 --sim_name lollipop_5_5__1__.5 # --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'

# lollipop with one 5x5 and 5 1x1 blocks, even signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name lollipop_5_5 --clust_mean_std_v1 1 --clust_mean_std_v2 1 --n_feats 10 --n_init 20 --bd_start_max_n_steps 10 --bd_final_max_n_steps 200 --log_pen_start_max_n_steps 10 --log_pen_final_max_n_steps 200 --n_mc_reps 20 --sim_name lollipop_5_5__1__1 # --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'

# sparse pi, uneven signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name sparse_pi --clust_mean_std_v1 1 --clust_mean_std_v2 .5 --n_feats 10 --n_init 20 --exclude_bd_mvmm --log_pen_start_max_n_steps 10 --log_pen_final_max_n_steps 200 --n_mc_reps 20 --sim_name sparse_pi__1__.5 # --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'

# sparse pi, uneven signal to noise ratios
python simulation_scripts/submit_full_sim.py --pi_name sparse_pi --clust_mean_std_v1 1 --clust_mean_std_v2 1 --n_feats 10 --n_init 20 --exclude_bd_mvmm --log_pen_start_max_n_steps 10 --log_pen_final_max_n_steps 200 --n_mc_reps 20 --sim_name sparse_pi__1__1 # --queue w-bigmem.q --submit --submit_sep_sims --mem 8G --node 'b15|b16|b17|b18'


python simulation_scripts/agg_sim_results --sim_name beads_2_5__1__.5
python simulation_scripts/agg_sim_results --sim_name beads_2_5__1__1
python simulation_scripts/agg_sim_results --sim_name lollipop_5_5__1__.5
python simulation_scripts/agg_sim_results --sim_name lollipop_5_5__1__1
python simulation_scripts/agg_sim_results --sim_name sparse_pi__1__.5
python simulation_scripts/agg_sim_results --sim_name sparse_pi__1__1

######################
# Real data analysis #
######################

# mouse ET analysis
sh data_analysis_scripts/mouse_et_run_analysis.sh

# single view analysis on BRCA data with icluster features
sh data_analysis_scripts/tcga_single_view_analysis.sh

# multi-vew analysis on BRCA data with icluster features with RNA vs. XXX
sh data_analysis_scripts/tcga_mvmm_analysis.sh mi_rna
sh data_analysis_scripts/tcga_mvmm_analysis.sh dna_meth
sh data_analysis_scripts/tcga_mvmm_analysis.sh cp
