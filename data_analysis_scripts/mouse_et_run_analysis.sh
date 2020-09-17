#!/bin/sh

# v0: transcriptomic
# v1: ephys

transcr_data='select_markers_pca'

# transcriptomic
min_ncomp_v0=1
max_ncomp_v0=60
bic_sel_ncomp_v0=41

# ephys
min_ncomp_v1=1
max_ncomp_v1=60
bic_sel_ncomp_v1=47

n_init=5
init_params=kmeans # rand_pts, kmeans


n_blocks_seq=10

node="b18"

# set workind directory
cwd=$(pwd)
IFS='/' read -ra SPLIT_CWD <<< "$cwd"
top_name=${SPLIT_CWD[1]}
if [ $top_name == "Users" ]; then
    sim_dir="/Users/iaincarmichael/Dropbox/Research/mvmm/public_release/"
else
    sim_dir="/home/guests/idc9/projects/mvmm/public_release/"
fi

name=analysis

# setup paths
top_dir="$sim_dir""mouse_et/"
raw_data_dir="$top_dir""raw_data/"
pro_data_dir="$top_dir""pro_data/"
results_dir="$top_dir""$name"/


fpath_v0="$pro_data_dir""transcriptomic_select_markers_pca_feats.csv"
super_fpath_v0="$pro_data_dir""transcriptomic_select_markers.csv"


fpath_v1="$pro_data_dir""ephys_pca_feats.csv"
super_fpath_v1=''

vars2compare_fpath="$pro_data_dir""vars2compare.csv"


################
# run analysis #
################

# source activate mvmm

# checked
# python mouse_et_process_data.py

# python single_view_fit_models.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --n_init "$n_init" --gmm_init_params "$init_params" --min_ncomps "$min_ncomp_v0" "$min_ncomp_v1" --max_ncomps "$max_ncomp_v0" "$max_ncomp_v1" --exclude_cat_gmm --max_n_steps 200 --n_jobs -1 --queue w-bigmem.q --submit --mem 20G --node "$node"

# todo
# python single_view_model_selection.py --results_dir "$results_dir"

# todo
# python single_view_interpret.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --vars2compare_fpath "$vars2compare_fpath" --super_fpaths "$super_fpath_v0"  "$super_fpath_v1"

# todo
# python mvmm_n_comp_submitter.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --min_n_view_comps "$bic_sel_ncomp_v0" "$bic_sel_ncomp_v1" --max_n_view_comps "$bic_sel_ncomp_v0" "$bic_sel_ncomp_v1" --n_blocks_seq "$n_blocks_seq" --n_init "$n_init" --gmm_init_params "$init_params" --bd_start_max_n_steps 10 --bd_final_max_n_steps 100  --bd_init_pen_use_bipt_sp --log_pen_n_pen_seq 100 --exclude_log_pen_mvmm --n_jobs -1 --queue w-bigmem.q --submit --mem 8G --node "$node"

# checked
python mvmm_model_selection.py --results_dir "$results_dir"

# checked
python mvmm_interpret.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --vars2compare_fpath "$vars2compare_fpath" --super_fpaths "$super_fpath_v0"  "$super_fpath_v1"
# checked
python mouse_et_ephys_viz.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1"
