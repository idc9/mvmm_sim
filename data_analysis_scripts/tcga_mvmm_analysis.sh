#!/bin/sh

v0='rna'
# v1=mi_rna, cp, dna_meth

cancer_type=$1 # 'BRCA'
feat_list=icluster  # always use the icluster feature list

v1=$2  # specify the second view
# node=$4

n_init=1
init_params=kmeans

n_blocks_seq=10


# check argument supplied
if [ -z "$1" ]
  then
    echo "No argument supplied for cancer type"
    exit
fi

if [ -z "$2" ]
  then
    echo "No argument supplied for the second view"
    exit
fi


# set working directory
cwd=$(pwd)
IFS='/' read -ra SPLIT_CWD <<< "$cwd"
top_name=${SPLIT_CWD[1]}
if [ $top_name == "Users" ]; then
    sim_dir="/Users/iaincarmichael/Dropbox/Research/mvmm/public_release/"
    pro_data_dir="/Volumes/Samsung_T5/tcga/pro_data/""$cancer_type"/

else
    sim_dir="/home/guests/idc9/projects/mvmm/public_release/"
    pro_data_dir="$sim_dir"tcga/pro_data/"$cancer_type"/

fi

# where all the results will be saved
name="$cancer_type"__"$feat_list"__"$v0"__"$v1"__MVMM
results_dir="$sim_dir""tcga"/"$name"/

# set fpaths to analysis data files
feat_save_dir="$pro_data_dir""$feat_list"/
fpath_v0="$feat_save_dir""$v0".csv
fpath_v1="$feat_save_dir""$v1".csv


# metadata for interpretation step
vars2compare_fpath="$pro_data_dir""vars2compare.csv"
metadata_fpath="$pro_data_dir""metadata.csv"

duration_col='PFI.time.1'
event_col='PFI.1'

# BIC selected number of components from single view analyses
if [ $cancer_type == "BRCA" ]
then
    rna_bic_ncomp=4
    mi_rna_bic_ncomp=3
    dna_meth_bic_ncomp=5
    cp_bic_ncomp=7
fi


# set view n comp to BIC selected n components
n_comp_v0="$rna_bic_ncomp"

if [ $v1 == "rna" ]
then
    n_comp_v1="$rna_bic_ncomp"

elif [ $v1 == "mi_rna" ]
then
    n_comp_v1="$mi_rna_bic_ncomp"

elif [ $v1 == "dna_meth" ]
then
    n_comp_v1="$dna_meth_bic_ncomp"

elif [ $v1 == "cp" ]
then
    n_comp_v1="$cp_bic_ncomp"
fi


################
# run analysis #
################
# the commented out code was used for the cluster Iain used to run the experiments

source activate mvmm

python mvmm_n_comp_submitter.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --min_n_view_comps "$min_ncomp_v0" "$min_ncomp_v1" --max_n_view_comps "$max_ncomp_v0" "$max_ncomp_v1" --n_init "$n_init" --gmm_init_params "$init_params" --bd_start_max_n_steps 10 --bd_final_max_n_steps 100 --bd_init_pen_use_bipt_sp --n_jobs -1 # --queue w-bigmem.q --submit --mem 8G --node "$node"

python mvmm_fit_models.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --n_view_comps "$n_comp_v0" "$n_comp_v1" --n_blocks_seq "$n_blocks_seq" --n_init "$n_init" --gmm_init_params "$init_params" --bd_start_max_n_steps 10 --bd_final_max_n_steps 100 --bd_init_pen_use_bipt_sp --exclude_log_pen_mvmm --n_jobs -1 # --queue w-bigmem.q --submit --mem 50G --node "$node"


python mvmm_model_selection.py --results_dir "$results_dir"

python mvmm_interpret.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --vars2compare_fpath "$vars2compare_fpath" --survival_fpath "$metadata_fpath" --duration_col "$duration_col" --event_col "$event_col"

python tcga_survival.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" --metadata_fpath "$metadata_fpath" --duration_col "$duration_col" --event_col "$event_col"
