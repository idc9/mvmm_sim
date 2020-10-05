#!/bin/sh

v0='rna'
v1='mi_rna'
v2='dna_meth'
v3='cp'

cancer_type='BRCA'
feat_list=icluster

node=$1 # cluster node to use, you can probably ignore this

init_params=kmeans

abs_tol=1e-10
n_init=100

# How many components to searach over for model selection
min_ncomps=1
max_ncomps=50

# check argument supplied
if [ -z "$1" ]
  then
    echo "No argument supplied for cancer type"
    exit
fi


# set workind directory
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

feat_save_dir="$pro_data_dir"/"$feat_list"/


# name="$base_name"__"$cancer_type"__"$v0"__"$v1"
name="$cancer_type"__"$feat_list"__single_view_clustering

fpath_v0="$feat_save_dir""$v0".csv
fpath_v1="$feat_save_dir""$v1".csv
fpath_v2="$feat_save_dir""$v2".csv
fpath_v3="$feat_save_dir""$v3".csv

super_fpath_v0=''
super_fpath_v1=''
super_fpath_v2=''
super_fpath_v3=''

results_dir="$sim_dir""tcga"/"$name"/

vars2compare_fpath="$pro_data_dir""vars2compare.csv"
metadata_fpath="$pro_data_dir""metadata.csv"

duration_col='PFS.time'
event_col='PFS'

################
# run analysis #
################

python tcga_process_data.py --cancer_type "$cancer_type" --feat_list "$feat_list" #  --queue w-bigmem.q --submit --mem 100G --node "$node"

python tcga_pca.py --cancer_type "$cancer_type" --feat_list "$feat_list" # --queue w-bigmem.q --submit --mem 100G --node "$node"


python single_view_fit_models.py --results_dir "$results_dir" --fpath "$fpath_v0" "$fpath_v1" "$fpath_v2" "$fpath_v3" --n_init "$n_init" --gmm_init_params "$init_params" --abs_tol "$abs_tol" --min_ncomps "$min_ncomps" "$min_ncomps" "$min_ncomps" "$min_ncomps" --max_ncomps "$max_ncomps" "$max_ncomps" "$max_ncomps" "$max_ncomps" --exclude_cat_gmm --n_jobs -1 #  --queue w-bigmem.q --submit --mem 20G --node "$node"

python single_view_model_selection.py --results_dir "$results_dir"

python single_view_interpret.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" "$fpath_v2" "$fpath_v3" --vars2compare_fpath "$vars2compare_fpath" --super_fpaths "$super_fpath_v0"  "$super_fpath_v1"  "$super_fpath_v2"  "$super_fpath_v3" --survival_fpath "$metadata_fpath" --duration_col "$duration_col" --event_col "$event_col"
