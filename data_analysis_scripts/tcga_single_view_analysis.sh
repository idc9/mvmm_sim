#!/bin/sh

v0='rna'
v1='mi_rna'
v2='dna_meth'
v3='cp'

cancer_type=$1 # 'BRCA'
feat_list=icluster

pca=true

# node="b18"
node=$3

init_params=kmeans # rand_pts, kmeans

# n_init=20
# min_ncomps=1

# check argument supplied
if [ -z "$1" ]
  then
    echo "No argument supplied for cancer type"
    exit
fi

if [ -z "$2" ]
  then
    echo "No argument supplied for feature list"
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

if [ $pca == "true" ]; then
    name="$name"__pca_feats

    fpath_v0="$feat_save_dir""$v0"_pca_feats.csv
    fpath_v1="$feat_save_dir""$v1"_pca_feats.csv
    fpath_v2="$feat_save_dir""$v2"_pca_feats.csv
    fpath_v3="$feat_save_dir""$v3"_pca_feats.csv

    super_fpath_v0="$feat_save_dir""$v0".csv
    super_fpath_v1="$feat_save_dir""$v1".csv
    super_fpath_v2="$feat_save_dir""$v2".csv
    super_fpath_v3="$feat_save_dir""$v3".csv

    abs_tol=1e-8

    # max_ncomps=30

else
    fpath_v0="$feat_save_dir""$v0".csv
    fpath_v1="$feat_save_dir""$v1".csv
    fpath_v2="$feat_save_dir""$v2".csv
    fpath_v3="$feat_save_dir""$v3".csv

    super_fpath_v0=''
    super_fpath_v1=''
    super_fpath_v2=''
    super_fpath_v3=''

    abs_tol=1e-10

    # max_ncomps=50
fi


if [ $cancer_type == "BRCA" ]
then
    if [ $feat_list == 'icluster' ]
    then
        n_init=100

        if [ $pca == "true" ]
        then
            min_ncomps=1
            max_ncomps=20
        else
            min_ncomps=1
            max_ncomps=50
        fi

    elif [ $feat_list == 'all' ]
    then
        n_init=100

        if [ $pca == "true" ]
        then
            min_ncomps=1
            max_ncomps=30
        else
            exit
            min_ncomps=TODO
            max_ncomps=TODO
        fi

    fi

elif [ $cancer_type == "pan" ]
then

    if [ $feat_list == "icluster" ]
    then
        if [ $pca == "true" ]
        then
            # TODO: rerun
            n_init=20

            min_ncomps=1
            max_ncomps=80

        else
            # TODO: rerun
            n_init=20

            min_ncomps=50
            max_ncomps=150
        fi

    elif [ $feat_list == "all" ]
    then


        if [ $pca == "true" ]
        then
            n_init=20
            min_ncomps=1
            max_ncomps=100
        else

            exit
            min_ncomps=TODO
            max_ncomps=TODO
        fi
    fi

elif [ $cancer_type == "LUAD" ]
then

    if [ $feat_list == "icluster" ]
    then
        if [ $pca == "true" ]
        then
            n_init=20

            min_ncomps=1
            max_ncomps=10

        else
            n_init=20

            min_ncomps=1
            max_ncomps=20
        fi

    elif [ $feat_list == "all" ]
    then


        if [ $pca == "true" ]
        then
            n_init=20
            min_ncomps=TODO
            max_ncomps=TODO
        else

            exit
            min_ncomps=TODO
            max_ncomps=TODO
        fi
    fi
fi

results_dir="$sim_dir""tcga"/"$name"/

vars2compare_fpath="$pro_data_dir""vars2compare.csv"
metadata_fpath="$pro_data_dir""metadata.csv"

duration_col='PFS.time'
event_col='PFS'


################
# run analysis #
################

# source activate mvmm

python tcga_process_data.py --cancer_type "$cancer_type" --feat_list "$feat_list" #  --queue w-bigmem.q --submit --mem 100G --node "$node"

# python tcga_pca.py --cancer_type "$cancer_type" --feat_list "$feat_list" # --queue w-bigmem.q --submit --mem 100G --node "$node"



# python single_view_fit_models.py --results_dir "$results_dir" --fpath "$fpath_v0" "$fpath_v1" "$fpath_v2" "$fpath_v3" --n_init "$n_init" --gmm_init_params "$init_params" --abs_tol "$abs_tol" --min_ncomps "$min_ncomps" "$min_ncomps" "$min_ncomps" "$min_ncomps" --max_ncomps "$max_ncomps" "$max_ncomps" "$max_ncomps" "$max_ncomps" --exclude_cat_gmm --n_jobs -1 #  --queue w-bigmem.q --submit --mem 20G --node "$node"

# python single_view_model_selection.py --results_dir "$results_dir"

# python single_view_interpret.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" "$fpath_v2" "$fpath_v3" --vars2compare_fpath "$vars2compare_fpath" --super_fpaths "$super_fpath_v0"  "$super_fpath_v1"  "$super_fpath_v2"  "$super_fpath_v3" --survival_fpath "$metadata_fpath" --duration_col "$duration_col" --event_col "$event_col"

# python tcga_survival.py --results_dir "$results_dir" --fpaths "$fpath_v0" "$fpath_v1" "$fpath_v2" "$fpath_v3" --metadata_fpath "$metadata_fpath" --duration_col "$duration_col" --event_col "$event_col"
