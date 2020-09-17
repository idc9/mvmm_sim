from os.path import join
import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import argparse

# from pca.PCA import PCA
# from pca.viz import scree_plot
from time import time
from mvmm_sim.tcga.TCGAPaths import TCGAPaths
from mvmm_sim.simulation.Paths import Paths
from mvmm_sim.tcga.process_raw_data import filter_observations
from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser
from mvmm_sim.data_analysis.utils import apply_pd
from mvmm_sim.simulation.utils import make_and_get_dir
from mvmm_sim.simulation.ResultsWriter import ResultsWriter


# cancer_type = 'LUAD'  # 'GBM'  # 'BRCA'
parser = argparse.ArgumentParser(description='Process TCGA data.')

parser.add_argument('--cancer_type', default='BRCA',
                    help='Which cancer type e.g. LUAD, BCRA, OV.')

parser.add_argument('--handle_nans', default='impute',
                    choices=['impute', 'drop'],
                    help='How to handle missing values.')

parser.add_argument('--feat_list', default='icluster',
                    help='Which feature list to use e.g. icluster,'
                    'all, top_2000 by variace.')

parser = bayes_parser(parser)
args = parser.parse_args()
bayes_submit(args)

cancer_type = args.cancer_type
feat_list = args.feat_list
handle_nans = args.handle_nans

assert feat_list in ['all', 'icluster'] or 'top' in feat_list

filter_kws = {'tumor_sample_only': True,
              'primary_tumor_only': True,
              'keep_first_of_participant_multiples': True,
              'ensure_participant_idx': True,
              'verbose': True}

raw_data_dir = TCGAPaths().raw_data_dir
pro_data_dir = make_and_get_dir(TCGAPaths().pro_data_dir, cancer_type)
feat_save_dir = make_and_get_dir(pro_data_dir, feat_list)

res_writer = ResultsWriter(join(feat_save_dir, 'log.txt'),
                           delete_if_exists=True)

res_writer.write(args)

fnames = {
    'mi_rna': 'pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv',

    'rna': 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2-v2.geneExp.tsv',

    'dna_meth': 'jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv',

    'cp': 'all_data_by_genes_whitelisted.tsv'

}

start_time = time()

##################
# load  metadata #
##################
# get patients for given type

fpath = join(raw_data_dir, 'TCGA-CDR-SupplementalTableS1.xlsx')
metadata = pd.read_excel(fpath, index_col=0, sheet_name='ExtraEndpoints')
metadata = metadata.set_index('bcr_patient_barcode')

# only get patients with given cancer type
if cancer_type != 'pan':
    patient_barcodes = metadata.\
        query("type == '{}'".format(cancer_type)).index.values

else:
    patient_barcodes = metadata.index.values
filter_kws['patient_barcodes'] = patient_barcodes

# filter metadata
metadata, metadata_bc_info = filter_observations(metadata, **filter_kws)


###############################
# load iCluster feature lists #
###############################

if feat_list == 'icluster':
    fpath = join(raw_data_dir, 'pancan33.iCluster.features.csv')
    iclust_feats = pd.read_csv(fpath)

    iclust_feat_list = {'rna': iclust_feats.
                        query("platform == 'mRNA'")['feature'].values,

                        'mi_rna': iclust_feats.
                        query("platform == 'miRNA'")['feature'].values,

                        'dna_meth': iclust_feats.
                        query("platform == 'meth'")['feature'].values,

                        'cp': iclust_feats.
                        query("platform == 'CN'")['feature'].values
                        }

elif 'top' in feat_list:
    n_to_keep = int(feat_list.split('top_')[1])


def variance_screen(df, n_to_keep):
    if n_to_keep >= df.shape[1]:
        return df

    stds = df.std(axis=0)
    stds = stds.sort_values(ascending=False)
    vars2keep = stds.index.values[0:n_to_keep]
    return df[vars2keep]


########################
# load quality control #
########################

fpath = join(raw_data_dir, 'merged_sample_quality_annotations.tsv')
quality_control = pd.read_csv(fpath, delimiter='\t')

bad_aliquots = {'rna': quality_control.
                query("platform in "
                      "['IlluminaGA_RNASeqV2', 'IlluminaHiSeq_RNASeqV2']").
                query("Do_not_use")['aliquot_barcode'].values,

                'mi_rna': quality_control.
                query("platform in "
                      "['IlluminaGA_miRNASeq', 'IlluminaHiSeq_miRNASeq']").
                query("Do_not_use")['aliquot_barcode'].values,

                'dna_meth': quality_control.
                query("platform in ['HumanMethylation27', 'HumanMethylation450']").
                query("Do_not_use")['aliquot_barcode'].values,

                'cp': quality_control.
                query("platform == 'Genome_Wide_SNP_6'").
                query("Do_not_use")['aliquot_barcode'].values
                }


#############
# load data #
#############
# TODOs: drop normal breast cancers
# maybe drop the do not use samples

# RNA
res_writer.write("RNA")
fpath = join(raw_data_dir, fnames['rna'])
rna = pd.read_csv(fpath, index_col=0, delimiter='\t').T.astype(float)
rna, _ = filter_observations(rna,
                             aliquots2remove=bad_aliquots['rna'],
                             **filter_kws)  # filter observations
rna = rna.apply(lambda x: np.log2(1 + x))  # log RNA feats

# possibly screen variables
if feat_list == 'icluster':
    rna = rna[iclust_feat_list['rna']]  # restrict to iCluster features
elif 'top' in feat_list:
    rna = variance_screen(rna, n_to_keep=n_to_keep)

rna_pro = StandardScaler(with_mean=True, with_std=True)  # standarize RNA feats
rna = apply_pd(func=rna_pro.fit_transform, df=rna)


# miRNA
res_writer.write("miRNA")
fpath = join(raw_data_dir, fnames['mi_rna'])
mi_rna = pd.read_csv(fpath, index_col=0).T
mi_rna = mi_rna.drop(index='Correction').astype(float)
mi_rna, _ = filter_observations(mi_rna,
                                aliquots2remove=bad_aliquots['mi_rna'],
                                **filter_kws)
mi_rna = mi_rna.apply(lambda x: np.log2(1 + x))

# possibly screen variables
if feat_list == 'icluster':
    mi_rna = mi_rna[iclust_feat_list['mi_rna']]
elif 'top' in feat_list:
    mi_rna = variance_screen(mi_rna, n_to_keep=n_to_keep)

mi_rna_pro = StandardScaler(with_mean=True, with_std=True)
mi_rna = apply_pd(func=mi_rna_pro.fit_transform, df=mi_rna)


# DNA Meth
res_writer.write("DNA meth")
fpath = join(raw_data_dir, fnames['dna_meth'])
dna_meth = pd.read_csv(fpath, index_col=0, delimiter='\t').T
dna_meth, _ = filter_observations(dna_meth,
                                  aliquots2remove=bad_aliquots['dna_meth'],
                                  **filter_kws)

# possibly screen variables
if feat_list == 'icluster':
    dna_meth = dna_meth[iclust_feat_list['dna_meth']]
elif 'top' in feat_list:
    dna_meth = variance_screen(dna_meth, n_to_keep=n_to_keep)


# Copy number
res_writer.write("CP number")
fpath = join(raw_data_dir, fnames['cp'])
cp = pd.read_csv(fpath, index_col=0, delimiter='\t').T
cp_other_info = cp.iloc[0:2].T
cp = cp.drop(index=['Locus ID', 'Cytoband']).astype(float)
cp, _ = filter_observations(cp,
                            aliquots2remove=bad_aliquots['cp'],
                            **filter_kws)

# possibly screen variables
if feat_list == 'icluster':
    # cp = cp[iclust_feat_list['dna_meth']]
    print("TODO: figure out copy number iCluster features!!!!")
    cp = variance_screen(cp, n_to_keep=3000)

elif 'top' in feat_list:
    cp = variance_screen(cp, n_to_keep=n_to_keep)


# cp = cp[feat_list['cp']]
# most_var_cols = cp.std(axis=0).\
#     sort_values(ascending=False).index.values[0:2000]
# cp = cp[most_var_cols]

data = {'mi_rna': mi_rna, 'rna': rna, 'dna_meth': dna_meth, 'cp': cp}

################
# missing data #
################
for k in data.keys():
    n_missing = data[k].isna().values.sum()
    nvar = (data[k].isna().sum(axis=0) > 0).sum()
    nobs = (data[k].isna().sum(axis=0) > 0).sum()

    if n_missing > 0:
        res_writer.write('{} has {} missing values.'.format(k, n_missing))
        res_writer.write('variables with NaNs: {}'.format(nvar))
        res_writer.write('samples with NaNs {}'.format(nobs))

        if handle_nans == 'impute':
            imputer = KNNImputer(metric='nan_euclidean',
                                 n_neighbors=5, weights='uniform')

            data[k] = apply_pd(func=imputer.fit_transform, df=data[k])

        elif handle_nans == 'drop':
            data[k] = data[k].dropna(axis=1)

#####################
# subject alignment #
#####################

# get common subjects
patient_idxs = set(metadata.index.values)
for k in data.keys():
    patient_idxs = patient_idxs.intersection(data[k].index.values)
res_writer.write("Number of common patients".format(len(patient_idxs)))

# filter subjects to common subjects
for k in data.keys():
    data[k] = data[k].loc[patient_idxs]
metadata = metadata.loc[patient_idxs]

# save processed data
for k in data.keys():
    res_writer.write('{} final shape {}'.format(k, data[k].shape))

    fpath = join(feat_save_dir, '{}.csv'.format(k))

    data[k].name = k
    data[k].to_csv(fpath)

############################
# load additional metadata #
############################

res_writer.write("Subtype data")
fpath = join(raw_data_dir, 'subtypes.csv')
subtypes = pd.read_csv(fpath, index_col=0)
subtypes, subtypes_bc_info = filter_observations(subtypes, **filter_kws)

if cancer_type == 'pan':
    subtypes = subtypes[['cancer.type']]
    vars2compare = ['cancer.type']

if cancer_type == 'BRCA':
    subtypes = subtypes[['Subtype_mRNA']]
    vars2compare = ['Subtype_mRNA']

elif cancer_type == 'LUAD':
    subtypes = subtypes[['Subtype_DNAmeth']]
    vars2compare = ['Subtype_DNAmeth']

elif cancer_type == 'OV':
    subtypes = subtypes[['Subtype_mRNA']]
    vars2compare = ['Subtype_mRNA']


metadata = pd.merge(metadata, subtypes,
                    left_index=True, right_index=True, how='outer')


if cancer_type == 'BRCA':
    fpath = join(raw_data_dir, 'brca_tcga_clinical_data.tsv')
    clinical_data = pd.read_csv(fpath, delimiter='\t')

    cols2keep = ['Sample ID', 'IHC-HER2', 'ER Status By IHC',
                 'PR status by ihc', 'Cancer Type Detailed']  # 'Patient ID'

    clinical_data = clinical_data[cols2keep].set_index("Sample ID")

    # rename columns
    clinical_data = clinical_data.\
        rename(columns={'IHC-HER2': 'HER2_IHC',
                        'ER Status By IHC': 'ER_IHC',
                        'PR status by ihc': 'PR_IHC',
                        'Cancer Type Detailed': 'histological_type'})

    # rename histological types, group rare (< 10 cases) together
    hist_type_map = {'Breast Invasive Ductal Carcinoma': 'ductal',
                     'Breast Invasive Lobular Carcinoma': 'lobular',
                     'Breast Mixed Ductal and Lobular Carcinoma': 'mixed',
                     'Breast Invasive Mixed Mucinous Carcinoma': 'mucinous',
                     'Metaplastic Breast Cancer': 'metaplastic',

                     'Breast': 'other',
                     'Invasive Breast Carcinoma': 'other',
                     'Paget Disease of the Nipple': 'other',
                     'Solid Papillary Carcinoma of the Breast': 'other',
                     'Malignant Phyllodes Tumor of the Breast': 'other',
                     'Adenoid Cystic Breast Cancer': 'other',
                     'Basal Cell Carcinoma': 'other',
                     'Breast Invasive Carcinoma, NOS': 'other',
                     np.nan: np.nan}

    clinical_data['histological_type'] = clinical_data['histological_type'].\
        apply(lambda x: hist_type_map[x])

    clinical_data['Subtype_mRNA'] = clinical_data['Subtype_mRNA'].\
        replace(to_replace='Normal', value=np.nan)

    # replace Indeterminate with nan
    clinical_data = clinical_data.replace(to_replace='Indeterminate',
                                          value=np.nan)

    clinical_data, clinical_data_bc_info = filter_observations(clinical_data,
                                                               **filter_kws)
    metadata = pd.merge(metadata, clinical_data,
                        left_index=True, right_index=True, how='outer')

    vars2compare += ['HER2_IHC', 'ER_IHC', 'PR_IHC', 'histological_type']

metadata.to_csv(join(pro_data_dir, 'metadata.csv'))
metadata[vars2compare].to_csv(join(pro_data_dir, 'vars2compare.csv'))

res_writer.write("processing data took {:1.2f} seconds".
                 format(time() - start_time))
# #####################
# # Process variables #
# #####################
# print("processing variables")
# # initial variable processing
# # drop columns with nans
# # keep top 1000 most variable vairiables
# # filter 0 variance variables
# for k in data.keys():
#     print('\n', k, data[k].shape)

#     # kill variables with nans
#     # TODO: should we impute
#     data[k] = data[k].dropna(axis=1)
#     print('after dropping nans', data[k].shape)

#     var_stds = data[k].std(axis=0).sort_values(ascending=False)

#     # subset to top 100
#     var_stds = var_stds[0:2000]

#     # remove zero variance varaiables
#     non_zero_variance_mask = var_stds > 0
#     var_stds = var_stds[non_zero_variance_mask]

#     vars2keep = var_stds.index.values

#     data[k] = data[k][vars2keep]
#     print('after variance filter', data[k].shape)


###########
# Run PCA #
###########
# py_fpath = join(Paths().sim_scripts_dir, 'data_analysis', 'tcga_pca.py')
# py_command = 'python {} --cancer_type {} --feat_list {}'.\
#              format(py_fpath, args.cancer_type, args.feat_list)
# os.system(py_command)
