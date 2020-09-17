from os.path import join
import pandas as pd

from mvmm_sim.mouse_et.MouseETPaths import MouseETPaths


def load_aligned_data():
    # load metadata
    fpath = join(MouseETPaths().raw_data_dir,
                 '20200625_patchseq_metadata_mouse.csv')
    metadata = pd.read_csv(fpath, index_col=0)
    metadata = metadata.set_index('cell_specimen_id')
    metadata.index.name = 'specimen_id'

    # load ephys PCA
    fpath = join(MouseETPaths().raw_data_dir,
                 'sparse_pca_components_mMET_revision_Apr2020.csv')
    ephys_pca = pd.read_csv(fpath, index_col=0)
    ephys_pca.index.name = 'specimen_id'

    # load transcriptomics
    fpath = join(MouseETPaths().raw_data_dir,
                 '20200513_Mouse_PatchSeq_Release_cpm.v2.csv')
    transc = pd.read_csv(fpath, index_col=0).T
    transc.index.name = 'transcriptomics_sample_id'

    # align metadata
    metadata = metadata.loc[ephys_pca.index]

    # align transcriptomics
    t2s = metadata['transcriptomics_sample_id'].\
        reset_index().set_index('transcriptomics_sample_id')
    transc = transc.loc[t2s.index]
    transc['specimen_id'] = t2s
    transc = transc.set_index('specimen_id', drop=True)
    transc = transc.loc[ephys_pca.index]

    # eleminate genes with no variance
    genes_with_var = (transc.std(axis=0) > 1e-6)
    genes_with_var = genes_with_var.index[genes_with_var]
    transc = transc[genes_with_var]

    return transc, ephys_pca, metadata
