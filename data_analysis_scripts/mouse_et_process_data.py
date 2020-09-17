import pandas as pd
import os
from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt

from ya_pca.PCA import PCA
from ya_pca.viz import scree_plot

from mvmm_sim.mouse_et.load_aligned_data import load_aligned_data
from mvmm_sim.mouse_et.MouseETPaths import MouseETPaths
from mvmm_sim.simulation.sim_viz import save_fig

pro_data_dir = MouseETPaths().pro_data_dir
os.makedirs(pro_data_dir, exist_ok=True)

transc, ephys_pca, metadata = load_aligned_data()
transc = np.log2(1 + transc)

ephys_pca.to_csv(join(pro_data_dir, 'ephys_pca_feats.csv'))
metadata.to_csv(join(pro_data_dir, 'metadata.csv'))

cols = ['corresponding_AIT2.3.1_alias']
# 'hemisphere', 'structure', 'dendrite_type', 'age', 'biological_sex']
vars2compare = metadata[cols]

vars2compare = vars2compare.rename(columns={'corresponding_AIT2.3.1_alias':
                                            "transcr_subtype"})

# these clusters shouldn't be here
vars2compare['transcr_subtype'] = vars2compare['transcr_subtype'].\
    replace(to_replace='L2/3 IT VISp Agmat', value=np.nan)
vars2compare['transcr_subtype'] = vars2compare['transcr_subtype'].\
    replace(to_replace='Meis2 Adamts19', value=np.nan)

# pull out super clusters
super_types = []
for subtype in vars2compare['transcr_subtype'].values:

    if type(subtype) != str:
        super_type = np.nan
    else:
        super_type = subtype.split(' ')[0]
    # if super_type in ['Meis2', 'L2/3']:  # these shouldn't be here
    #     super_type = np.nan
    super_types.append(super_type)
vars2compare['transcr_super_type'] = super_types

vars2compare.to_csv(join(pro_data_dir, 'vars2compare.csv'))

# select markers only
select_markers = pd.read_csv(join(MouseETPaths().raw_data_dir,
                             'select_markers.csv'), index_col=0)
select_markers = select_markers['Gene'].values
# Drop this geen bc it is not in transcriptomic file
select_markers = select_markers[select_markers != 'F630028O10Rik']
transc_sel_markers = transc[select_markers]
transc_sel_markers.\
    to_csv(join(pro_data_dir, 'transcriptomic_select_markers.csv'))

# PCA
pca = PCA(n_components='rmt_threshold',
          rank_sel_kws={'thresh_method': 'dg'})

pca.fit(transc_sel_markers.values)


# save scores
UD = pca.unnorm_scores_
UD = pd.DataFrame(UD, index=transc_sel_markers.index,
                  columns=['pc_{}'.format(k + 1)
                           for k in range(UD.shape[1])])
UD.index.name = transc_sel_markers.index.name
UD.name = 'transcriptomic_select_markers'
UD.to_csv(join(pro_data_dir, '{}_pca_feats.csv'.
          format('transcriptomic_select_markers')))

# save scree plot
plt.figure(figsize=(8, 8))
scree_plot(pca.all_svals_, color='black')
plt.axvline(pca.n_components_,
            label='{}'.format(pca.n_components_),
            color='red')
plt.legend()
save_fig(join(MouseETPaths().results_dir, '{}_rank_selection.png'.
              format('transcriptomic_select_markers')))
