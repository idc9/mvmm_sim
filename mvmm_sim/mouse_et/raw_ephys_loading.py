import os
from glob import glob
import pandas as pd


def load_spca_comp_mapping(fpath):
    df = []
    with open(fpath, "r") as fd:
        for line in fd:
            line = line.strip()
            line = line.split('   ')
            df.append(line)
    df = pd.DataFrame(df)
    df.columns = ['spca_col', 'dataset']
    df = df.set_index('spca_col')
    return df


def load_raw_ephys(data_dir, concat=True):
    csv_files = glob(data_dir + '/*.csv')

    all_dfs = {}
    for fpath in csv_files:
        df = pd.read_csv(fpath, index_col=0)
        df.index.name = 'specimen_id'

        name = os.path.basename(fpath).split('.')[0]
        col_names = [name + '__' + str(j + 1) for j in range(df.shape[1])]
        df.columns = col_names

        all_dfs[name] = df

    if concat:
        cat_df = [all_dfs[k] for k in all_dfs.keys()]
        cat_df = pd.concat(cat_df, axis=1)
        return cat_df
    else:
        return all_dfs
