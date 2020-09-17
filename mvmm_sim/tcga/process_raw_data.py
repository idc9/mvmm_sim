import pandas as pd
from collections import Counter
import numpy as np


def parse_barcode(x, delim='-'):
    # https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/#:~:text=A%20TCGA%20barcode%20is%20composed,the%20highest%20number%20of%20identifiers.

    s = x.split(delim)

    i2name = ['project', 'tss', 'participant', 'sample', 'portion',
              'plate', 'center']
    info = {'barcode': x}
    for i in range(min(7, len(s))):
        info[i2name[i]] = s[i]

    # process sample
    if 'sample' in info.keys():

        if len(info['sample']) == 3:
            info['vial'] = info['sample'][2]
            info['sample'] = info['sample'][0:2]

        else:
            info['vial'] = None

        if int(info['sample']) <= 9:
            info['is_tumor_sample'] = True
        else:
            info['is_tumor_sample'] = False

        if int(info['sample']) == 1:
            info['is_primary_tumor'] = True
        else:
            info['is_primary_tumor'] = False

    else:
        # TODO: if sample number is missing I assume it means
        # this is the primary tumor
        info['is_tumor_sample'] = True
        info['is_primary_tumor'] = True

    if 'portion' in info.keys():
        if len(info['portion']) == 3:
            info['analyte'] = info['portion'][2]
            info['portion'] = info['portion'][0:2]
        else:
            info['portion'] = None

    return info


def get_patient_barcode(barcode):
    b = parse_barcode(barcode)
    return '-'.join([b['project'], b['tss'], b['participant']])


def add_patient_barcode(df):
    df['patient_barcode'] = list(map(get_patient_barcode, df.index.values))
    return df


def get_portion_barcode(barcode):
    b = parse_barcode(barcode)
    return '-'.join([b['project'], b['tss'], b['participant'],
                     b['sample'] + b['vial'], b['portion'] + b['analyte']])


def get_barcode_df(x):

    if isinstance(x, pd.DataFrame):
        barcodes = x.index
    else:
        barcodes = x
    info = pd.DataFrame([parse_barcode(i, delim='-')
                        for i in barcodes]).set_index('barcode')

    info = add_patient_barcode(info)

    return info


def filter_observations(data, patient_barcodes=None,
                        tumor_sample_only=True,
                        primary_tumor_only=True,
                        keep_first_of_participant_multiples=True,
                        ensure_participant_idx=True,
                        aliquots2remove=None,
                        verbose=True):

    if verbose:
        print("Initial shape", data.shape)

    bc_info = get_barcode_df(data)

    if patient_barcodes is not None:
        bc_info = bc_info.query("patient_barcode in @patient_barcodes")
        data = data.loc[bc_info.index]

        if verbose:
            print("After patient_barcodes", data.shape)

    # kill cols with all nans
    all_nans = bc_info.isna().mean(axis=0) == 1
    bc_info = bc_info.drop(columns=all_nans[all_nans].index.values)

    ##########################
    # Only use tumor samples #
    ##########################
    if tumor_sample_only and 'sample' in bc_info.columns:
        bc_info = bc_info.query('is_tumor_sample')
        data = data.loc[bc_info.index]

        if verbose:
            print("After tumor_sample_only", data.shape)

    ###########################
    # Only use primary tumors #
    ###########################
    if primary_tumor_only and 'sample' in bc_info.columns:
        bc_info = bc_info.query('is_primary_tumor')
        data = data.loc[bc_info.index]

        if verbose:
            print("After primary_tumor_only", data.shape)

    ###################
    # Quality control #
    ###################
    # remove bad portions from quality control file
    if aliquots2remove is not None:
        portions2remove = set([get_portion_barcode(a)
                               for a in aliquots2remove])

        current_portions = [get_portion_barcode(b) for b in bc_info.index]
        to_remove_mask = np.array([(p in portions2remove)
                                   for p in current_portions])

        bc_info = bc_info.loc[~to_remove_mask]
        data = data.loc[bc_info.index]

        if verbose:
            print("removing {} bad portions".format(sum(to_remove_mask)))
            print("After removing bad portions", data.shape)

    # Only keep one participant row for rows that have multiple participants
    if keep_first_of_participant_multiples:
        part_cnts = pd.Series(Counter(bc_info['participant']))
        part_cnts = part_cnts[part_cnts >= 2]

        if len(part_cnts) > 0:
            idxs2drop = []
            for participant in part_cnts.index:
                part_idxs = bc_info.query("participant == '{}'".
                                          format(participant)).index.values
                idxs2drop.extend(part_idxs[1:])
            bc_info = bc_info.drop(index=idxs2drop)
            data = data.loc[bc_info.index]

        if verbose:
            print("After keep_first_of_participant_multiples", data.shape)

        if set_participant_idx:
            data = set_participant_idx(data, bc_info)

    return data, bc_info


def set_participant_idx(df, bc_info):
    assert all(df.index.values == bc_info.index.values)
    assert bc_info.shape[0] == len(set(bc_info['participant']))

    df['participant'] = bc_info['participant']
    df = df.set_index('participant', drop=True)
    return df
