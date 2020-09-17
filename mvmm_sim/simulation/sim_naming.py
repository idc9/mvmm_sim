import names
import numpy as np
import os


def load_all_names():
    all_names = []
    for gen in ['male', 'female']:
        fpath = names.full_path('dist.{}.first'.format(gen))
        with open(fpath) as name_file:
            for line in name_file:
                name, _, cummulative, _ = line.split()
                all_names.append(name.lower())

    return all_names


def get_subfolders(folder):
    return [name for name in os.listdir(folder) if os.path.isdir(name)]


def get_new_name(folder):
    all_names = load_all_names()

    existing_names = get_subfolders(folder)

    possibilities = set(all_names).difference(existing_names)
    assert len(possibilities) > 0
    possibilities = np.array(list(possibilities))
    possibilities = np.sort(possibilities)
    return possibilities[0]
