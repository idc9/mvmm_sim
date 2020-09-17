import os
from os.path import join

# To use the code you should change hard_coded_out_dir and hard_coded_code_dir
# to fit your system


def which_computer():
    """
    Detect if we are working on Iain's laptop or on the cluster
    """
    cwd = os.getcwd()
    if 'iaincarmichael' in cwd:
        return 'iain_laptop'
    elif 'idc9' in cwd:
        return 'bayes'
    else:
        return None
        # raise ValueError('Not sure which comptuer we are on!')


if which_computer() == 'iain_laptop':
    # where all the simulation data is saved
    hard_coded_out_dir = '/Users/iaincarmichael/Dropbox/Research/mvmm/public_release/'

    # directory containing the simulation scripts
    hard_coded_code_dir = '/Users/iaincarmichael/Dropbox/Research/local_packages/python/mvmm_sim'

elif which_computer() == 'bayes':
    hard_coded_out_dir = '/home/guests/idc9/projects/mvmm/public_release'

    hard_coded_code_dir = '/home/guests/idc9/local_packages/mvmm_sim'


class Paths(object):
    def __init__(self, out_dir=hard_coded_out_dir,
                 code_dir=hard_coded_code_dir):

        self.out_dir = out_dir
        self.code_dir = code_dir

        # self.out_dir = '/Users/iaincarmichael/Dropbox/Research/mvmm/public_release/'

        # self.code_dir = '/Users/iaincarmichael/Dropbox/Research/local_packages/python/mvmm_sim'

        if which_computer() == 'bayes':
            self.home_dir = '/home/guests/idc9'
            # self.out_dir = join(self.home_dir, 'projects/mvmm/public_release/')
            # self.code_dir = join(self.home_dir, 'local_packages/mvmm_sim')

            # where to save the cluster printout
            self.cluster_out_dir = join(self.home_dir, 'cluster_out')

        self.results_dir = join(self.out_dir, 'results')
        self.out_data_dir = join(self.out_dir, 'out_data')

        self.sim_scripts_dir = join(self.code_dir, 'simulation_scripts')

    def make_dirs(self):
        to_make = [self.out_data_dir, self.results_dir]

        for folder in to_make:
            os.makedirs(to_make, exist_ok=True)
