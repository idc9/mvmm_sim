from os.path import join

from mvmm_sim.simulation.Paths import Paths as sim_Paths
from mvmm_sim.simulation.Paths import which_computer


class TCGAPaths(object):
    def __init__(self):

        computer = which_computer()

        # TODO: automate this
        if computer == 'iain_laptop':
            self.top_dir = '/Volumes/Samsung_T5/tcga/'
        else:
            self.top_dir = join(sim_Paths().out_dir, 'tcga')

        self.raw_data_dir = join(self.top_dir, 'raw_data')
        self.pro_data_dir = join(self.top_dir, 'pro_data')
        self.intermediate_data_dir = join(self.top_dir, 'intermediate_data')
