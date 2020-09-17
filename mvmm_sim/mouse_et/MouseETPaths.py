from os.path import join

from mvmm_sim.simulation.Paths import Paths as sim_Paths
# from mvmm.simulation.Paths import which_computer


class MouseETPaths(object):
    def __init__(self):
        self.results_dir = join(sim_Paths().out_dir, 'mouse_et')
        self.raw_data_dir = join(self.results_dir, 'raw_data')
        self.pro_data_dir = join(self.results_dir, 'pro_data')
