import argparse
import os
from joblib import dump

from mvmm_sim.simulation.Paths import Paths
from mvmm_sim.simulation.sim_aggregation import aggregate_sims

parser = argparse.\
    ArgumentParser(description='Make simulation results visualization.')
parser.add_argument('--sim_name', type=str,
                    help='Which simulation to run.')
parser.add_argument('--delete_files', action='store_true', default=False,
                    help='Delete files for each MC/n simulation.')
args = parser.parse_args()

delete_files = args.delete_files
sim_name = args.sim_name


#################################
# agregate data and save results#
#################################

save_dir = os.path.join(Paths().out_data_dir, sim_name)
results, extra_data_mc_0 = aggregate_sims(save_dir=save_dir,
                                          delete_files=delete_files)
dump(results, os.path.join(save_dir, 'simulation_results'))
dump(extra_data_mc_0, os.path.join(save_dir, 'extra_data_mc_0'))

############################
# run visualization script #
############################
py_fpath = os.path.join(Paths().sim_scripts_dir, 'viz_results.py')
py_command = 'python {} --sim_name {}'.format(py_fpath, sim_name)
os.system(py_command)
