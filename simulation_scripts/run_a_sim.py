from joblib import load
import os

import argparse
import warnings

from mvmm_sim.simulation.submit.bayes import bayes_submit, bayes_parser
from mvmm_sim.simulation.run_sim import run_sim_from_configs

parser = argparse.ArgumentParser(description='Run a single simulation.')
parser.add_argument('--config_fpath', type=str, required=True,
                    help='Path to configuration file.')
parser = bayes_parser(parser)
args = parser.parse_args()
bayes_submit(args)

config_fpath = args.config_fpath

# load config
config = load(config_fpath)
os.remove(config_fpath)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# run simulation
run_sim_from_configs(**config)
