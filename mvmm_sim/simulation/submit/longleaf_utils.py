from itertools import product
import os

import json
from mvmm_sim.simulation.Paths import which_computer


def run_command(py_script, py_params, longleaf_params, args,
                break_first=False):
    """
    Runs a python script.

    Parameters
    ----------
    py_script: str
        Path to python script to run.

    py_params: dict
        Python arguments to run. Dict key/values are converted to
            --key value (if value is not a bool).
            --key if value is bool and True. Otherwise key is ignored
            if value == False.

    longleaf_params: dict
        Longleaf submission parameters

    arg:
        args from bulk_submit_parser

    break_first: bool
        Will throw error after first run when in print mode.

    """

    py_args_str = arg_str(py_params)

    if args.mini:
        py_args_str += '--mini '

    if which_computer() == 'longleaf' or args.no_submit:
        longleaf_args_str = ''
    else:
        longleaf_args_str = arg_str(longleaf_params)

    command = 'python {} {} {}'.format(py_script, py_args_str, longleaf_args_str)
    print(command)
    if not args.print_only:
        os.system(command)
        if break_first:
            1 / 0


def submit_expers(py_script, py_params, longleaf_params, args,
                  data_dir=None, base_name=None):
    """
    TODO: fill out documentation

    Parameters
    ----------
    py_script: str
        Absolute path to python script to call

    py_params: dict

    longleaf_params: dict

    args: arg

    data_dir:

    base_name:

    """

    py_args_list = to_list_of_params(py_params)
    py_args_list = exper_naming(py_args_list, base_name=base_name)

    for i, py_args in enumerate(py_args_list):
        run_command(py_script=py_script, py_params=py_args,
                    longleaf_params=longleaf_params, args=args)

    save_params(py_args_list=py_args_list, data_dir=data_dir, base_name=base_name)


def exper_naming(py_args_list, base_name=None):
    for i, p in enumerate(py_args_list):
        if base_name is not None:
            p['name'] = '{}_{}'.format(base_name, i)
    return py_args_list


def save_params(py_args_list, data_dir, base_name=None):

    if base_name is not None:
        fpath = os.path.join(data_dir, '{}.json'.format(base_name))
        with open(fpath, 'w') as f:
            json.dump(py_args_list, f, indent=2)


def bulk_submit_parser(parser):
    parser.add_argument('--print_only', action='store_true', default=False,
                        help='To print or run command.')

    parser.add_argument('--mini', action='store_true', default=False,
                        help='Run small example for debugging.')

    parser.add_argument('--no_submit', action='store_true', default=False,
                        help='Will not submit on longleaf.')

    return parser


def to_list_of_params(dl):
    """
    Converts a dict of parameter lists to a list of parameter combinations

    Parameters
    ----------
    dict:
        Each key of the dict is a parameter and each entry is a list
        of parameters to use. Non-lists will be converted to lists
        of lengh 1.

    dl = {'a': [0, 1], 'b': [8, 9], 'c': 11}

    to_list_of_params(dl)

    [{'a': 0, 'b': 8, 'c': 11},
     {'a': 0, 'b': 9, 'c': 11},
     {'a': 1, 'b': 8, 'c': 11},
     {'a': 1, 'b': 9, 'c': 11}]
    """

    list_params = []
    for k in dl.keys():
        if type(dl[k]) != list:
            dl[k] = [dl[k]]

    all_pairs = []
    for k in dl:
        k_v_pairs = []
        for v in dl[k]:
            k_v_pairs.append((k, v))
        all_pairs.append(k_v_pairs)

    param_combos = list(product(*all_pairs))

    list_params = []
    for i, tup in enumerate(param_combos):
        setting = {}
        for k, v in tup:
            setting[k] = v
        list_params.append(setting)

    return list_params


def arg_str(params):
    args = ''
    for k, v in params.items():
        if type(v) == bool:
            if v:
                args += '--{} '.format(k)
        else:
            if v is not None:
                args += '--{} {} '.format(k, v)

    return args
