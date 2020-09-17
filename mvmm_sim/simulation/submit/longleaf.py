import os
import sys
import __main__

from mvmm_sim.simulation.Paths import Paths, which_computer


def is_longleaf():
    """
    Returns True if the current computer is Longleaf.
    """

    if which_computer() == 'longleaf':
        return True
    else:
        return False


def longleaf_parser(parser):

    parser.add_argument('--print', action='store_true', dest='print_only',
                        help='Only print the command.')
    if is_longleaf():
        return add_longleaf_args(parser)
    else:
        return parser


def longleaf_submit(args, outname=None):
    """
    Submits job on longleaf then exits.

    Parameters
    ----------
    args: output of argparse which includes add_longleaf_args

    outname (str): name of the out file.

    """

    if outname is None:
        outname = __main__.__file__.split('.py')[0]

    if is_longleaf() and args.submit and not args.override_submit:
        py_fpath = os.path.join(os.getcwd(), __main__.__file__)
        new_args = ' '.join(sys.argv[1:]) + ' --override_submit'

        py_command = 'python {} {}'.format(py_fpath, new_args)
        longleaf_command = get_longleaf_command(args, py_command, outname)

        print(longleaf_command)

        if not args.print_only:
            os.system(longleaf_command)

        sys.exit(0)


def get_longleaf_command(args, py_command, outname=''):
    """
    Constructs longleaf submission command.

    Parameters
    ----------
    args: output of argparse which includes add_longleaf_args

    py_command (str): the python command to be run.

    outname (str): name of the out file.
    """

    longleaf_args = {}
    longleaf_args['py_command'] = py_command
    longleaf_args['time'] = args.time
    longleaf_args['mem'] = args.mem
    longleaf_args['ntasks'] = args.ntasks
    longleaf_args['nodes'] = args.nodes

    longleaf_args['out_fname'] = os.path.join(Paths().cluster_out_dir,
                                              '{}%J.out'.format(outname))

    partition = args.partition

    command = 'sbatch '

    if args.exclusive:
        command += '--exclusive '

    if partition == 'general':

        command += '-p general --out {out_fname}  --mem {mem} -t {time}'\
                   ' --ntasks {ntasks} --nodes {nodes}'\
                   ' --wrap="{py_command}"'.format(**longleaf_args)

    elif partition == 'bigmem':

        command += '-p bigmem --qos bigmem_access --out {out_fname}'\
                   ' --mem {mem} -t {time} --ntasks {ntasks} --nodes {nodes}'\
                   ' --wrap="{py_command}"'.format(**longleaf_args)

    elif partition == 'volta':

        command += '-p volta-gpu -q gpu_access --gres=gpu:1 --out {out_fname}'\
                   '  --mem {mem} -t {time} --ntasks {ntasks} --nodes {nodes}'\
                   ' --wrap="{py_command}"'.format(**longleaf_args)

    elif partition == 'gpu':

        command += '-p gpu -q gpu_access --gres=gpu:1 --out {out_fname}'\
                   ' --mem {mem} -t {time} --ntasks {ntasks} --nodes {nodes}'\
                   ' --wrap="{py_command}"'.format(**longleaf_args)

    else:
        raise ValueError('{} is invalid partition'.format(longleaf_args['partition']))

    return command


def add_longleaf_args(parser):
    """
    Adds longleaf submission arguments to an argparser
    """
    parser.add_argument('--time', default='12:00:00',
                        help='Time to allocate for the job.')

    parser.add_argument('--mem', default=20000,
                        help='Memory to allocate for each job.')

    parser.add_argument('--ntasks', default=8,
                        help='Number of tasks.')

    parser.add_argument('--nodes', default=1,
                        help='Number of nodes.')

    parser.add_argument('--exclusive', action='store_true', default=False,
                        help='Gets whole node to myself.')

    parser.add_argument('--partition', default='general',
                        choices=['gpu', 'general', 'volta', 'bigmem'],
                        help='Which partition to run on.')

    parser.add_argument('--submit', action='store_true',
                        help='Whether to submit the script or to just run it.')

    parser.add_argument('--override_submit', action='store_true',
                        help='Do not submit, even if --submit is called.')

    return parser
