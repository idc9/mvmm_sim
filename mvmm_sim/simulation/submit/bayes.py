import os
import sys
import __main__
import numpy as np
from copy import deepcopy

from warnings import warn
from mvmm_sim.simulation.Paths import Paths
from mvmm_sim.simulation.Paths import which_computer


def bayes_parser(parser):

    parser.add_argument('--print', action='store_true', dest='print_only',
                        help='Only print the command.')

    if which_computer() == 'bayes':
        return add_bayes_args(parser)
    else:
        return parser


def bayes_submit(args):
    """
    Submits job on bayes then exits.

    Parameters
    ----------
    args: output of argparse which includes add_bayes_args

    outname (str): name of the out file.

    """

    if which_computer() == 'bayes' and args.submit \
            and not args.override_submit:

        py_fpath = os.path.join(os.getcwd(), __main__.__file__)
        # new_args = ' '.join(sys.argv[1:]) + ' --override_submit'
        # new_args = ' '.join(argv) + ' --override_submit'

        # reformat node argument
        argv = sys.argv[1:]
        if '--node' in argv:
            idx = np.where(np.array(argv) == '--node')[0].item()
            argv[idx + 1] = "'{}'".format(argv[idx + 1])
        new_args = " ".join(argv) + " --override_submit"

        py_command = '{} {}'.format(py_fpath, new_args)

        bayes_command = get_bayes_command(py_command, args)

        print('submission command:', bayes_command)

        if not args.print_only:
            os.system(bayes_command)

        sys.exit(0)


def get_bayes_command(py_command, args):
    """
    Constructs longleaf submission command.

    Parameters
    ----------
    py_command (str): the python command to be run.

    """

    # bayes_args = {}
    # bayes_args['py_command'] = py_command
    # if time is not None:
    #     bayes_args['time'] = time  # args.time
    # bayes_args['mem'] = mem  # args.mem
    # bayes_args['queue'] = queue  # args.queue
    # bayes_args['out_fname'] = os.path.join(Paths().cluster_out_dir,
    #                                           '{}%J.out'.format(outname))

    command = 'qsub -q {} -V'.format(args.queue)

    if args.time is not None:
        command += ' -l h_rt={}'.format(args.time)

    if args.mem is not None:
        command += ' -l h_vmem={}'.format(args.mem)

    if args.node is not None:
        command += ' -l h="{}"'.format(args.node)

    if args.n_slots is not None:

        n_slots = args.n_slots

        if n_slots == 'all':
            n_slots = 12

        if int(n_slots) > 12:
            warn('n_slots={} > 12 which is the maximum number of '
                 'available slots (Im pretty sure)'.format(n_slots))

        command += ' -pe local {}'.format(n_slots)

    if args.job_name is not None:
        command += ' -N {}'.format(args.job_name)

    # output files
    out_fpath = os.path.join(Paths().cluster_out_dir,
                             '\$JOB_NAME__\$JOB_ID.out')
    err_fpath = os.path.join(Paths().cluster_out_dir,
                             '\$JOB_NAME__\$JOB_ID.err')

    command += ' -o {} -e {}'.format(out_fpath, err_fpath)

    command += ' /home/guests/idc9/bayes_scripts/run_python.sh' \
               ' {}'.format(py_command)
    # command += ' ' + py_command

    # command = 'qsub -q {queue} -V -l h_rt={time} -l h_vmem={mem} '\
    #     '/home/guests/idc9/bayes_scripts/run_python.sh
    # {py_command}'.format(**bayes_args)

    # command = 'qsub -V /home/guests/idc9/bayes_scripts/run_python.sh
    # {py_command}'.format(**bayes_args)
    return command


def add_bayes_args(parser):
    """
    Adds longleaf submission arguments to an argparser
    """

    parser.add_argument('--mem', default='8G',
                        help='Memory to allocate for each job.')

    parser.add_argument('--queue', default='w-normal.q',
                        choices=['w-bigmem.q', 'w-normal.q', 'normal.q'],
                        help='Which queue to submit to.')

    parser.add_argument('--time', default=None,
                        help='Time to allocate for the job.')

    parser.add_argument('--node', default=None,
                        help='Which node to use e.g. b34 or b34|b35.')

    parser.add_argument('--n_slots', default=None,
                        help='Number of job slots to request.')

    parser.add_argument('--job_name', default=None,
                        help='Name of the job.')

    parser.add_argument('--submit', action='store_true',
                        help='Whether to submit the script or to just run it.')

    parser.add_argument('--override_submit', action='store_true',
                        help='Do not submit, even if --submit is called.')

    return parser
