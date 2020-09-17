import os

from mvmm_sim.simulation.Paths import Paths


def iain_to_bayes(iain, bayes, password, print_only=False):

    command = "sshpass -p '{}' rsync -aP --delete {} " \
              "idc9@bayes.biostat.washington.edu:{}".format(password,
                                                            iain, bayes)

    if print_only:
        print(command)
    else:
        os.system(command)


def bayes_to_iain(bayes, password, iain='./', print_only=False):
    command = "sshpass -p '{}' rsync -aP idc9@bayes.biostat.washington.edu:"\
              "{} {}".format(password, bayes, iain)

    if print_only:
        print(command)
    else:
        os.system(command)


def get_sim_results(name, password, print_only=False):
    bayes = os.path.join(Paths('bayes').results_dir, name)
    iain = os.path.join(Paths('iain_laptop').results_dir)
    bayes_to_iain(bayes=bayes, iain=iain, password=password,
                  print_only=print_only)


def get_sim_outdata(name, password, print_only=False):
    bayes = os.path.join(Paths('bayes').out_data_dir, name)
    iain = os.path.join(Paths('iain_laptop').out_data_dir)
    bayes_to_iain(bayes=bayes, iain=iain, password=password,
                  print_only=print_only)


def get_mouse_em(name, password, print_only=False):
    bayes = os.path.join(Paths('bayes').out_dir, 'mouse_em', name)
    iain = os.path.join(Paths('iain_laptop').out_dir, 'mouse_em')
    bayes_to_iain(bayes=bayes, iain=iain, password=password,
                  print_only=print_only)
