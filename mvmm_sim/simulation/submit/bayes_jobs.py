import os
from mvmvmm_simmm.simulation.submit.file_transfer import bayes_to_iain


def get_iain_job_data(time_idx, password, who='csv', iain='./'):
    assert who in ['csv', 'txt']
    home_dir = '/home/guests/idc9'

    bayes = os.path.join(home_dir,
                         'status_of_jobs/iain_jobs_{}.{}'.format(time_idx,
                                                                 who))
    bayes_to_iain(bayes=bayes, password=password, iain=iain)


def write_jobs2del(job_idxs, fpath='to_delete.txt'):

    f = open(fpath, "w")
    for job_idx in job_idxs:
        # write line to output file
        f.write(str(job_idx))
        f.write("\n")
    f.close()


def read_jobs2del(fpath='/home/guests/idc9/status_of_jobs/to_delete.txt'):
    f = open(fpath, "r")

    job_idxs = []
    for line in f.readlines():
        job_idxs.append(line.strip())
    return job_idxs
