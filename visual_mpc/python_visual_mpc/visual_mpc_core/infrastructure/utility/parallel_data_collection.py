from multiprocessing import Pool
import argparse
import imp

import os
import os.path
import sys
sys.path.append("..")   # enter "Infrastructure"
#from python_visual_mpc.visual_mpc_core.infrastructure.lsdc_main_mod import LSDCMain
from lsdc_main_mod import LSDCMain

import copy
import random
import numpy as np
import shutil
import pdb

def worker(conf):
    print 'started process with PID:', os.getpid()
    print 'making trajectories {0} to {1}'.format(
        conf['start_index'],
        conf['end_index'],
    )

    random.seed(None)
    np.random.seed(None)

    lsdc = LSDCMain(conf)
    lsdc.run()


def bench_worker(conf):
    print 'started process with PID:', os.getpid()

    random.seed(None)
    np.random.seed(None)
    perform_benchmark(conf)

class Modhyper(object):
    def __init__(self, conf):
        self.agent = conf.agent
        self.common = conf.common
        self.config = conf.config
        self.policy = conf.policy

def main():
    parser = argparse.ArgumentParser(description='run parllel data collection')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('--parallel', type=str, help='use multiple threads or not', default=False)

    args = parser.parse_args()
    exp_name = args.experiment
    parallel= args.parallel


    if parallel == 'True':
        n_worker = 10
        parallel = True
        print 'using ', n_worker, ' workers'
    if parallel == 'False':
        parallel = False
        n_worker = 1
    print 'parallel ', bool(parallel)
    n_worker = 1

    from python_visual_mpc import __file__ as basedir

    basedir = os.path.abspath(basedir)
    lsdc_dir = '/'.join(str.split(basedir, '/')[:-2])
    data_coll_dir = lsdc_dir + '/pushing_data/' + exp_name
    hyperparams_file = data_coll_dir + '/hyperparams.py'
    do_benchmark = False

    if os.path.isfile(hyperparams_file):
        hyperparams = imp.load_source('hyperparams', hyperparams_file)
        n_traj = hyperparams.config['end_index']
    else:
        print 'doing benchmark ...'
        do_benchmark = True
        experimentdir = lsdc_dir + '/experiments/cem_exp/benchmarks_goalimage/' + exp_name
        hyperparams_file = experimentdir + '/mod_hyper.py'
        mod_hyperparams = imp.load_source('hyperparams', hyperparams_file)
        n_traj = mod_hyperparams.config['end_index']
        mod_hyper = Modhyper(mod_hyperparams)
        mod_hyper.config['bench_dir'] = experimentdir

    traj_per_worker = int(n_traj / np.float32(n_worker))
    start_idx = [traj_per_worker * i for i in range(n_worker)]
    end_idx =  [traj_per_worker * (i+1)-1 for i in range(n_worker)]

    conflist = []


    for i in range(n_worker):
        if do_benchmark:
            modconf = copy.deepcopy(mod_hyper)
            modconf.config['start_index'] = start_idx[i]
            modconf.config['end_index'] = end_idx[i]
        else:
            modconf = copy.deepcopy(hyperparams.config)
            modconf['start_index'] = start_idx[i]
            modconf['end_index'] = end_idx[i]

        conflist.append(modconf)

    if do_benchmark:
        use_worker = bench_worker
    else: use_worker = worker

    if parallel:
        p = Pool(n_worker)
        p.map(use_worker, conflist)
    else:
        use_worker(conflist[0])

    # move first file from train to test
    conf = hyperparams.common
    file = conf['data_files_dir']+ '/traj_0_to_255.tfrecords'
    dest_file = '/'.join(str.split(conf['data_files_dir'], '/')[:-1]) + '/test/traj_0_to_255.tfrecords'
    shutil.move(file, dest_file)
    print 'Done'

if __name__ == '__main__':
    main()