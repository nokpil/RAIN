"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""
from synthetic_sim import *
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num-train', type=int, default=10000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=1500,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=1500,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=10,
                    help='How often to sample the trajectory.')
parser.add_argument('-n', '--n-balls', type=int, default=10,
                    help='Number of balls in the simulation.')
parser.add_argument('-dt', '--delta-T', type=float, default=.01,
                    help='Timestep interval of the simulation.')
parser.add_argument('-sp', '--spring-prob', type=float, default=.1,
                    help='Timestep interval of the simulation.')
parser.add_argument('-ns', '--noise-strength', type=float, default=0.,
                    help='Timestep interval of the simulation.')
parser.add_argument('-it', '--interaction-type', type=str, default='N',
                    help='interaction type / N (normal), S (signed), D (directed), T (time-delayed), SD, TS, TD, TSD')
parser.add_argument('-sm', '--sample-mode', type=str, default='uniform',
                    help='interaction weight type / uniform, normal, duplex')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('-sf', '--savefolder', type=str, default='kuramoto_test',
                    help='name of folder to save everything in')


args = parser.parse_args()
#args.savefolder = str(args.sim_type) + '_' + str(args.n_balls) + '_t' + str(int(args.delta_T*1000)) + '_' + args.sample_mode

args.savefolder = "D:\Onedrive\연구\ML\Myproject\RAIN\RAIN\data\\" + args.savefolder
os.makedirs(args.savefolder, exist_ok=True)
par_file = open(os.path.join(args.savefolder, 'sim_args.txt'), 'w')
print(args, file=par_file)
par_file.flush()
par_file.close()

sim = KuramotoSim(n_osc=args.n_balls, interaction_type=args.interaction_type, sample_mode=args.sample_mode, dt=args.delta_T, m_order=1, noise=args.noise_strength)
np.random.seed(args.seed)

def generate_dataset(num_sims, length, sample_freq):
    dphi_all = list()
    sinphi_all = list()
    phi_all = list()
    freq_all = list()
    edges_all = list()
    order_all = list()
    p = args.spring_prob
    for i in range(num_sims):
        t = time.time()
        dphi, sinphi, phi, freq, edges, order = sim.sample_trajectory(T=length, sample_freq=sample_freq, edge_prob=[1 - p, p])
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        dphi_all.append(dphi)
        sinphi_all.append(sinphi)
        phi_all.append(phi)
        freq_all.append(freq)
        edges_all.append(edges)
        order_all.append(order)

    dphi_all = np.stack(dphi_all)
    sinphi_all = np.stack(sinphi_all)
    phi_all = np.stack(phi_all)
    freq_all = np.stack(freq_all)
    edges_all = np.stack(edges_all)
    order_all = np.stack(order_all)

    return dphi_all, sinphi_all, phi_all, freq_all, edges_all, order_all

print("Generating {} training simulations".format(args.num_train))
dphi_train, sinphi_train, phi_train, freq_train, edges_train, order_train = generate_dataset(args.num_train, args.length, args.sample_freq)

np.save(os.path.join(args.savefolder, 'dphi_train.npy'), dphi_train)
np.save(os.path.join(args.savefolder, 'sinphi_train.npy'), sinphi_train)
np.save(os.path.join(args.savefolder, 'phi_train.npy'), phi_train)
np.save(os.path.join(args.savefolder, 'freq_train.npy'), freq_train)
np.save(os.path.join(args.savefolder, 'edges_train.npy'), edges_train)
np.save(os.path.join(args.savefolder, 'order_train.npy'), order_train)

print("Generating {} validating simulations".format(args.num_valid))
dphi_valid, sinphi_valid, phi_valid, freq_valid, edges_valid, order_valid = generate_dataset(args.num_valid, args.length, args.sample_freq)

np.save(os.path.join(args.savefolder, 'dphi_valid.npy'), dphi_valid)
np.save(os.path.join(args.savefolder, 'sinphi_valid.npy'), sinphi_valid)
np.save(os.path.join(args.savefolder, 'phi_valid.npy'), phi_valid)
np.save(os.path.join(args.savefolder, 'freq_valid.npy'), freq_valid)
np.save(os.path.join(args.savefolder, 'edges_valid.npy'), edges_valid)
np.save(os.path.join(args.savefolder, 'order_valid.npy'), order_valid)

print("Generating {} test simulations".format(args.num_test))
dphi_test, sinphi_test, phi_test, freq_test, edges_test, order_test = generate_dataset(args.num_test, args.length, args.sample_freq)

np.save(os.path.join(args.savefolder, 'dphi_test.npy'), dphi_test)
np.save(os.path.join(args.savefolder, 'sinphi_test.npy'), sinphi_test)
np.save(os.path.join(args.savefolder, 'phi_test.npy'), phi_test)
np.save(os.path.join(args.savefolder,'freq_test.npy'), freq_test)
np.save(os.path.join(args.savefolder, 'edges_test.npy'), edges_test)
np.save(os.path.join(args.savefolder, 'order_test.npy'), order_test)