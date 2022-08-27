# coding=utf-8

import math
import copy
import distutils
import os.path as path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from data.motion_capture.cmu_motion_data import CmuMotionData


# Data type conversion


def DCN(x):
    return x.data.cpu().numpy()
def CN(x):
    return x.cpu().numpy()
def TTC(x):
    return torch.Tensor(x).cuda()

# Easy plots

def imshow_now(x):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(x, cmap = cm.RdBu, aspect = 'auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_now(x):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(1,1,1)
    for X in x:
        ax.plot(X[0], X[1])


def scatter_now(x):
    fig = plt.figure(figsize = (4,4), dpi = 150)
    ax = fig.add_subplot(1,1,1)
    for X in x:
        ax.scatter(X[0], X[1], s = 2)


# Useful functions


def sign():
    return 1 if np.random.random() < 0.5 else -1

def str2bool(v):
    return bool(distutils.util.strtobool(v))

def false_shuffle(length, threshold):
    target = [0 for _ in range(length)]
    fixed = set([])
    for i in range(length):
        possible = set(range(max(0, i-threshold), min(length, i+threshold)))
        true_possible = possible - fixed 
        if true_possible:
            s = np.random.choice(list(true_possible))
        #print(i, true_possible, set.intersection(possible, fixed), s)
        fixed.add(s)
        target[i] = s
    return target


def pdist(z):
    z_norm = (z ** 2).sum(2).view(-1, z.shape[1], 1)
    w_t = torch.transpose(z, 1, 2)
    w_norm = z_norm.view(-1, 1, z.shape[1])
    dist = z_norm + w_norm - 2.0 * torch.bmm(z, w_t)
    dist = torch.clamp(dist, 0., np.inf)
    # return torch.pow(dist, 0.5)
    return dist


def angle_between(v1, v2):
    return np.degrees(np.math.atan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))


def angle_between_vec(x, y):
    x1, y1, vx1, vy1 = x
    x2, y2, vx2, vy2 = y
    v1 = np.array([vx1, vy1])
    v2 = np.array([x2 - x1, y2 - y1])
    if np.sum(v2) == 0:
        v2 = v1
    return angle_between(v1, v2)


def logsumexp(x, dim=None):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    z = torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)) + x_max
    return z.sum(dim=dim)


def norm_sigmoid(a, b, x):
    s= 1/(1+np.exp(b*(x-a)))
    return 1*(s-np.min(s))/(np.max(s)-np.min(s)) # normalize function to 0-1


def idx2idx(idx, target):
    f = 0
    cnvt = []
    for i in target:
        f = np.argwhere(idx[f:]==i)[0][0]+f
        cnvt.append(f)
    return cnvt


def interpolate_polyline(polyline, num_points):
    duplicates = []
    for i in range(1, len(polyline)):
        if np.allclose(polyline[i], polyline[i - 1]):
            duplicates.append(i)
    if duplicates:
        polyline = np.delete(polyline, duplicates, axis=0)
    tck, u = interp.splprep(polyline.T, s=0)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))


def summary(ob):
    print('x    | max :' + str(np.max(ob['x'])) + ', min : ' + str(np.min(ob['x'])))
    print('y    | max :' + str(np.max(ob['y'])) + ', min : ' + str(np.min(ob['y'])))
    print('z    | max :' + str(np.max(ob['z'])) + ', min : ' + str(np.min(ob['z'])))
    print('vx   | max :' + str(np.max(ob['vx'])) + ', min : ' + str(np.min(ob['vx'])))
    print('vy   | max :' + str(np.max(ob['vy'])) + ', min : ' + str(np.min(ob['vy'])))
    print('vz   | max :' + str(np.max(ob['vz'])) + ', min : ' + str(np.min(ob['vz'])))


def autocorr(id_list):
    from pandas.plotting import autocorrelation_plot
    from pandas.plotting import lag_plot

    fig_list = []

    for i in range(len(id_list)):
        d = autocorrelation_plot(id_list[i][['vx', 'vy', 'vz']])
        x = d.axes.lines[5]
        data.append(x.get_xydata())

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    for i in range(len(id_list)):
        ax.plot(data[i][:, 0], np.abs(data[i][:, 1]), c='fuchsia', alpha=0.01, lw=2)

    ax.set_xlim(0, 1000)

    fig_list.append(fig)

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    for i in range(len(id_list)):
        for j in range(1, len(data[i][:, 1])):
            if data[i][:, 1][j] * data[i][:, 1][j - 1] < 0:
                break
        ax.plot(data[i][:j + 1, 0], np.abs(data[i][:j + 1, 1]), c='fuchsia', alpha=0.01, lw=2)

    ax.set_xlim(0, 500)

    fig_list.append(fig)

    return fig_list


def trajectory(id_list, index_list, time, interval):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')

    for index in index_list:
        if time > 0:
            data = id_list[index][:time:interval]
        else:
            data = id_list[index][::interval]
        ax.plot(data['x'].values, data['y'].values, data['z'].values, lw=5, c='red')
        ax.quiver(data['x'].values, data['y'].values, data['z'].values, data['vx'].values, data['vy'].values,
                  data['vz'].values, color='blueviolet', length=0.1, normalize=True)


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


# Pytorch Dataloader

def dataloader_spring(batch_size=16, sim_folder="", data_folder="data", len_enc=50, len_dec=50, data_ratio=1.0, noise_var=0., distributed=True):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder, sim_folder, "loc_train.npy"))
    vel_train = np.load(path.join(data_folder, sim_folder, "vel_train.npy"))
    edges_train = np.load(path.join(data_folder, sim_folder, "edges_train.npy"))

    loc_test = np.load(path.join(data_folder, sim_folder, "loc_test.npy"))
    vel_test = np.load(path.join(data_folder, sim_folder, "vel_test.npy"))
    edges_test = np.load(path.join(data_folder, sim_folder, "edges_test.npy"))

    loc_train += np.random.randn(*loc_train.shape) * noise_var
    vel_train += np.random.randn(*vel_train.shape) * noise_var
    
    loc_test += np.random.randn(*loc_test.shape) * noise_var
    vel_test += np.random.randn(*vel_test.shape) * noise_var

    # [num_samples, num_timesteps, num_dims, num_atoms]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    feat_train = np.transpose(np.concatenate([loc_train, vel_train], axis=2), (0, 1, 3, 2))
    feat_test = np.transpose(np.concatenate([loc_test, vel_test], axis=2), (0, 1, 3, 2))

    train_data_len = int(feat_train.shape[0] * data_ratio)
    feat_train_enc = torch.FloatTensor(feat_train[:, :len_enc])[:train_data_len]
    feat_train_dec = torch.FloatTensor(feat_train[:, len_enc:len_enc + len_dec])[:train_data_len]
    edges_train = torch.FloatTensor(edges_train)[:train_data_len]

    test_data_len = int(feat_test.shape[0] * data_ratio)
    feat_test_enc = torch.FloatTensor(feat_test[:, :len_enc])[:test_data_len]
    feat_test_dec = torch.FloatTensor(feat_test[:, len_enc:len_enc + len_dec])[:test_data_len]
    edges_test = torch.FloatTensor(edges_test)[:test_data_len]
    
    train_data = TensorDataset(feat_train_enc, feat_train_dec, edges_train)
    test_data = TensorDataset(feat_test_enc, feat_test_dec, edges_test)

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=train_sampler,
        )
        test_sampler = torch.utils.data.DistributedSampler(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=test_sampler,
        )

        return (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        
        return (
            train_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        )

def dataloader_kuramoto(batch_size=16, sim_folder="", data_folder="data", len_enc=50, len_dec=50, data_ratio=1.0, noise_var=0., ww=True, distributed=True):
    ### the edges numpy arrays below are [ num_sims, N, N ]
    dphi_train = np.load(path.join(data_folder, sim_folder, "dphi_train.npy"))
    sinphi_train = np.load(path.join(data_folder, sim_folder, "sinphi_train.npy"))
    freq_train = np.load(path.join(data_folder, sim_folder, "freq_train.npy"))
    edges_train = np.load(path.join(data_folder, sim_folder, "edges_train.npy"))
    order_train = np.load(path.join(data_folder, sim_folder, "order_train.npy"))

    dphi_test = np.load(path.join(data_folder, sim_folder, "dphi_test.npy"))
    sinphi_test = np.load(path.join(data_folder, sim_folder, "sinphi_test.npy"))
    freq_test = np.load(path.join(data_folder, sim_folder, "freq_test.npy"))
    edges_test = np.load(path.join(data_folder, sim_folder, "edges_test.npy"))
    order_test = np.load(path.join(data_folder, sim_folder, "order_test.npy"))
    #phi_train = np.load(path.join(data_folder, sim_folder, "phi_train.npy"))
    #phi_test = np.load(path.join(data_folder, sim_folder, "phi_test.npy"))

    dphi_train += np.random.randn(*dphi_train.shape) * noise_var
    sinphi_train += np.random.randn(*sinphi_train.shape) * noise_var

    dphi_test += np.random.randn(*dphi_test.shape) * noise_var
    sinphi_test += np.random.randn(*sinphi_test.shape) * noise_var

    dphi_max = dphi_train.max()
    sinphi_max = sinphi_train.max()
    dphi_min = dphi_train.min()
    sinphi_min = sinphi_train.min()
    #phi_max = phi_train.max()
    #phi_min = phi_train.min()

    ### Normalize to [-1, 1]
    dphi_train = (dphi_train - dphi_min) * 2 / (dphi_max - dphi_min) - 1
    sinphi_train = (sinphi_train - sinphi_min) * 2 / (sinphi_max - sinphi_min) - 1

    dphi_test = (dphi_test - dphi_min) * 2 / (dphi_max - dphi_min) - 1
    sinphi_test = (sinphi_test - sinphi_min) * 2 / (sinphi_max - sinphi_min) - 1

    #phi_train = (phi_train - phi_min) * 2 / (phi_max - phi_min) - 1
    #phi_test = (phi_test - phi_min) * 2 / (phi_max - phi_min) - 1

    ### Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    ### shape : train_len, agent_num, (len_enc+len_dec), 1
    dphi_train = dphi_train.reshape(*dphi_train.shape, 1)
    sinphi_train = sinphi_train.reshape(*sinphi_train.shape, 1)
    freq_train = np.expand_dims(np.expand_dims(freq_train, -1).repeat(dphi_train.shape[-2], -1), -1) 
    
    ### shape : test_len, agent_num, (len_enc+len_dec), 1
    dphi_test = dphi_test.reshape(*dphi_test.shape, 1)
    sinphi_test = sinphi_test.reshape(*sinphi_test.shape, 1)
    freq_test = np.expand_dims(np.expand_dims(freq_test, -1).repeat(dphi_test.shape[-2], -1), -1)
    #phi_train = phi_train.reshape(*phi_train.shape, 1)
    #phi_test = phi_test.reshape(*phi_test.shape, 1)

    if ww:
        feat_train = np.transpose(np.concatenate([dphi_train, sinphi_train, freq_train], axis=-1), (0, 2, 1, 3))
        feat_test = np.transpose(np.concatenate([dphi_test, sinphi_test, freq_test], axis=-1), (0, 2, 1, 3))
    else:
        feat_train = np.transpose(np.concatenate([dphi_train, sinphi_train], axis=-1), (0, 2, 1, 3))
        feat_test = np.transpose(np.concatenate([dphi_test, sinphi_test], axis=-1), (0, 2, 1, 3))
    
    train_data_len = int(feat_train.shape[0] * data_ratio)
    feat_train_enc = torch.FloatTensor(feat_train[:, :len_enc])[:train_data_len]
    feat_train_dec = torch.FloatTensor(feat_train[:, len_enc:len_enc + len_dec])[:train_data_len]
    freq_train = torch.FloatTensor(freq_train)[:train_data_len]
    edges_train = torch.FloatTensor(edges_train)[:train_data_len]
    order_train = torch.FloatTensor(order_train)[:train_data_len]

    test_data_len = int(feat_test.shape[0] * data_ratio)
    feat_test_enc = torch.FloatTensor(feat_test[:, :len_enc])[:test_data_len]
    feat_test_dec = torch.FloatTensor(feat_test[:, len_enc:len_enc + len_dec])[:test_data_len]
    freq_test = torch.FloatTensor(freq_test)[:test_data_len]
    edges_test = torch.FloatTensor(edges_test)[:test_data_len]
    order_test = torch.FloatTensor(order_test)[:test_data_len]

    train_data = TensorDataset(feat_train_enc, feat_train_dec, edges_train.squeeze(), freq_train[:, :, 0].squeeze(), order_train)
    test_data = TensorDataset(feat_test_enc, feat_test_dec, edges_test.squeeze(), freq_test[:, :, 0].squeeze(), order_test)

    print('dataset finished')

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=train_sampler,
        )
        test_sampler = torch.utils.data.DistributedSampler(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=test_sampler,
        )
        return (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
            dphi_max,
            dphi_min,
            sinphi_max,
            sinphi_min,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        
        return (
            train_loader,
            test_loader,
            dphi_max,
            dphi_min,
            sinphi_max,
            sinphi_min,
        )

def dataloader_charge(batch_size=16, sim_folder="", data_folder="data", len_enc=50, len_dec=50, data_ratio=1.0, noise_var=0., distributed=True):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder, sim_folder, "loc_train.npy"))
    vel_train = np.load(path.join(data_folder, sim_folder, "vel_train.npy"))
    edges_train = np.load(path.join(data_folder, sim_folder, "edges_train.npy"))

    loc_test = np.load(path.join(data_folder, sim_folder, "loc_test.npy"))
    vel_test = np.load(path.join(data_folder, sim_folder, "vel_test.npy"))
    edges_test = np.load(path.join(data_folder, sim_folder, "edges_test.npy"))

    loc_train += np.random.randn(*loc_train.shape) * noise_var
    vel_train += np.random.randn(*vel_train.shape) * noise_var
    
    loc_test += np.random.randn(*loc_test.shape) * noise_var
    vel_test += np.random.randn(*vel_test.shape) * noise_var

    # [num_samples, num_timesteps, num_dims, num_atoms]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    feat_train = np.transpose(np.concatenate([loc_train, vel_train], axis=2), (0, 1, 3, 2))
    feat_test = np.transpose(np.concatenate([loc_test, vel_test], axis=2), (0, 1, 3, 2))

    train_data_len = int(feat_train.shape[0] * data_ratio)
    feat_train_enc = torch.FloatTensor(feat_train[:, :len_enc])[:train_data_len]
    feat_train_dec = torch.FloatTensor(feat_train[:, len_enc:len_enc + len_dec])[:train_data_len]
    edges_train = torch.FloatTensor(edges_train)[:train_data_len]

    test_data_len = int(feat_test.shape[0] * data_ratio)
    feat_test_enc = torch.FloatTensor(feat_test[:, :len_enc])[:test_data_len]
    feat_test_dec = torch.FloatTensor(feat_test[:, len_enc:len_enc + len_dec])[:test_data_len]
    edges_test = torch.FloatTensor(edges_test)[:test_data_len]
    
    train_data = TensorDataset(feat_train_enc, feat_train_dec, edges_train)
    test_data = TensorDataset(feat_test_enc, feat_test_dec, edges_test)

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=train_sampler,
        )
        test_sampler = torch.utils.data.DistributedSampler(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=test_sampler,
        )

        return (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        
        return (
            train_loader,
            test_loader,
            loc_max,
            loc_min,
            vel_max,
            vel_min,
        )

def dataloader_motion(batch_size=8, sim_folder="", data_folder="data", len_enc=50, len_dec=50, data_ratio=1.0, noise_var=0., distributed=True):

    params = {}
    params['data_path'] = f"./{data_folder}/motion_capture/output"
    params['error_out_name'] = f"./{data_folder}/motion_capture/output"
    params['expand_train'] = True
    params['train_data_len'] = 100
    params['num_vars'] = 31
    params['input_noise_type'] = 'none'
    params['input_size'] = 6
    params['input_time_steps'] = params['train_data_len']

    train_data = CmuMotionData('cmu', params['data_path'], 'train', params).feat
    test_data = CmuMotionData('cmu', params['data_path'], 'test', params, test_full=True).feat

    train_data += torch.randn(*train_data.shape) * noise_var
    test_data += torch.randn(*test_data.shape) * noise_var
    
    train_data_len = int(train_data.shape[0] * data_ratio)
    feat_train_enc = train_data[:, :len_enc][:train_data_len]
    feat_train_dec = train_data[:, len_enc:len_enc + len_dec][:train_data_len]

    test_data_len = int(test_data.shape[0] * data_ratio)
    feat_test_enc = test_data[:, :len_enc][:test_data_len]
    feat_test_dec = test_data[:, len_enc:len_enc + len_dec][:test_data_len]
    
    
    train_data = TensorDataset(feat_train_enc, feat_train_dec)
    test_data = TensorDataset(feat_test_enc, feat_test_dec)

    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_data)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=train_sampler,
        )
        test_sampler = torch.utils.data.DistributedSampler(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=test_sampler,
        )

        return (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        
        return (
            train_loader,
            test_loader,
        )
   
def ns(loader):
    new_loader = copy.deepcopy(loader)
    data_source = new_loader.sampler.data_source
    new_loader.batch_sampler.sampler = torch.utils.data.sampler.SequentialSampler(data_source)
    return new_loader

class DistributedSampler_LSTM(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = false_shuffle(len(self.dataset), 10)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        #indices = indices[self.rank*self.num_samples : (self.rank+1)*self.num_samples]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        #print(len(indices), self.num_samples)
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def data_cat(train_set):
    train_data_list = []
    train_labels_list = []

    for i in range(len(train_set.train_data)):
        data = train_set.train_data[i]
        labels = train_set.train_labels[i]
        mask = torch.tensor([True if labels[j, 0] == 1 else False for j in range(labels.shape[0])])
        intp_data = data[mask, :]
        labels = labels[mask, :]
        train_data_list.append(DCN(intp_data))
        train_labels_list.append(DCN(labels))

    train_data = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_labels_list)

    return train_data, train_labels


def make_label_mask(labels, boolean):
    if boolean:
        return torch.tensor([True if labels[j, 0] == 1 else False for j in range(labels.shape[0])])
    else:
        return torch.tensor([1. if labels[j, 0] == 1 else -1. for j in range(labels.shape[0])])


def masked_pair(data, label, i, device):
    mask = label_mask(label[i], boolean=True)
    return mask, (data[i, mask, :].to(device), label[i, mask, 1:].to(device))


def group_reattach(old_state, new_state):
    return torch.cat((new_state, old_state[:, :, -1].unsqueeze(-1)), dim=-1)

def group_weight(module, name_list):
    group_decay = []
    group_no_decay = []
    for m in module.named_parameters():
        if m[0] in name_list:
            group_no_decay.append(m[1])
        else:
            group_decay.append(m[1])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    if len(group_no_decay) != 0:
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups
    else:
        return [dict(params=group_decay)]

# Pytorch Batched Calculations


def batchedDot(a, b):
    return torch.matmul(a.view([*a.shape[:-1], 1, a.shape[-1]]), b.view([*b.shape, 1])).squeeze(-1).squeeze(-1)


def batchedInv(batchedTensor):
        if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
            chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
            if chunk_num >= (256*256 - 1):
                print("TOO BIG TENSOR")
            max_split = (256 * 256 - 1)//chunk_num
            temp = []
            for t in torch.split(batchedTensor, max_split):
                temp.append(torch.inverse(t))
            return torch.cat(temp)
        else:
            return torch.inverse(batchedTensor)


def batchedDet(batchedTensor):
        if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
            chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
            if chunk_num >= (256*256 - 1):
                print("TOO BIG TENSOR")
            max_split = (256 * 256 - 1)//chunk_num
            temp = []
            for t in torch.split(batchedTensor, max_split):
                temp.append(torch.det(t))
            return torch.cat(temp)
        else:
            return torch.det(batchedTensor)


def batchedDet_old(batchedTensor):
    jitter = 1e-6
    if np.prod(batchedTensor.shape[:-2]) >= 256 * 256 - 1:
        chunk_num = int(np.prod(batchedTensor.shape[1:-2]))
        if chunk_num >= (256*256 - 1):
            print("TOO BIG TENSOR")
        max_split = (256 * 256 - 1)//chunk_num
        det_list = []
        for t in torch.split(batchedTensor, max_split):
            temp.append(torch.prod(torch.diagonal(torch.cholesky(t), dim1=-2, dim2=-1), dim = -1)**2)
        return torch.cat(temp)
    else:
        return torch.prod(torch.diagonal(torch.cholesky(batchedTensor), dim1=-2, dim2=-1), dim = -1)**2


# Pytorch Loss


def KLGaussianGaussian(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.
    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * torch.log(sig2) - 2 * torch.log(sig1) +
                    (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2 - 1)
    else:
        kl = torch.sum(0.5 * (2 * torch.log(sig2) - 2 * torch.log(sig1) +
                              (sig1 ** 2 + (mu1 - mu2) ** 2) /
                              sig2 ** 2 - 1), dim=-1)

    return kl


class NLL():
    def __init__(self, mode = 'default'):
        self.mode = mode

    def __call__(self, y, mu , sig, loss_out = True, add_const = True):
        #print(mu.shape, y.shape, sig.shape)
        neg_log_p = ((mu - y) ** 2 / (2 * sig**2))
        if add_const:
            const = 0.5 * torch.log(2 * sig**2)
            neg_log_p = neg_log_p + const
        if self.mode == 'default':
            return (neg_log_p) 
        elif self.mode == 'sum':
            return torch.sum(neg_log_p) / (y.size(0) * y.size(1))


def bivariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2]).reshape(2, 2)
    #print(mu)
    #print(cv)
    xmu = np.expand_dims(x - mu, axis=-1)

    z = np.squeeze(xmu.transpose(0, 1, 3, 2) @ np.linalg.inv(cv) @ xmu)
    denom = 2 * np.pi * np.sqrt(np.linalg.det(cv))

    return np.exp(-z / 2) / denom


def trivariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0], sig[0] * sig[2] * corr[1],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2, sig[1] * sig[2] * corr[2],
                   sig[0] * sig[2] * corr[1], sig[1] * sig[2] * corr[2], sig[2] ** 2]).reshape(3, 3)

    xmu = np.expand_dims(x - mu, axis=-1)
    z = np.squeeze(xmu.transpose(0, 1, 2, 4, 3) @ np.linalg.inv(cv) @ xmu)
    denom = np.power(2 * np.pi, 3 / 2) * np.sqrt(np.linalg.det(cv))

    return np.exp(-z / 2) / denom


def quadvariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0], sig[0] * sig[2] * corr[1], sig[0] * sig[3] * corr[2],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2, sig[1] * sig[2] * corr[3], sig[1] * sig[3] * corr[4],
                   sig[0] * sig[2] * corr[1], sig[1] * sig[2] * corr[3], sig[2] ** 2, sig[2] * sig[3] * corr[5],
                   sig[0] * sig[3] * corr[2], sig[1] * sig[3] * corr[4], sig[2] * sig[3] * corr[5], sig[3] ** 2]).reshape(3, 3)

    xmu = np.expand_dims(x - mu, axis=-1)
    z = np.squeeze(xmu.transpose(0, 1, 2, 3, 5, 4) @ np.linalg.inv(cv) @ xmu)
    denom = np.power(2 * np.pi, 4 / 2) * np.sqrt(np.linalg.det(cv))

    return np.exp(-z / 2) / denom


def hexavariate_normal(x, mu, sig, corr):
    cv = np.array([sig[0] ** 2, sig[0] * sig[1] * corr[0], sig[0] * sig[2] * corr[1], sig[0] * sig[3] * corr[2],
                   sig[0] * sig[4] * corr[3], sig[0] * sig[5] * corr[4],
                   sig[0] * sig[1] * corr[0], sig[1] ** 2, sig[1] * sig[2] * corr[5], sig[1] * sig[3] * corr[6],
                   sig[1] * sig[4] * corr[7], sig[1] * sig[5] * corr[8],
                   sig[0] * sig[2] * corr[1], sig[1] * sig[2] * corr[5], sig[2] ** 2, sig[2] * sig[3] * corr[9],
                   sig[2] * sig[4] * corr[10], sig[2] * sig[5] * corr[11],
                   sig[0] * sig[3] * corr[2], sig[1] * sig[3] * corr[6], sig[2] * sig[3] * corr[9], sig[3] ** 2,
                   sig[3] * sig[4] * corr[12], sig[3] * sig[5] * corr[13],
                   sig[0] * sig[4] * corr[3], sig[1] * sig[4] * corr[7], sig[2] * sig[4] * corr[10],
                   sig[3] * sig[4] * corr[12], sig[4] ** 2, sig[4] * sig[5] * corr[14],
                   sig[0] * sig[5] * corr[4], sig[1] * sig[5] * corr[8], sig[2] * sig[5] * corr[11],
                   sig[3] * sig[5] * corr[13], sig[4] * sig[5] * corr[14], sig[5] ** 2]).reshape(6, 6)

    xmu = np.expand_dims(x - mu, axis=-1)
    z = np.squeeze(xmu.transpose(0, 1, 2, 3, 4, 5, 7, 6) @ np.linalg.inv(cv) @ xmu)
    denom = np.power(2 * np.pi, 6 / 2) * np.sqrt(np.linalg.det(cv))
    return np.exp(-z / 2) / denom


def corr_bivariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * bivariate_normal(x, mu[i], sig[i], corr[i])
    return z


def corr_trivariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * trivariate_normal(x, mu[i], sig[i], corr[i])
    return z


def corr_quadvariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, :, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * quadvariate_normal(x, mu[i], sig[i], corr[i])
    return z


def corr_hexavariate(x, mu, sig, corr, coef):
    assert mu.shape[0] == sig.shape[0] == corr.shape[0] == coef.shape[0]
    z = np.zeros_like(x[:, :, :, :, :, :, 0])
    for i in range(coef.shape[0]):
        z += coef[i] * hexavariate_normal(x, mu[i], sig[i], corr[i])
    return z


class BiGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu ,sig, corr, coef, loss_out = True, cv_out = False):
        y = y.unsqueeze(-2)
        corr = corr.squeeze(-1)
        #print(y.shape, mu.shape, sig.shape, corr.shape)
        cv = torch.stack((sig[..., 0] ** 2, sig[..., 0] * sig[..., 1] * corr,
                        sig[..., 0] * sig[..., 1] * corr, sig[..., 1] ** 2),
                       dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 2, 2)
        inv_cv = batchedInv(cv)
        if cv_out and not loss_out:
            return cv
        else:
            if torch.sum(torch.isnan(inv_cv)) > 0:
                print(sig[0,0,:], corr[0,0,:])

            xmu = (y - mu).unsqueeze(-1)
            nll = 0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)).squeeze(-1)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class BiGMM_old():
    def __init__(self):
        pass
    def __call__(self, y, mu ,sig, corr, coef, loss_out = True, cv_out = False):
        y = y.unsqueeze(-2)
        corr = corr.squeeze(-1)
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr,
                        sig[:, :, :, 0] * sig[:, :, :, 1] * corr, sig[:, :, :, 1] ** 2),
                       dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 2, 2)
        inv_cv = batchedInv(cv)
        if cv_out and not loss_out:
            return cv
        else:
            if torch.sum(torch.isnan(inv_cv)) > 0:
                print(sig[0,0,:], corr[0,0,:])

            xmu = (y - mu).unsqueeze(-1)

            terms = -0.5 * (torch.log(cv[:, :, :, 0, 0] * cv[:, :, :, 1, 1] - cv[:, :, :, 1, 0] * cv[:, :, :, 0, 1])
                        + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
                            + torch.log(torch.tensor(2 * np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class TriGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss_out = True, cv_out = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1],
                sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 1] ** 2, sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2],
                sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1], sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2], sig[:, :, :, 2] ** 2), dim = -1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 3, 3)

        inv_cv = batchedInv(cv)
        #inv_cv = torch.tensor(np.linalg.inv(DCN(cv))).to(y.device)
        #inv_cv = torch.inverse(cv)
        if torch.sum(torch.isnan(inv_cv)) > 0:
            print('inv_cv')
            print(sig[0, 0, :], corr[0, 0, :])
        if cv_out and not loss_out:
            return cv
        else:
            xmu = (y - mu).unsqueeze(-1)
            nll = 0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)).squeeze(-1)
        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class TriGMM_old():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss_out = True, cv_out = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        cv = torch.stack((sig[:, :, :, 0] ** 2, sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1],
                sig[:, :, :, 0] * sig[:, :, :, 1] * corr[:, :, :, 0], sig[:, :, :, 1] ** 2, sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2],
                sig[:, :, :, 0] * sig[:, :, :, 2] * corr[:, :, :, 1], sig[:, :, :, 1] * sig[:, :, :, 2] * corr[:, :, :, 2], sig[:, :, :, 2] ** 2), dim = -1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 3, 3)

        #inv_cv = batchedInv(cv)
        inv_cv = torch.tensor(np.linalg.inv(DCN(cv))).to(y.device)

        #inv_cv = torch.inverse(cv)
        mode_num = coef.shape[-1]
        if torch.sum(torch.isnan(inv_cv)) > 0:
            print('inv_cv')
            print(sig[0, 0, :], corr[0, 0, :])
        if cv_out and not loss_out:
            return cv
        else:
            xmu = (y - mu).unsqueeze(-1)
            #det = torch.abs(cv[:, :, :, 0, 0]*(cv[:, :, :, 1, 1]*cv[:, :, :, 2, 2]-cv[:, :, :, 1, 2]*cv[:, :, :, 2, 1])-cv[:, :, :, 0, 1]*(cv[:, :, :, 1, 0]*cv[:, :, :, 2, 2]-cv[:, :, :, 1, 2]*cv[:, :, :, 2, 0])+cv[:, :, :, 0, 2]*(cv[:, :, :, 1, 0]*cv[:, :, :, 2, 1]-cv[:, :, :, 1, 1]*cv[:, :, :, 2, 0]))
            #terms = -0.5 * (torch.log(det) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1) + torch.log(torch.tensor(2 * np.pi)))
            terms = -0.5 * (torch.logdet(cv) + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1) + torch.log(torch.tensor(2 * np.pi)))
            nll = -torch.logsumexp(torch.log(coef) + terms, dim = -1)
            print(nll.shape, cv.shape)

        if loss_out and cv_out:
            return nll, cv
        else:
            return nll


class QuadGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss = True, cv = False):

        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        L = torch.stack(
            (sig[:,:,:,0], corr[:,:,:,0], corr[:,:,:,1], corr[:,:,:,2],
             zeros, sig[:,:,:,1], corr[:,:,:,3], corr[:,:,:,4],
             zeros, zeros, sig[:,:,:,2], corr[:,:,:,5],
             zeros, zeros, zeros, sig[:,:,:,3]),
            dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 4, 4)
        inv_cv = torch.matmul(L.transpose(-1, -2), L)
        if cv and not loss:
            return batchedInv(inv_cv)
        else:
            log_det = -2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
            if torch.sum(torch.isnan(log_det)) > 0:
                print(sig[0, 0, :], corr[0, 0, :])
            #print(y.shape, mu.shape)
            xmu = (y - mu).unsqueeze(-1)
            
            terms = -0.5 * (log_det
                            + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
                            + torch.log(torch.tensor(2 * np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss and cv:
            return nll, batchedInv(inv_cv)
        else:
            return nll


class HexaGMM():
    def __init__(self):
        pass
    def __call__(self, y, mu, sig, corr, coef, loss = True, cv = False):
        y = y.unsqueeze(-2)
        zeros = torch.zeros_like(sig[:,:,:,0])
        L = torch.stack((sig[:,:,:,0], corr[:,:,:,0], corr[:,:,:,1], corr[:,:,:,2], corr[:,:,:,3], corr[:,:,:,4],
                          zeros, sig[:,:,:,1], corr[:,:,:,5], corr[:,:,:,6], corr[:,:,:,7] , corr[:,:,:,8],
                         zeros, zeros, sig[:,:,:,2], corr[:,:,:,9], corr[:,:,:,10], corr[:,:,:,11],
                         zeros, zeros, zeros, sig[:,:,:,3], corr[:,:,:,12], corr[:,:,:,13],
                         zeros, zeros, zeros, zeros, sig[:,:,:,4], corr[:,:,:,14],
                         zeros, zeros, zeros, zeros, zeros, sig[:,:,:,5]),
                       dim=-1).reshape(sig.shape[0], sig.shape[1], sig.shape[2], 6, 6)
        inv_cv = torch.matmul(L.transpose(-1, -2), L)
        if cv and not loss:
            return batchedInv(inv_cv)
        else:

            log_det = -2*torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2 = -1)), dim = -1)
            if torch.sum(torch.isnan(log_det)) > 0:
                print(sig[0,0,:], corr[0,0,:])
            #print(y.shape, mu.shape)
            xmu = (y - mu).unsqueeze(-1)

            terms = -0.5 * (log_det
            + torch.matmul(torch.matmul(xmu.transpose(-1, -2), inv_cv), xmu).squeeze(-1).squeeze(-1)
            + torch.log(torch.tensor(2*np.pi)))

            nll = -torch.logsumexp(torch.log(coef) + terms, dim=-1)

        if loss and cv:
            return nll, batchedInv(inv_cv)
        else :
            return nll


def gmm_criterion(D_s, mode = 'default'):
    criterion = None
    if D_s == 1:
        criterion = NLL(mode)
    elif  D_s== 2:
        criterion = BiGMM()
    elif D_s == 3:
        criterion = TriGMM()
    elif D_s == 4:
        criterion = QuadGMM()
    elif D_s == 6:
        criterion = HexaGMM()
    else:
        print('NOT IMPLEMENTED : GMM')
    return criterion


class gmm_sample():
    def __init__(self, D_s, r=False):
        self.D_s = D_s
        self.r = r
    def __call__(self, mu, L):
        if self.D_s == 1:
            distrib = torch.distributions.Normal(mu.cpu(), L.cpu())
            if self.r:
                sampled_mu = distrib.rsample()
            else:
                sampled_mu = distrib.sample()
            return sampled_mu
        else:
            original_shape = mu.shape
            mu = mu.view(-1, self.D_s)
            L = L.view(-1, self.D_s, self.D_s)
            try:
                distrib = torch.distributions.MultivariateNormal(loc=mu.cpu(), covariance_matrix=L.cpu())
                if self.r:
                    sampled_mu = distrib.rsample()
                else:
                    sampled_mu = distrib.sample()
                return sampled_mu.view(original_shape)
            except Exception as e:
                print(e)
                return None


# Pytorch Activations


class SReLU(nn.Module):
    """
    SReLU (S-shaped Rectified Linear Activation Unit): a combination of three linear functions, which perform mapping R → R with the following formulation:
    .. math::
        h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.
    with 4 trainable parameters.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        .. math:: \\{t_i^r, a_i^r, t_i^l, a_i^l\\}
    4 trainable parameters, which model an individual SReLU activation unit. The subscript i indicates that we allow SReLU to vary in different channels. Parameters can be initialized manually or randomly.
    References:
        - See SReLU paper:
        https://arxiv.org/pdf/1512.07030.pdf
    Examples:
        >>> srelu_activation = srelu((2,2))
        >>> t = torch.randn((2,2), dtype=torch.float, requires_grad = True)
        >>> output = srelu_activation(t)
    """

    def __init__(self, in_features=1, parameters=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - parameters: (tr, tl, ar, al) parameters for manual initialization, default value is None. If None is passed, parameters are initialized randomly.
        """
        super(SReLU, self).__init__()
        self.in_features = in_features

        if parameters is None:
            self.tr = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
            self.tl = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
            self.ar = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
            self.al = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
        else:
            self.tr, self.tl, self.ar, self.al = parameters

    def forward(self, x):
        """
        Forward pass of the function
        """
        return (
            (x >= self.tr).float() * (self.tr + self.ar * (x + self.tr))
            + (x < self.tr).float() * (x > self.tl).float() * x
            + (x <= self.tl).float() * (self.tl + self.al * (x + self.tl))
        )


class SReLU_limited(nn.Module):
    """
    SReLU (S-shaped Rectified Linear Activation Unit): a combination of three linear functions, which perform mapping R → R with the following formulation:
    .. math::
        h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.
    with 4 trainable parameters.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        .. math:: \\{t_i^r, a_i^r, t_i^l, a_i^l\\}
    4 trainable parameters, which model an individual SReLU activation unit. The subscript i indicates that we allow SReLU to vary in different channels. Parameters can be initialized manually or randomly.
    References:
        - See SReLU paper:
        https://arxiv.org/pdf/1512.07030.pdf
    Examples:
        >>> srelu_activation = srelu((2,2))
        >>> t = torch.randn((2,2), dtype=torch.float, requires_grad = True)
        >>> output = srelu_activation(t)
    """

    def __init__(self, parameters=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - parameters: (tr, tl, ar, al) parameters for manual initialization, default value is None. If None is passed, parameters are initialized randomly.
        """
        super(SReLU_limited, self).__init__()

        if parameters is None:
            self.tr = Parameter(
                torch.tensor(2.0, dtype=torch.float, requires_grad=True)
            )
            self.tl = Parameter(
                torch.tensor(-2.0, dtype=torch.float, requires_grad=True)
            )

            self.ar = Parameter(
                torch.tensor(2.0, dtype=torch.float, requires_grad=True)
            )

        else:
            self.tr, self.tl, self.yr, self.yl = parameters

    def forward(self, x):
        """
        Forward pass of the function
        """
        return (
            (x >= self.tr).float() * self.ar
            + (x < self.tr).float() * (x > self.tl).float() * self.ar * (x-self.tl) / (self.tr-self.tl)
            + (x <= self.tl).float() * 0
        )


# Neural Network Blocks


class mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def MLP_layers(cfg, nl_type, batch_norm=False, dropout=False, final_bias=True):
    layers = []
    nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2), 'EL':nn.ELU(), 'GL': nn.GELU(), 'SL': nn.SELU(), 'MS': mish()}
    nl = nl_dict[nl_type]

    for i in range(1, len(cfg)):
        if i != len(cfg) - 1:
            layers += [('FC' + str(i) + '0', nn.Linear(cfg[i - 1], cfg[i], bias=True))]
            if batch_norm:
                layers += [('BN' + str(i) + '0', nn.BatchNorm1d(batch_norm))]
            if dropout:
                layers += [('DO' + str(i) + '0', nn.Dropout(dropout))]
            layers += [(nl_type + str(i) + '0', nl)]
        else:
            layers += [('FC' + str(i) + '0', nn.Linear(cfg[i - 1], cfg[i], bias=final_bias))]

    return nn.Sequential(OrderedDict(layers))


def Res_layers(cfg, nl_type, batch_norm=False, dropout=False):
    meta_layers = []
    nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2), 'EL':nn.ELU(), 'GL' : nn.GELU(), 'SL' : nn.SELU(), 'MS' : mish()}
    nl = nl_dict[nl_type]
    bias=True if len(cfg) > 2 else False

    for i in range(1, len(cfg)):
        layers = []
        for j in range(2):
            layers += [('FC' + str(i + 1) + str(j), nn.Linear(cfg[i], cfg[i], bias=bias))]
            if batch_norm:
                layers += [('BN' + str(i + 1) + str(j), nn.BatchNorm1d(batch_norm))]
            if dropout:
                layers += [('DO' + str(i + 1) + str(j), nn.Dropout(dropout))]
            if j == 0:
                layers += [(nl_type + str(i + 1) + '0', nl)]

        meta_layers.append(nn.Sequential(OrderedDict(layers)))

    return nn.Sequential(*meta_layers)


class MLP_Block(nn.Module):
    def __init__(self, cfg, nl_type, batch_norm=False, dropout=False, final_bias=True):
        super(MLP_Block, self).__init__()
        self.FC = MLP_layers(cfg, nl_type, batch_norm, dropout, final_bias=final_bias)

    def forward(self, x):
        return self.FC(x)


class Res_Block(nn.Module):
    def __init__(self, cfg, nl_type, batch_norm=False, dropout=False):
        super(Res_Block, self).__init__()

        nl_dict = {'RL': nn.ReLU(), 'TH': nn.Tanh(), 'LR': nn.LeakyReLU(0.2), 'GL' : nn.GELU(), 'SL' : nn.SELU(), 'MS' : mish()}
        self.nl = nl_dict[nl_type]
        self.cfg = cfg

        self.FC1 = MLP_layers(cfg[:2], nl_type)
        self.RS = Res_layers(cfg[1:-1], nl_type, batch_norm, dropout)
        self.FC2 = MLP_layers(cfg[-2:], nl_type)

    def forward(self, x):
        x = self.FC1(x)
        for m in self.RS.children():
            x = self.nl(m(x) + x)
        x = self.FC2(x)
        return x


def cfg_Block(block_type, cfg, nl_type, batch_norm=False, dropout=False, final_bias=True):
    if block_type == 'mlp':
        block = MLP_Block(cfg, nl_type, batch_norm, dropout, final_bias=final_bias)
    elif block_type =='res':
        block = Res_Block(cfg, nl_type, batch_norm, dropout)
    else:
        print("NOT IMPLEMENTED : cfg_Block")
    return block

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# Utility classes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)