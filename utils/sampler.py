import random
from itertools import product

import torch


class BatchSampler:
    """Random subset sampling from {0, 1, ..., E-1} X {0, 1, ..., M-1} X {0, 1, ..., L-1} where X is Cartesian product.
    Attributes:
        E: number of ensembles.
        M: number of agents.
        L: trajectory length.
        batch_size: input batch size for training the policy $ \pi $ and state-value ftn $ v $.
        train: if True randomly sample a subset else ordered sample. (default: True)
    Examples::
        >>> # 16 ensembles, 100 agents, trajectory length 50, batch size 32 for training
        >>> sampler = BatchSampler(16, 100, 50, 32)
        >>> batch = next(sampler)
    """

    def __init__(self, E, M, L, batch_size, device="cpu", train=True):
        self.size = E * M * L
        self.E = E
        self.M = M
        self.L = L
        self.batch_size = batch_size
        self.device = device
        self.training = train
        self.index = 0

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.training:
            ens_idx = torch.randint(self.E, (self.batch_size,), device=self.device)
            agn_idx = torch.randint(self.M, (self.batch_size,), device=self.device)
            traj_idx = torch.randint(self.L, (self.batch_size,), device=self.device)
            return (ens_idx, agn_idx, traj_idx)
        else:
            prev_idx = self.index * self.batch_size
            next_idx = (self.index + 1) * self.batch_size
            if prev_idx >= self.size:
                raise StopIteration
            elif next_idx >= self.size:
                next_idx = self.size
            ens_idx = torch.arange(prev_idx, next_idx, device=self.device) // (self.M * self.L)
            agn_idx = torch.arange(prev_idx, next_idx, device=self.device) % (self.M * self.L) // self.L
            traj_idx = torch.arange(prev_idx, next_idx, device=self.device) % self.L
            self.index += 1
            return (ens_idx, agn_idx, traj_idx)
            

class BatchSampler_split:
    """Random subset sampling from {0, 1, ..., E-1} X {0, 1, ..., M-1} X {0, 1, ..., N-1} X {0, 1, ..., L-1} where X is Cartesian product.
    Attributes:
        E: number of ensembles.
        M: number of agents.
        N: number of gene dimensions
        L: trajectory length.
        batch_size: input batch size for training the policy $ \pi $ and state-value ftn $ v $.
        train: if True randomly sample a subset else ordered sample. (default: True)
    Examples::
        >>> # 16 ensembles, 100 agents, 15 gene dimensions, trajectory length 50, batch size 32 for training
        >>> sampler = BatchSampler(16, 100, 15, 50, 32)
        >>> batch = next(sampler)
    """

    def __init__(self, E, M, N, L, batch_size, device="cpu", train=True):
        self.size = E * M * N * L
        self.E = E
        self.M = M
        self.N = N
        self.L = L
        self.batch_size = batch_size
        self.device = device
        self.training = train
        self.index = 0

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.training:
            ens_idx = torch.randint(self.E, (self.batch_size,), device=self.device)
            agn_idx = torch.randint(self.M, (self.batch_size,), device=self.device)
            dim_idx = torch.randint(self.N, (self.batch_size,), device=self.device)
            traj_idx = torch.randint(self.L, (self.batch_size,), device=self.device)
            return (ens_idx, agn_idx, dim_idx, traj_idx)
        else:
            prev_idx = self.index * self.batch_size
            next_idx = (self.index + 1) * self.batch_size
            if prev_idx >= self.size:
                raise StopIteration
            elif next_idx >= self.size:
                next_idx = self.size
            ens_idx = torch.arange(prev_idx, next_idx, device=self.device) // (self.M * self.N * self.L)
            agn_idx = torch.arange(prev_idx, next_idx, device=self.device) % (self.M * self.N * self.L) // (self.N * self.L)
            dim_idx = torch.arange(prev_idx, next_idx, device=self.device) % (self.M * self.N * self.L) % (self.N * self.L) // self.L
            traj_idx = torch.arange(prev_idx, next_idx, device=self.device) % self.L
            self.index += 1
            return (ens_idx, agn_idx, dim_idx, traj_idx)