import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode

import networkx as nx
from itertools import permutations
import random
from jitcdde import jitcdde, y, t
from numpy import pi, arange, random, max
from symengine import sin

def sign(size):
    return np.random.randint(0, 2, (size)) * 2 - 1

class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=1.,
                 interaction_strength=.05, noise_var=0.0, delta_T=0.005):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        # ball30 : interaction strength = .05, noise_var = 0., loc_std, =2., vel_norm = 1.
        # ball5N : interaction strength = .1, noise_var = .01, loc_std, vel_norm = .5
        # ball5N : interaction strength = .1, noise_var = 0., loc_std, vel_norm = .5

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = delta_T
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,
                          edge_prob=[0.5, 0.5]): #### spring_prob=[1. / 2, 0, 1. / 2]
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq) - 1
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        # Sample edges
        edges = np.random.uniform(0, 1, size=(self.n_balls, self.n_balls))
        mask = np.random.choice(
            [0, 1], size=(self.n_balls, self.n_balls), p=edge_prob
        )
        edges = edges * mask
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std     # randn samples from a unit normal dist.
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges

    def sample_trajectory_test(self, T, sample_freq, edges, loc, vel, loc_next, vel_next): #### spring_prob=[1. / 2, 0, 1. / 2]
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq) - 1
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size, 0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges

class KuramotoSim(object):
    """
    Implementation of Kuramoto coupling model [1] with harmonic terms
    and possible perturbation.
    It uses NumPy and Scipy's implementation of Runge-Kutta 4(5)
    for numerical integration.

    Usage example:
    >>> kuramoto = Kuramoto(initial_values)
    >>> phase = kuramoto.solve(X)

    [1] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
        (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3
    """


    def __init__(self, n_osc, interaction_type, sample_mode, dt=0.01, m_order=1, noise = None):
        """
        Passed arguments should be a dictionary with NumPy arrays
        for initial phase (Y0), intrisic frequencies (W) and coupling
        matrix (K).
        """
        self.dtype = np.float32

        self.dt = dt
        self.init_phase = None
        self.W = None
        self.K = None

        self.n_osc = n_osc
        self.m_order = m_order
        self.noise = noise
        
        self.interaction_type = interaction_type  # N : normal, S : signed, D : directed, SD : signed+directed
        self.sample_mode = sample_mode

    def kuramoto_ODE(self, t, y, arg):
        """General Kuramoto ODE of m'th harmonic order.
           Argument `arg` = (w, k), with
            w -- iterable frequency
            k -- 3D coupling matrix, unless 1st order
            """

        w, k = arg
        yt = y[:,None]
        dy = y-yt
        phase = w.astype(self.dtype)
        if self.noise != None:
            n = np.random.randn() * self.noise
            phase += n
        for m, _k in enumerate(k):
            phase += np.sum(_k*np.sin((m+1)*dy),axis=1)

        return phase

    def kuramoto_ODE_jac(self, t, y, arg):
        """Kuramoto's Jacobian passed for ODE solver."""

        w, k = arg
        yt = y[:,None]
        dy = y-yt

        phase = [m*k[m-1]*np.cos(m*dy) for m in range(1,1+self.m_order)]
        phase = np.sum(phase, axis=0)

        for i in range(self.n_osc):
            phase[i,i] = -np.sum(phase[:,i])

        return phase

    def sample_trajectory(self, T=10000, sample_freq=100, edge_prob=[0.5, 0.5]):
    
        """Solves Kuramoto ODE for time series `t` with initial
        parameters passed when initiated object.
        """
        kODE = ode(self.kuramoto_ODE, jac=self.kuramoto_ODE_jac)
        kODE.set_integrator("dopri5")

        # Set parameters into model
        order = np.arange(self.n_osc)
        init_phase = np.random.uniform(0, 2 * np.pi, self.n_osc)
        w = np.abs(np.random.randint(1, 10, self.n_osc))

        if self.sample_mode == 'constant':
            p = edge_prob[1]
            q = (self.n_osc * (self.n_osc - 1) / (self.n_osc**2)) * p  # modified p
            edge_prob = [1 - q, q]
            mask = np.random.choice(
                [0, 1], size=(self.n_osc, self.n_osc), p=edge_prob
            )
            edges = mask
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)

        elif self.sample_mode == 'uniform':
            p = edge_prob[1]
            q = (self.n_osc * (self.n_osc - 1) / (self.n_osc**2)) * p  # modified p
            edge_prob = [1 - q, q]
            if 'S' not in self.interaction_type:
                edges = np.random.uniform(0, 2, size=(self.n_osc, self.n_osc))
            else:
                edges = np.random.uniform(-1, 1, size=(self.n_osc, self.n_osc))
            mask = np.random.choice(
                [0, 1], size=(self.n_osc, self.n_osc), p=edge_prob
            )
            edges = edges * mask
            if 'D' not in self.interaction_type:
                edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)

        elif self.sample_mode == 'normal':
            p = edge_prob[1]
            q = (self.n_osc * (self.n_osc - 1) / (self.n_osc**2)) * p  # modified p
            edge_prob = [1 - q, q]
            edges = np.abs(np.random.normal(0, 1, size=(self.n_osc, self.n_osc)))
            mask = np.random.choice(
                [0, 1], size=(self.n_osc, self.n_osc), p=edge_prob
            )
            edges = edges * mask
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)
            
        elif self.sample_mode == 'duplex':

            # [[A, B], [C, D]]
            assert self.n_osc % 2 == 0
            np.random.shuffle(order)
            
            p = edge_prob[1]
            q = (self.n_osc * (self.n_osc - 1) / (self.n_osc**2)) * p  # modified p
            edge_prob = [1 - q, q]

            edges_A = np.random.uniform(1, 2, (int(self.n_osc / 2), int(self.n_osc / 2)))
            mask_A = np.random.choice(
                [0, 1], size=(int(self.n_osc / 2), int(self.n_osc / 2)), p=edge_prob
            )
            edges_A = edges_A * mask_A
            edges_A = np.tril(edges_A) + np.tril(edges_A, -1).T
            np.fill_diagonal(edges_A, 0)
            
            edges_C = np.random.uniform(1, 2, (int(self.n_osc / 2), int(self.n_osc / 2)))
            mask_C = np.random.choice(
                [0, 1], size=(int(self.n_osc / 2), int(self.n_osc / 2)), p=edge_prob
            )
            edges_C = edges_C * mask_C
            edges_C = np.tril(edges_C) + np.tril(edges_C, -1).T
            np.fill_diagonal(edges_C, 0)
            
            #edges_B = np.ones((int(self.n_osc / 2), int(self.n_osc / 2))) * 0.5
            edges_B = np.random.binomial(1, 0.2, (int(self.n_osc / 2), int(self.n_osc / 2))) * 0.3
            edges = np.block([[edges_A, edges_B], [edges_B, edges_C]])
            edges = edges[order][:, order]
        else:
            raise NotImplementedError
        
        if 'T' not in self.interaction_type:
            k = np.expand_dims(edges, 0)
            t = np.arange(0, self.dt * (T + 1), self.dt)

            kODE.set_initial_value(init_phase, t[0])
            kODE.set_f_params((w, k))
            kODE.set_jac_params((w, k))

            phase = np.empty((self.n_osc, len(t)))

            # Run ODE integrator
            for idx, _t in enumerate(t[1:]):
                phase[:,idx] = kODE.y
                kODE.integrate(_t)

            phase[:,-1] = kODE.y
            phase = phase[:, ::sample_freq]
            
            dphi = np.diff(phase)/self.dt
            sinphi = np.sin(phase[:, :-1])
            phi = phase[:, :-1]
            freq = w
            edges = k

        else:  # Time-delyaed ODE solver
            τ = 0.3  # now fixed
            def kuramotos():
                for i in range(self.n_osc):
                    yield w[i] + sum(sin(y(j, t - τ) - y(i))for j in range(self.n_osc)if edges[j, i])

            I = jitcdde(kuramotos, n=self.n_osc, verbose=False)
            I.set_integration_parameters(rtol=0, atol=1e-5)
            I.constant_past(random.uniform(0, 2 * pi, self.n_osc), time=0.0)
            I.integrate_blindly(τ, 0.1)

            phase = []
            for time in I.t + arange(0, self.dt * (T + 1), self.dt):
                phase.append(I.integrate(time))

            phase = (np.array(phase).T)[:, ::sample_freq]
            sinphi = np.sin(phase[:, :-1])
            dphi = np.diff(phase) / self.dt
            phi = phase[:, :-1]
            freq = w
            edges = np.expand_dims(edges, 0)

        return dphi, sinphi, phi, freq, edges, order

    def sample_trajectory_test(self, T=10000, sample_freq=100, edge_prob=[0.5, 0.5], init_phase=None, w=None, edges=None):
        """Solves Kuramoto ODE for time series `t` with initial
        parameters passed when initiated object.
        """
        kODE = ode(self.kuramoto_ODE, jac=self.kuramoto_ODE_jac)
        kODE.set_integrator("dopri5")

        # Set parameters into model
        order = np.arange(self.n_osc)
        
        if 'T' not in self.interaction_type:
            k = np.expand_dims(edges, 0)
            t = np.arange(0, self.dt * (T + 1), self.dt)

            kODE.set_initial_value(init_phase, t[0])
            kODE.set_f_params((w, k))
            kODE.set_jac_params((w, k))

            phase = np.empty((self.n_osc, len(t)))

            # Run ODE integrator
            for idx, _t in enumerate(t[1:]):
                phase[:,idx] = kODE.y
                kODE.integrate(_t)

            phase[:,-1] = kODE.y
            phase = phase[:, ::sample_freq]
            
            dphi = np.diff(phase)/self.dt
            sinphi = np.sin(phase[:, :-1])
            phi = phase[:, :-1]
            freq = w
            edges = k

        else:  # Time-delyaed ODE solver
            τ = 0.3  # now fixed
            def kuramotos():
                for i in range(self.n_osc):
                    yield w[i] + sum(sin(y(j, t - τ) - y(i))for j in range(self.n_osc)if edges[j, i])

            I = jitcdde(kuramotos, n=self.n_osc, verbose=False)
            I.set_integration_parameters(rtol=0, atol=1e-5)
            I.constant_past(random.uniform(0, 2 * pi, self.n_osc), time=0.0)
            I.integrate_blindly(τ, 0.1)

            phase = []
            for time in I.t + arange(0, self.dt * (T + 1), self.dt):
                phase.append(I.integrate(time))

            phase = (np.array(phase).T)[:, ::sample_freq]
            sinphi = np.sin(phase[:, :-1])
            dphi = np.diff(phase) / self.dt
            phi = phase[:, :-1]
            freq = w
            edges = np.expand_dims(edges, 0)

        return dphi, sinphi, phi, freq, edges, order

class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=1.,
                 interaction_strength=1., noise_var=0.0, delta_T=0.005):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._delta_T = delta_T
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=10000, sample_freq=10, edge_prob=[0.5, 0.5]):
        
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq) - 1
        counter = 0

        p = edge_prob[1]
        q = (n * (n - 1) / (n ** 2)) * p  # modified p
        edge_prob = [1 - q, q]
        edges = np.random.uniform(-1, 1, size=(n, n))
        mask = np.random.choice(
            [0, 1], size=(n, n), p=edge_prob
        )
        edges = edges * mask
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)

        # Initialize location and velocity

        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore', invalid='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            #assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges

class SpringChargeSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5, noise_var=0.,
                 spring_interaction_strength=.1, 
                 charge_interaction_strength=.2,
                 spring_types=[0., 0.5, 1.],
                 charge_types=[-1., 0., 1.],
                 spring_prob=[1./2, 0, 1./2],
                 charge_prob=[0, 0.5, 0.5],
                 uniform_draw=True):

        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.spring_interaction_strength = spring_interaction_strength
        self.charge_interaction_strength = charge_interaction_strength
        self.noise_var = noise_var

        self.spring_types = np.array(spring_types)
        self.charge_types = np.array(charge_types)
        self.spring_prob   = spring_prob
        self.charge_prob   = charge_prob

        self.uniform_draw = uniform_draw

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges_spring, edges_charge):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.spring_interaction_strength * edges_spring[
                            i, j] * (dist ** 2) / 2
                        U += 0.5 * self.charge_interaction_strength * edges_charge[
                            i, j] / dist
            return U + K


    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _edge_type_encode(edges): # this is used to gives each 'interaction strength' a unique integer = 0, 1, 2 ..
        unique = np.unique(edges)
        encode = np.zeros(edges.shape)
        for i in range(unique.shape[0]):
            encode += np.where( edges == unique[i], i, 0)
        return encode

    def sample_trajectory(self, T=10000, sample_freq=10):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0

        if self.uniform_draw:

            total_num_edges = int(0.5*self.n_balls*(self.n_balls-1))
            
            num_edges = random.randint(0, total_num_edges)
            edges = [0 for i in range(total_num_edges)]
            for i in range(num_edges):
                edges[i] = 1
            random.shuffle(edges)
            spring_edges = np.zeros( (self.n_balls,self.n_balls) )
            spring_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
            spring_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)

            charges = [0 for i in range(self.n_balls)]
            n_c = random.randint(1, self.n_balls)       # choose a random number of charges, 1 to 5
            for i in range(n_c):
                charges[i] = 1
            random.shuffle(charges)
            charges = np.expand_dims(np.array(charges),-1)
            charge_edges = charges.dot(charges.transpose()).astype('float')


        else:
            spring_edges = np.random.choice(self.spring_types, # self.spring_types is an array of relative spring strengths eg. [0., 0.5, 1.]
                                     size=(self.n_balls, self.n_balls),
                                     p=self.spring_prob)      # prob. of each spring type
            spring_edges = np.tril(spring_edges) + np.tril(spring_edges, -1).T # this makes the edges matrix symmetric
            np.fill_diagonal(spring_edges, 0)                    # remove self loops

            # Sample charge edges
            charges = np.random.choice(self.charge_types, size=(self.n_balls, 1),p=self.charge_prob)
            charge_edges = charges.dot(charges.transpose())
            #np.fill_diagonal(charge_edges, 0)                    # remove self loops

        
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std     # randn samples from a unit normal dist.
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore', invalid='ignore'):

            spring_forces_size = - self.spring_interaction_strength * spring_edges
            #np.fill_diagonal(spring_forces_size, 0)  # self forces are zero (fixes division by zero)

            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            charge_forces_size = - self.charge_interaction_strength * charge_edges / l2_dist_power3
            np.fill_diagonal(charge_forces_size, 0)  # self forces are zero (fixes division by zero)
            #assert (np.abs(charge_forces_size[diag_mask]).min() > 1e-10)

            F_s = (spring_forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            #assert (np.abs(charge_forces_size[diag_mask]).min() > 1e-10)
            F_c = (charge_forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F = F_s + F_c

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
                # size of forces up to a 1/|r| factor
                # since I later multiply by an unnormalized r vector
                charge_forces_size = self.charge_interaction_strength * charge_edges / l2_dist_power3
                np.fill_diagonal(charge_forces_size, 0)  # self forces are zero (fixes division by zero)

                F_s = (spring_forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n)))).sum(
                    axis=-1)
                #assert (np.abs(charge_forces_size[diag_mask]).min() > 1e-10)
                F_c = (charge_forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n)))).sum(
                    axis=-1)
                F = F_s + F_c

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            np.fill_diagonal(charge_edges, 0)
            edges = np.concatenate( (np.expand_dims(spring_edges,0), np.expand_dims(charge_edges,0) ), axis=0)
            
            return loc, vel, edges

class SpringChargeQuadSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5, noise_var=0.,
                 spring_interaction_strength=.1, 
                 charge_interaction_strength=.2,
                 quad_interaction_strength=.1 ):

        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.spring_interaction_strength = spring_interaction_strength
        self.charge_interaction_strength = charge_interaction_strength
        self.quad_interaction_strength = quad_interaction_strength
        self.noise_var = noise_var

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges_spring, edges_charge):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.spring_interaction_strength * edges_spring[
                            i, j] * (dist ** 2) / 2
                        U += 0.5 * self.charge_interaction_strength * edges_charge[
                            i, j] / dist
            return U + K


    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _edge_type_encode(edges): # this is used to gives each 'interaction strength' a unique integer = 0, 1, 2 ..
        unique = np.unique(edges)
        encode = np.zeros(edges.shape)
        for i in range(unique.shape[0]):
            encode += np.where( edges == unique[i], i, 0)
        return encode

    def _get_force(self, forces_size, loc_next):
        n = self.n_balls
        F = (forces_size.reshape(1, n, n) *
                np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
        return F


    def sample_trajectory(self, T=10000, sample_freq=10):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0


        total_num_edges = int(0.5*self.n_balls*(self.n_balls-1))
        
        num_spring_edges = random.randint(0, total_num_edges)
        edges = [0 for i in range(total_num_edges)]
        for i in range(num_spring_edges):
            edges[i] = 1
        random.shuffle(edges)
        spring_edges = np.zeros( (self.n_balls,self.n_balls) )
        spring_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        spring_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)

        charges = [0 for i in range(self.n_balls)]
        n_c = random.randint(1, self.n_balls)       # choose a random number of charges, 1 to 5
        for i in range(n_c):
            charges[i] = 1
        random.shuffle(charges)
        charges = np.expand_dims(np.array(charges),-1)
        charge_edges = charges.dot(charges.transpose()).astype('float')

        num_quad_edges = random.randint(0, total_num_edges)
        edges = [0 for i in range(total_num_edges)]
        for i in range(num_quad_edges):
            edges[i] = 1
        random.shuffle(edges)
        quad_edges = np.zeros( (self.n_balls,self.n_balls) )
        quad_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        quad_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std     # randn samples from a unit normal dist.
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore', invalid='ignore'):

            spring_forces_size = - self.spring_interaction_strength * spring_edges
            #np.fill_diagonal(spring_forces_size, 0)  # self forces are zero (fixes division by zero)

            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            charge_forces_size = - self.charge_interaction_strength * charge_edges / l2_dist_power3
            np.fill_diagonal(charge_forces_size, 0)  # self forces are zero (fixes division by zero)
            #assert (np.abs(charge_forces_size[diag_mask]).min() > 1e-10)

            l2_dist_powerhalf = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 1. / 2.)
            quad_forces_size = - self.quad_interaction_strength * quad_edges *l2_dist_powerhalf
            np.fill_diagonal(quad_forces_size, 0)  # self forces are zero (fixes division by zero)


            F_s = self._get_force(spring_forces_size, loc_next)
            F_c = self._get_force(charge_forces_size, loc_next)
            F_q = self._get_force(quad_forces_size, loc_next)
            F = F_s + F_c + F_q

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
                # size of forces up to a 1/|r| factor
                # since I later multiply by an unnormalized r vector
                charge_forces_size = self.charge_interaction_strength * charge_edges / l2_dist_power3
                np.fill_diagonal(charge_forces_size, 0)  # self forces are zero (fixes division by zero)

                l2_dist_powerhalf = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 1. / 2.)
                quad_forces_size = - self.quad_interaction_strength * quad_edges * l2_dist_powerhalf
                np.fill_diagonal(quad_forces_size, 0)  # self forces are zero (fixes division by zero)

                F_s = self._get_force(spring_forces_size, loc_next)
                F_c = self._get_force(charge_forces_size, loc_next)
                F_q = self._get_force(quad_forces_size, loc_next)
                F = F_s + F_c + F_q

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            np.fill_diagonal(charge_edges, 0)
            edges = np.concatenate( (np.expand_dims(spring_edges,0), 
                                     np.expand_dims(charge_edges,0), 
                                     np.expand_dims(quad_edges,0)), axis=0)
            
            return loc, vel, edges

class SpringQuadSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5, noise_var=0.,
                 spring_interaction_strength=.1, 
                 quad_interaction_strength=.1 ):

        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.spring_interaction_strength = spring_interaction_strength
        self.quad_interaction_strength = quad_interaction_strength
        self.noise_var = noise_var

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges_spring, edges_charge):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.spring_interaction_strength * edges_spring[
                            i, j] * (dist ** 2) / 2
                        U += 0.5 * self.charge_interaction_strength * edges_charge[
                            i, j] / dist
            return U + K


    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _edge_type_encode(edges): # this is used to gives each 'interaction strength' a unique integer = 0, 1, 2 ..
        unique = np.unique(edges)
        encode = np.zeros(edges.shape)
        for i in range(unique.shape[0]):
            encode += np.where( edges == unique[i], i, 0)
        return encode

    def _get_force(self, forces_size, loc_next):
        n = self.n_balls
        F = (forces_size.reshape(1, n, n) *
                np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
        return F


    def sample_trajectory(self, T=10000, sample_freq=10):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0


        total_num_edges = int(0.5*self.n_balls*(self.n_balls-1))
        
        num_spring_edges = random.randint(0, total_num_edges)
        edges = [0 for i in range(total_num_edges)]
        for i in range(num_spring_edges):
            edges[i] = 1
        random.shuffle(edges)
        spring_edges = np.zeros( (self.n_balls,self.n_balls) )
        spring_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        spring_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)

        num_quad_edges = random.randint(0, total_num_edges)
        edges = [0 for i in range(total_num_edges)]
        for i in range(num_quad_edges):
            edges[i] = 1
        random.shuffle(edges)
        quad_edges = np.zeros( (self.n_balls,self.n_balls) )
        quad_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        quad_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std     # randn samples from a unit normal dist.
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore', invalid='ignore'):

            spring_forces_size = - self.spring_interaction_strength * spring_edges
            #np.fill_diagonal(spring_forces_size, 0)  # self forces are zero (fixes division by zero)

            l2_dist_powerhalf = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 1. / 2.)
            quad_forces_size = - self.quad_interaction_strength * quad_edges *l2_dist_powerhalf
            np.fill_diagonal(quad_forces_size, 0)  # self forces are zero (fixes division by zero)


            F_s = self._get_force(spring_forces_size, loc_next)
            F_q = self._get_force(quad_forces_size, loc_next)
            F = F_s + F_q

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_powerhalf = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 1. / 2.)
                quad_forces_size = - self.quad_interaction_strength * quad_edges * l2_dist_powerhalf
                np.fill_diagonal(quad_forces_size, 0)  # self forces are zero (fixes division by zero)

                F_s = self._get_force(spring_forces_size, loc_next)
                F_q = self._get_force(quad_forces_size, loc_next)
                F = F_s + F_q

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            np.fill_diagonal(quad_edges, 0)
            edges = np.concatenate( (np.expand_dims(spring_edges,0), 
                                     np.expand_dims(quad_edges,0)), axis=0)
            
            return loc, vel, edges

class SpringChargeFspringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5, noise_var=0.,
                 spring_interaction_strength=.1, 
                 charge_interaction_strength=.2,
                 fspring_interaction_strength=.1,
                 fspring_length=1. ):

        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.spring_interaction_strength = spring_interaction_strength
        self.charge_interaction_strength = charge_interaction_strength
        self.fspring_interaction_strength = fspring_interaction_strength
        self.fspring_length = fspring_length
        self.noise_var = noise_var

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _edge_type_encode(edges): # this is used to gives each 'interaction strength' a unique integer = 0, 1, 2 ..
        unique = np.unique(edges)
        encode = np.zeros(edges.shape)
        for i in range(unique.shape[0]):
            encode += np.where( edges == unique[i], i, 0)
        return encode

    def _get_force(self, forces_size, loc_next):
        n = self.n_balls
        F = (forces_size.reshape(1, n, n) *
                np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
        return F


    def sample_trajectory(self, T=10000, sample_freq=10):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0


        total_num_edges = int(0.5*self.n_balls*(self.n_balls-1))
        
        num_spring_edges = random.randint(0, total_num_edges)
        edges = [0 for i in range(total_num_edges)]
        for i in range(num_spring_edges):
            edges[i] = 1
        random.shuffle(edges)
        spring_edges = np.zeros( (self.n_balls,self.n_balls) )
        spring_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        spring_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)

        charges = [0 for i in range(self.n_balls)]
        n_c = random.randint(1, self.n_balls)       # choose a random number of charges, 1 to 5
        for i in range(n_c):
            charges[i] = 1
        random.shuffle(charges)
        charges = np.expand_dims(np.array(charges),-1)
        charge_edges = charges.dot(charges.transpose()).astype('float')

        num_fspring_edges = random.randint(0, total_num_edges)
        edges = [0 for i in range(total_num_edges)]
        for i in range(num_fspring_edges):
            edges[i] = 1
        random.shuffle(edges)
        fspring_edges = np.zeros( (self.n_balls,self.n_balls) )
        fspring_edges[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        fspring_edges.T[ np.triu_indices(self.n_balls,1) ] = np.array(edges)
        
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std     # randn samples from a unit normal dist.
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore', invalid='ignore'):

            spring_forces_size = - self.spring_interaction_strength * spring_edges
            #np.fill_diagonal(spring_forces_size, 0)  # self forces are zero (fixes division by zero)

            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            charge_forces_size = - self.charge_interaction_strength * charge_edges / l2_dist_power3
            np.fill_diagonal(charge_forces_size, 0)  # self forces are zero (fixes division by zero)
            #assert (np.abs(charge_forces_size[diag_mask]).min() > 1e-10)

            l2_dist_powerhalf = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 1. / 2.)
            fspring_forces_size = - self.fspring_interaction_strength * fspring_edges * \
                                    ( 1 - self.fspring_length / l2_dist_powerhalf )
            np.fill_diagonal(fspring_forces_size, 0)  # self forces are zero (fixes division by zero)


            F_s = self._get_force(spring_forces_size, loc_next)
            F_c = self._get_force(charge_forces_size, loc_next)
            F_f = self._get_force(fspring_forces_size, loc_next)
            F = F_s + F_c + F_f

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F

            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)
                # size of forces up to a 1/|r| factor
                # since I later multiply by an unnormalized r vector
                charge_forces_size = self.charge_interaction_strength * charge_edges / l2_dist_power3
                np.fill_diagonal(charge_forces_size, 0)  # self forces are zero (fixes division by zero)

                l2_dist_powerhalf = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 1. / 2.)
                fspring_forces_size = - self.fspring_interaction_strength * fspring_edges *  \
                                        ( 1 - self.fspring_length / l2_dist_powerhalf )
                np.fill_diagonal(fspring_forces_size, 0)  # self forces are zero (fixes division by zero)

                F_s = self._get_force(spring_forces_size, loc_next)
                F_c = self._get_force(charge_forces_size, loc_next)
                F_f = self._get_force(fspring_forces_size, loc_next)
                F = F_s + F_c + F_f

                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            np.fill_diagonal(charge_edges, 0)
            edges = np.concatenate( (np.expand_dims(spring_edges,0), 
                                     np.expand_dims(charge_edges,0), 
                                     np.expand_dims(fspring_edges,0)), axis=0)
            
            return loc, vel, edges