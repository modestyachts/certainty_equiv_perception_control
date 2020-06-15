"""Classes for defining and running system interconnections."""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from observers import *

class DoubleIntegratorPlant():
    """Implementation of double integrator dynamics."""

    def __init__(self, ndims=1, dt=0.1, decay=(1, 1), sw=0.1, x0=None):
        # constructing dynamics
        A1 = np.array([[decay[0], dt], [0, decay[1]]])
        B1 = np.array([[0], [1]])
        self.A = scipy.linalg.block_diag(*[A1]*ndims)
        self.B = scipy.linalg.block_diag(*[B1]*ndims)

        # process noise
        self.Bw = sw * np.eye(self.A.shape[0])

        # initializing state
        self.x = np.zeros(self.A.shape[0]) if x0 is None else x0
        self.t = 0
        self._w = None
        self._v = None

    def reset(self, x0=None):
        """Reset to x0."""
        self.x = np.zeros(self.A.shape[0]) if x0 is None else x0
        self.t = 0

    def step(self, u):
        """Advance the system."""
        self._w = self.Bw @ np.clip(np.random.normal(scale=1, size=self.Bw.shape[1]), -1, 1)
        self.x = self.A @ self.x + self.B @ u + self._w
        self.t += 1

class PeriodicTrackingController():
    """Static reference tracking controller.
            u_k = K * (xh_k - xr_k)
            xh_{k+1} = A xh_k + B u_k + L * (y - C xh_k)

    Parameters
    ----------
    trajectory : list
        List of reference trajectory to follow.
    A, B : numpy arrays
        Dynamics matrices.
    C : numpy array
        Measurement matrix.
    K, L : numpy arrays
        Feedback and estimation parameters.
    su : float
        Additional noise to add to inputs.
    x0 : numpy array
        Initial state estimate.
    perception : function, optional
        If included, will pass measurements through before using.

    """
    def __init__(self, trajectory, A, B, C, K, L, su=0., perception=None,
                 x0=None):
        n, m = B.shape
        self.t = 0
        self.x = np.zeros(n) if x0 is None else x0
        self.K = K
        self.L = L
        self.A = A
        self.B = B
        self.C = C
        self.u = np.zeros(m)
        self.period = len(trajectory)
        self.ref = trajectory
        self.su = su
        self.perception = perception
        if self.perception is not None:
            self.ys = []

    def input(self, y):
        """Design input and update internal state based on y."""
        self.u = self.K @ (self.x - self.ref[self.t % self.period])
        self.u += np.clip(np.random.normal(scale=self.su, size=self.u.shape), -1, 1)
        self.t += 1
        if self.perception is not None:
            y = self.perception([y])
            y = y[0]
            self.ys.append(y)
        self.update_state_estimate(y)
        return self.u

    def update_state_estimate(self, y):
        """Update internal state."""
        self.x = self.A @ self.x + self.B @ self.u + self.L @ (y - self.C @ self.x)

class PeriodicOLControl():
    """Open loop control signals."""
    def __init__(self, freq=[0.01], amp=[1], funtype=['sin'], su=0.5):
        fun = []
        for t in funtype:
            if t == 'sin':
                fun.append(np.sin)
            elif t == 'cos':
                fun.append(np.cos)
            else:
                assert False, "unrecognized function type {}".format(t)

        self.control_fun = (lambda t: [a * f(2*np.pi*fr*t)
                                       for fr, a, f in zip(freq, amp, fun)])
        self.t = 0
        self.su = su

    def input(self, y):
        """Select input based on time."""
        u = np.array(self.control_fun(self.t))
        noise = np.clip(np.random.normal(scale=self.su, size=u.shape), -1, 1)
        self.t += 1
        return u + noise

class Interconnection():
    """Interconnection puts together controller, dynamical system, and observer.

    Parameters
    ----------
    plant : object
        Dynamical system.
    get_observation : function
        Returns observation based on state.
    controller : object
        Closed-loop controller.
    get_observation_for_controller : function, optional
        Additional observation method to be used by controller.

    """

    def __init__(self, plant, get_observation, controller,
                 get_observation_for_controller=None):
        self.plant = plant
        self.get_observation = get_observation
        self.get_observation_c = get_observation_for_controller
        self.controller = controller

        self.xs = [plant.x]
        self.us = []
        self.zs = [self.get_observation(plant.x)]
        if self.get_observation_c is not None:
            self.zs_c = [self.get_observation_c(plant.x)]
        else:
            self.zs_c = self.zs

    def step(self):
        """Advance the system."""
        u = self.controller.input(self.zs_c[-1])
        self.plant.step(u)
        self.us.append(u)
        self.xs.append(self.plant.x)
        self.zs.append(self.get_observation(self.plant.x))
        if self.get_observation_c is not None:
            self.zs_c.append(self.get_observation_c(self.plant.x))

    def plot_trajectory(self, axs):
        """Plot trajectory."""
        xs = np.array(self.xs)
        us = np.array(self.us)
        for i in range(xs.shape[1]):
            axs[0].plot(xs[:, i], alpha=0.7, label='x{}'.format(i+1))
        axs[0].legend()
        axs[0].set_title('States')
        for i in range(us.shape[1]):
            axs[1].plot(us[:, i], alpha=0.7, label='u{}'.format(i+1))
        axs[1].legend()
        axs[1].set_title('Inputs')

    def plot_observations(self, ax):
        """Plot observations."""
        zs = np.array(self.zs)
        zs = zs.reshape(len(self.zs), -1)
        im = ax.imshow(zs.T, aspect='auto')
        ax.set_xlabel('time')
        ax.set_title('observation values over time')
        plt.colorbar(im)

def optimal_k(A, B, R, P):
    """Compute optimal static feedback controller based on Riccati solution."""
    return scipy.linalg.inv(B.T.dot(P).dot(B) + R).dot(B.T.dot(P).dot(A))

def lqr_inf_horizon(A, B, Q, R):
    """Compute optimal infinite horizon LQR controller."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = optimal_k(A, B, R, P)
    return K, P

def kalman_gain(C, V, S):
    """Compute optimal static gain based on Riccati solution."""
    return S.dot(C.T).dot(scipy.linalg.inv(C.dot(S).dot(C.T) + V))

def lqg_inf_horizon(A, C, W, V):
    """Compute optimal infinite horizon Kalman filter."""
    S = scipy.linalg.solve_discrete_are(A.T, C.T, W, V)
    L = kalman_gain(C, V, S)
    return L, S
