from numpy import eye, zeros, isscalar, dot
import numpy as np
from filterpy.common import pretty_str, reshape_z
from copy import deepcopy

class CustomKalmanFilter():
    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros(dim_x)             # state vector
        self.P = eye(dim_x) * 1000        # initial uncertainty
        self.Q = zeros((dim_x, dim_x))    # process noise covariance
        self.F = zeros((dim_x, dim_x))    # state transition matrix
        self.H = zeros((dim_z, dim_x))    # measurement function
        self.R = zeros((dim_z, dim_z))    # measurement noise covariance
        self.M = zeros((dim_x, dim_u))    # control input matrix
        self.z = zeros(dim_z)             # measurement vector

        # gain and residual are computed during the innovation step
        self.K = zeros((dim_x, dim_z))    # Kalman gain
        self.y = zeros(dim_z)             # residual (innovation)
        self.S = zeros((dim_z, dim_z))    # residual covariance
        self.SI = zeros((dim_z, dim_z))   # inverse of residual covariance

        # identity matrix. Do not alter this.
        self._I = eye(dim_x)

        # these will always be a copy of x, P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x, P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=None, B=None, F=None, Q=None):
        """Predict the next state and covariance."""
        if F is not None:
            self.F = F
        if Q is not None:
            self.Q = Q

        if u is not None and B is not None:
            self.x = dot(self.F, self.x) + dot(B, u)
        else:
            self.x = dot(self.F, self.x)

        self.P = dot(dot(self.F, self.P), self.F.T) + self.Q

        # Save prior state
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """Update the state with a new measurement."""
        if R is not None:
            self.R = R
        if H is not None:
            self.H = H

        self.y = z - dot(self.H, self.x)
        self.S = dot(dot(self.H, self.P), self.H.T) + self.R
        self.SI = np.linalg.inv(self.S)
        self.K = dot(dot(self.P, self.H.T), self.SI)

        self.x = self.x + dot(self.K, self.y)
        self.P = dot(self._I - dot(self.K, self.H), self.P)

        # Save posterior state
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

def main():
    kf = CustomKalmanFilter(4,2)
    kf.x = np.array([[0, 0, 0, 0]]).T
    kf.P = np.eye(4) * 500.
    dt = 1.0
    kf.F = np.array([[1, dt, 0,  0],
                    [0,  1, 0,  0],
                    [0,  0, 1, dt],
                    [0,  0, 0,  1]])

    kf.H = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])

    from scipy.linalg import block_diag
    from filterpy.common import Q_discrete_white_noise

    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    kf.Q = block_diag(q, q)
    kf.R = np.array([[5., 0],
                        [0, 5]])

    zs = [[1, 1], [2,2], [3,3], [4,4]]
    for z in zs:
        kf.predict()
        print("Prediction...")
        print("Predicted X")
        print(kf.x_prior)
        print("Predicted P")
        print(kf.P_prior)
        kf.update(z)
        print("Measurement... z = ")
        print(z)
        print("Updated X")
        print(kf.x_post)
        print("Updated P")
        print(kf.P_post)

if __name__ == "__main__":
    main()