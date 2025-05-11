from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt

def TwoDimensionsKF(R_std, Q_std, dt):
    """ Create first order Kalman filter. 
    Specify R and Q as floats."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.P = np.eye(4)*500 # covariance   
    kf.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    kf.Q = block_diag(q, q)
    kf.F = np.array([[1, dt, 0,  0],
                    [0,  1, 0,  0],
                    [0,  0, 1, dt],
                    [0,  0, 0,  1]])

    kf.H = np.array([[1, 0, 0, 0],  # [X,X_dot,Y,Y_dot]
                    [0, 0, 1, 0]])
    return kf


