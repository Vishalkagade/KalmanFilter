# KalmanFilter
## Overview
This repository demonstrates a simple implementation of a Kalman Filter for state estimation in dynamic systems. The project serves as both an educational resource and a starting point for more advanced filtering applications.

## Bayes Theorem

Bayes Theorem provides a way to update beliefs in light of new evidence. The formula is expressed as:

$$
P(A|B) = \frac{P(B|A) \, P(A)}{P(B)}
$$

Where:
- P(A) is the prior probability of event A ().
- P(B|A) is the likelihood of event B given event A.
- P(B) is the marginal probability of event B.
- P(A|B) is the posterior probability of event A given event B.

---

## Kalman Filter Matrices and Equations

### Prediction Step
- State prediction:  
    $$
    \hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_k
    $$
- Covariance prediction:  
    $$
    P_{k|k-1} = A P_{k-1|k-1} A^T + Q
    $$

### Update Step
- Kalman Gain:  
    $$
    K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
    $$
- State update:  
    $$
    \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})
    $$
- Covariance update:  
    $$
    P_{k|k} = (I - K_k H) P_{k|k-1}
    $$

Where:
- A is the state transition matrix.
- B is the control input matrix.
- uₖ is the control input at step k.
- Q is the process noise covariance.
- H is the observation matrix.
- R is the measurement noise covariance.
- zₖ is the measurement at step k.


## Features
- Prediction and update steps for tracking dynamic states
- Handling measurement and process noise
- Used euclidian distance with Hungerian method for identity matching

## Installation
Clone the repository and install the required dependencies:
```
git clone https://github.com/yourusername/KalmanFilter.git
cd KalmanFilter
pip install -r requirements.txt
```

## Usage
Run the main script to see the Kalman Filter in action:
```
python KF_Visualize_tracking.py
```
