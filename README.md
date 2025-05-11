# KalmanFilter
## Overview
This repository demonstrates a simple implementation of a Kalman Filter for state estimation in dynamic systems. The project serves as both an educational resource and a starting point for more advanced filtering applications.

## Demonstration Video

Below is a demonstration of the Kalman Filter tracking output. Click the video below or use your browser's video controls to play it.

https://github.com/user-attachments/assets/4eaa35b3-e4b1-46c3-867d-7fb266979771

## Bayes Theorem

Bayes Theorem provides a way to update beliefs in light of new evidence. The formula is expressed as:

$$
P(A|B) = \frac{P(B|A) \* P(A)}{P(B)}
$$

Where:
- P(A) is the prior probability of event A ().
- P(B|A) is the likelihood of event B given event A.
- P(B) is the marginal probability of event B.
- P(A|B) is the posterior probability of event A given event B.

---

## Kalman Filter Matrices and Equations

### Prediction Step


### Update Step


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
