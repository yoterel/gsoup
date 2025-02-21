# shows some use cases with the Kalman filter class

import numpy as np
import matplotlib.pyplot as plt
from gsoup.track import KalmanFilter
from scipy import signal

# discretization step of simulation
dt = 0.1
# initial values for the simulation
initialPosition = 10
initialVelocity = -5
acceleration = 0.5
# measurement noise standard deviation
noiseStd = 0.01
# number of discretization time steps
numberTimeSteps = 1000
# for a more elaborate use case: number of lookahead future prediction steps
k = 10
# for a more elaborate use case: missing measurements stride
mask_interval = 10
gt_motion = "sawtooth"  # sine, constant_acc, sawtooth
# assume that for short durations, the motion is constant acceleration
A = np.matrix(
    [
        [1, dt, 0.5 * (dt**2)],
        [0, 1, dt],
        [0, 0, 1],
    ]
)  # state-change/transition matrix
B = np.matrix([[0], [0], [0]])  # control matrix
C = np.matrix([[1, 0, 0]])  # measurement/observation matrix
Q = 2500 * np.matrix(np.eye(3))  # process noise covariance matrix
R = (noiseStd**2) * np.matrix([[1]])  # measurement noise covariance matrix

x0 = np.matrix([[0], [0], [0]])  # guess of the initial estimate
P0 = 1 * np.matrix(np.eye(3))  # guess of the initial estimate error covariance matrix

############################################################################
############################################################################
############################################################################

timeVector = np.linspace(0, (numberTimeSteps - 1) * dt, numberTimeSteps)
mask = np.zeros(np.size(timeVector))
mask[::mask_interval] = 1
steps = np.arange(np.size(timeVector))
position = np.zeros(np.size(timeVector))
velocity = np.zeros(np.size(timeVector))
acceleration = np.zeros(np.size(timeVector))

# simulate the system behavior
for i in np.arange(np.size(timeVector)):
    if gt_motion == "sawtooth":
        amp = 5
        # lets use a saw tooth wave
        position[i] = amp * signal.sawtooth(i / 20, 0.5)
        # its derivative
        velocity[i] = 2 * np.pi * 5 / 20 * signal.square(i / 20, 0.5)
        # formally should take value 1, -1 in peaks...lets just use 0
        acceleration[i] = 0
    elif gt_motion == "sine":
        # sine wave
        position[i] = 5 * np.sin(0.1 * timeVector[i])
        velocity[i] = 0.1 * 5 * np.cos(0.1 * timeVector[i])
        acceleration[i] = 0.1 * 0.1 * 5 * -np.sin(0.1 * timeVector[i])
    elif gt_motion == "constant_acc":
        # constant acc
        position[i] = (
            initialPosition
            + initialVelocity * timeVector[i]
            + (acceleration * timeVector[i] ** 2) / 2
        )
        velocity[i] = initialVelocity + acceleration * timeVector[i]
        acceleration[i] = acceleration
if gt_motion == "sawtooth":
    acceleration = np.concatenate((np.diff(velocity), np.zeros(1)))
# add the measurement noise
positionNoisy = position + np.random.normal(0, noiseStd, size=np.size(timeVector))
# verify the position vector by plotting the results
# plotStep = numberTimeSteps
# plt.figure(figsize=(10, 10))
# plt.plot(steps[0:plotStep], position[0:plotStep], linewidth=4, label="Ideal position")
# plt.plot(steps[0:plotStep], positionNoisy[0:plotStep], "r", label="Observed position")
# plt.xlabel("time")
# plt.ylabel("position")
# plt.legend()
# plt.savefig("data.png", dpi=300)
# plt.show()


# create a Kalman filter object
KalmanFilterObject = KalmanFilter(x0, P0, A, B, C, Q, R)
inputValue = np.matrix([[0]])
xk_plus, pk_plus = x0, P0
masked_xk_plus, masked_pk_plus = x0, P0
predictions = []
estimates_k = []
estimates_masked = []
estimates_masked.append(x0)
estimates = []
estimates.append(x0)

for i in range(k):
    estimates_k.append(np.matrix([[0], [0], [0]]))
# simulate online prediction
for j in np.arange(np.size(timeVector)):
    print("Time step: {}".format(j))
    xk_minus_k = KalmanFilterObject.predict_multi(inputValue, k, xk_plus)
    estimates_k.append(xk_minus_k)
    xk_minus, pk_minus = KalmanFilterObject.predict(inputValue, xk_plus, pk_plus)
    predictions.append(xk_minus)
    masked_xk_minus, masked_pk_minus = KalmanFilterObject.predict(
        inputValue, masked_xk_plus, masked_pk_plus
    )
    if mask[j] == 0:
        masked_xk_plus = masked_xk_minus
        masked_pk_plus = masked_pk_minus
    else:
        masked_xk_plus, masked_pk_plus = KalmanFilterObject.update(
            positionNoisy[j], masked_xk_minus, masked_pk_minus
        )
    xk_plus, pk_plus = KalmanFilterObject.update(positionNoisy[j], xk_minus, pk_minus)
    estimates.append(xk_plus)
    estimates_masked.append(masked_xk_plus)


# extract the state estimates in order to plot the results
estimate1 = []
estimate2 = []
estimate3 = []
# k steps ahead
estimate1_k = []
estimate2_k = []
estimate3_k = []
# masked
estimate1_masked = []
estimate2_masked = []
estimate3_masked = []


for j in np.arange(np.size(timeVector)):
    # estimates n+1
    estimate1.append(estimates[j][0, 0])
    estimate2.append(estimates[j][1, 0])
    estimate3.append(estimates[j][2, 0])
    # estimates n+k
    estimate1_k.append(estimates_k[j][0, 0])
    estimate2_k.append(estimates_k[j][1, 0])
    estimate3_k.append(estimates_k[j][2, 0])
    # estimates n+k masked
    estimate1_masked.append(estimates_masked[j][0, 0])
    estimate2_masked.append(estimates_masked[j][1, 0])
    estimate3_masked.append(estimates_masked[j][2, 0])

# create vectors corresponding to the true values in order to plot the results
estimate1true = position
estimate2true = velocity
# estimate3true = acceleration * np.ones(np.size(timeVector))
estimate3true = acceleration


# plot the results

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

ax[0].scatter(
    steps,
    estimate1_k,
    color="orange",
    s=3,
    # linestyle="-",
    label="k={} lookahead steps".format(k),
)
ax[0].scatter(
    steps,
    estimate1,
    color="blue",
    s=3,
    # linestyle="-",
    label="k=1 lookahead steps",
)
ax[0].scatter(
    steps,
    estimate1_masked,
    color="green",
    s=3,
    # linestyle="-",
    label="{}/{} missing measurements".format(mask_interval - 1, mask_interval),
)
ax[0].scatter(
    steps,
    estimate1true,
    color="red",
    s=3,
    # linestyle="-",
    label="ground truth",
)

ax[0].set_xlabel("Discrete-time steps k", fontsize=14)
ax[0].set_ylabel("Position", fontsize=14)
ax[0].tick_params(axis="both", labelsize=12)
ax[0].grid()
ax[0].legend(fontsize=14)
ax[1].plot(
    steps,
    estimate2_k,
    color="orange",
    linestyle="-",
    label="k={} lookahead steps".format(k),
)
ax[1].plot(
    steps,
    estimate2,
    color="blue",
    linestyle="-",
    label="k=1 lookahead steps",
)
ax[1].plot(
    steps,
    estimate2_masked,
    color="green",
    linestyle="-",
    label="{}/{} missing measurements".format(mask_interval - 1, mask_interval),
)
ax[1].plot(
    steps,
    estimate2true,
    color="red",
    linestyle="-",
    label="ground truth",
)

ax[1].set_xlabel("Discrete-time steps k", fontsize=14)
ax[1].set_ylabel("Velocity", fontsize=14)
ax[1].tick_params(axis="both", labelsize=12)
ax[1].grid()
ax[1].legend(fontsize=14)
ax[2].plot(
    steps,
    estimate3_k,
    color="orange",
    linestyle="-",
    label="k={} lookahead steps".format(k),
)
ax[2].plot(
    steps,
    estimate3,
    color="blue",
    linestyle="-",
    label="k=1 lookahead steps",
)
ax[2].plot(
    steps,
    estimate3_masked,
    color="green",
    linestyle="-",
    label="{}/{} missing measurements".format(mask_interval - 1, mask_interval),
)
ax[2].plot(
    steps,
    estimate3true,
    color="red",
    linestyle="-",
    label="ground truth",
)

ax[2].set_xlabel("Discrete-time steps k", fontsize=14)
ax[2].set_ylabel("Acceleration", fontsize=14)
ax[2].tick_params(axis="both", labelsize=12)
ax[2].grid()
ax[2].legend(fontsize=14)
fig.savefig("plots.png", dpi=600)
