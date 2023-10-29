import numpy as np


class KalmanFilter(object):
    # x0 - initial guess of the state vector
    # P0 - initial guess of the covariance matrix of the state estimation error
    # A,B,C - system matrices describing the system model
    # Q - covariance matrix of the process noise
    # R - covariance matrix of the measurement noise
    def __init__(self, x0, P0, A, B, C, Q, R):
        # initialize vectors and matrices
        self.x0 = x0
        self.P0 = P0
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        ####### members for debuggin #######
        self.currentTimeStep = 0
        self.estimates_aposteriori = []
        self.estimates_aposteriori.append(x0)
        self.estimates_apriori = []
        self.estimationErrorCovarianceMatricesAposteriori = []
        self.estimationErrorCovarianceMatricesAposteriori.append(P0)
        self.estimationErrorCovarianceMatricesApriori = []
        self.gainMatrices = []
        self.errors = []

    def set_matrices(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

    # propagate x_{k-1}^{+} to compute x_{k}^{-}
    # this function also propagates P_{k-1}^{+} to compute P_{k}^{-}
    def predict(self, input_value, prev_xk_plus, prev_pk_plus):
        xk_minus = self.A * prev_xk_plus + self.B * input_value
        pk_minus = self.A * prev_pk_plus * (self.A.T) + self.Q
        return xk_minus, pk_minus

    # predict for k steps ahead (usefull for reducing latency in real systems)
    def predict_multi(self, inputValue, steps, prev_xk_plus):
        xk_minus = self.A * prev_xk_plus + self.B * inputValue
        for _ in range(steps - 1):
            xk_minus = self.A * xk_minus + self.B * inputValue
        return xk_minus

    # for debugging purposes, perdicts and stores the states as members of the class
    def predict_and_store(self, input_value):
        xk_minus = (
            self.A * self.estimates_aposteriori[self.currentTimeStep]
            + self.B * input_value
        )
        pk_minus = (
            self.A
            * self.estimationErrorCovarianceMatricesAposteriori[self.currentTimeStep]
            * (self.A.T)
            + self.Q
        )

        self.estimates_apriori.append(xk_minus)
        self.estimationErrorCovarianceMatricesApriori.append(pk_minus)

        self.currentTimeStep = self.currentTimeStep + 1

    # given a new measurement, perform the update state to fuse the measurement with the prediction
    def update(self, current_measurement, xk_minus, pk_minus):
        # gain matrix
        Kk = (
            pk_minus
            * (self.C.T)
            * np.linalg.inv(self.R + self.C * pk_minus * (self.C.T))
        )
        # prediction error
        error_k = current_measurement - self.C * xk_minus
        # a posteriori estimate
        xk_plus = xk_minus + Kk * error_k
        # a posteriori matrix update
        IminusKkC = np.matrix(np.eye(self.x0.shape[0])) - Kk * self.C
        pk_plus = IminusKkC * pk_minus * (IminusKkC.T) + Kk * (self.R) * (Kk.T)
        # return updated state
        return xk_plus, pk_plus

    # perform the update step and also stores the states as members of the class
    def update_and_store(self, current_measurement):
        Kk = (
            self.estimationErrorCovarianceMatricesApriori[self.currentTimeStep - 1]
            * (self.C.T)
            * np.linalg.inv(
                self.R
                + self.C
                * self.estimationErrorCovarianceMatricesApriori[
                    self.currentTimeStep - 1
                ]
                * (self.C.T)
            )
        )

        # prediction error
        error_k = (
            current_measurement
            - self.C * self.estimates_apriori[self.currentTimeStep - 1]
        )
        # a posteriori estimate
        xk_plus = self.estimates_apriori[self.currentTimeStep - 1] + Kk * error_k

        # a posteriori matrix update
        IminusKkC = np.matrix(np.eye(self.x0.shape[0])) - Kk * self.C
        Pk_plus = IminusKkC * self.estimationErrorCovarianceMatricesApriori[
            self.currentTimeStep - 1
        ] * (IminusKkC.T) + Kk * (self.R) * (Kk.T)

        # update the lists that store the vectors and matrices
        self.gainMatrices.append(Kk)
        self.errors.append(error_k)
        self.estimates_aposteriori.append(xk_plus)
        self.estimationErrorCovarianceMatricesAposteriori.append(Pk_plus)
