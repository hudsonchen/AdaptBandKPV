from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from src.data.data_class import TrainDataSet, TestDataSet


class KernelIVAdaptBandModel:

    def __init__(self, X_train: np.ndarray, O_train: np.ndarray, Y_train: np.ndarray,
                 J: np.ndarray, 
                 sigmaX: float, sigmaO: float, lambda2: float):
        """

        Parameters
        ----------
        X_train: np.ndarray[n_stage1, dim_treatment]
            data for treatment
        sigma: gauss parameter
        """
        self.X_train = X_train
        self.O_train = O_train
        self.Y_train = Y_train
        self.J = J
        self.sigmaX = sigmaX
        self.sigmaO = sigmaO
        self.lambda2 = lambda2

    @staticmethod
    def cal_gauss(XA, XB, sigma: float = 1):
        """
        Returns gaussian kernel matrix
        Parameters
        ----------
        XA : np.ndarray[n_data1, n_dim]
        XB : np.ndarray[n_data2, n_dim]
        sigma : float

        Returns
        -------
        mat: np.ndarray[n_data1, n_data2]
        """
        dist_mat = cdist(XA, XB, "sqeuclidean")
        return np.exp(-dist_mat / sigma)

    def predict(self, treatment: np.ndarray, covariate: np.ndarray):
        N = self.O_train.shape[0]
        X = np.array(treatment, copy=True)
        O = np.array(covariate, copy=True)
        Kx = self.cal_gauss(X, self.X_train, self.sigmaX) # n_test \times m
        Ko = self.cal_gauss(O, self.O_train, self.sigmaO) # n_test \times n
        KX1X1 = self.cal_gauss(self.X_train, self.X_train, self.sigmaX)
        KO2O2 = self.cal_gauss(self.O_train, self.O_train, self.sigmaO)
        part1 = np.multiply(Kx.dot(self.J), Ko)
        part2 = np.linalg.solve(np.multiply(self.J.T.dot(KX1X1.dot(self.J)), KO2O2) + N * self.lambda2 * np.eye(N), self.Y_train)
        pred = part1.dot(part2)
        return pred

    def evaluate(self, test_data: TestDataSet):
        pred = self.predict(test_data.treatment, test_data.covariate)
        return np.mean((test_data.structural - pred)**2)

