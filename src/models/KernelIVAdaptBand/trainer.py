from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


from src.data import generate_train_data, generate_test_data
from src.data.data_class import TrainDataSet, TrainDataSetTorch, TestDataSet, TestDataSetTorch
from src.models.KernelIVAdaptBand.model import KernelIVAdaptBandModel

logger = logging.getLogger()


def get_median(X) -> float:
    dist_mat = cdist(X, X, "sqeuclidean")
    res: float = np.median(dist_mat)
    return res


class KernelIVAdaptBandTrainer:

    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any],
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        self.data_config = data_configs

        self.lambda1 = train_params["lam1"]
        self.lambda2 = train_params["lam2"]
        self.eta = train_params["eta"]
        self.split_ratio = train_params["split_ratio"]

    def split_train_data(self, train_data: TrainDataSet, test_data: TrainDataSet, normalize: bool = True):
        if normalize:
            mean_O = np.mean(train_data.covariate, axis=0)
            std_O = np.std(train_data.covariate, axis=0)
            mean_X = np.mean(train_data.treatment, axis=0)
            std_X = np.std(train_data.treatment, axis=0)
            mean_Z = np.mean(train_data.instrumental, axis=0)
            std_Z = np.std(train_data.instrumental, axis=0)

            train_data = TrainDataSet(
                treatment=(train_data.treatment - mean_X) / std_X,
                instrumental=(train_data.instrumental - mean_Z) / std_Z,
                covariate=(train_data.covariate - mean_O) / std_O,
                outcome=train_data.outcome,
                structural=train_data.structural
            )

            test_data = TestDataSet(
                treatment=(test_data.treatment - mean_X) / std_X,
                covariate=(test_data.covariate - mean_O) / std_O,
                structural=test_data.structural
            )
        n_data = train_data[0].shape[0]
        idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=self.split_ratio)

        def get_data(data, idx):
            return data[idx] if data is not None else None

        train_1st_data = TrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
        train_2nd_data = TrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])

        return train_1st_data, train_2nd_data, test_data

    def train(self, rand_seed: int = 42, verbose: int = 0) -> float:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """
        train_data = generate_train_data(rand_seed=rand_seed, **self.data_config)
        test_data = generate_test_data(**self.data_config)
        normalize = True
        train_1st_data, train_2nd_data, test_data = self.split_train_data(train_data, test_data, normalize)

        # get stage1 data
        X1 = train_1st_data.treatment
        O1 = train_1st_data.covariate
        Z1 = train_1st_data.instrumental
        Y1 = train_1st_data.outcome
        M = X1.shape[0]

        # get stage2 data
        X2 = train_2nd_data.treatment
        O2 = train_2nd_data.covariate
        Z2 = train_2nd_data.instrumental
        Y2 = train_2nd_data.outcome
        N = X2.shape[0]
        d_X = X2.shape[1]
        d_O = O2.shape[1]
        s_X = 1.
        s_O = 1.

        if verbose > 0:
            logger.info("start stage1 and stage 2")

        gammaZ = get_median(Z1)
        gammaO_stage1 = get_median(O1)
        KZ1Z1 = KernelIVAdaptBandModel.cal_gauss(Z1, Z1, gammaZ)
        KZ1Z2 = KernelIVAdaptBandModel.cal_gauss(Z1, Z2, gammaZ)
        KO1O2_stage1 = KernelIVAdaptBandModel.cal_gauss(O1, O2, gammaO_stage1)
        KO1O1_stage1 = KernelIVAdaptBandModel.cal_gauss(O1, O1, gammaO_stage1)
        if isinstance(self.eta, list):
            self.eta = 3 ** np.linspace(self.eta[0], self.eta[1], 10)
        else:
            self.eta = 3 ** np.array(self.eta)
        eta_dict = {eta: {'Y_hat': 0, 'lambda2': 0} for eta in self.eta}

        for eta in self.eta:
            frac_X = (1 / d_X) / (1 + 2 * (s_X / d_X + eta) + d_O / s_O * (s_X / d_X + eta))
            frac_O = (1 / s_O * (s_X / d_X)) / (1 + 2 * (s_X / d_X + eta) + d_O / s_O * (s_X / d_X + eta))
            gammaX = 5 * N ** (-frac_X)
            gammaO_stage2 = 5 * N ** (-frac_O)
            # gammaX = get_median(X1)
            # gammaO_stage2 = get_median(O1)
            KX1X1 = KernelIVAdaptBandModel.cal_gauss(X1, X1, gammaX)
            KX1X2 = KernelIVAdaptBandModel.cal_gauss(X1, X2, gammaX)
            KO1O2_stage2 = KernelIVAdaptBandModel.cal_gauss(O1, O2, gammaO_stage2)
            KO2O2_stage2 = KernelIVAdaptBandModel.cal_gauss(O2, O2, gammaO_stage2)

            if isinstance(self.lambda1, list):
                lambda1 = 10 ** np.linspace(self.lambda1[0], self.lambda1[1], 50)
                J = self.stage1_tuning(KX1X1, KX1X2, KZ1Z1, KO1O1_stage1, KZ1Z2, KO1O2_stage1, lambda1)
            else:
                J = np.linalg.solve(np.multiply(KZ1Z1, KO1O1_stage1) + M * self.lambda1 * np.eye(M), np.multiply(KZ1Z2, KO1O2_stage1))
            
            if isinstance(self.lambda2, list):
                lambda2 = 10 ** np.linspace(self.lambda2[0], self.lambda2[1], 50)
                feature_2 = np.multiply(KX1X1.dot(J), KO1O2_stage2)
                lambda2, Y_hat = self.stage2_tuning(feature_2, J, KX1X1, KO2O2_stage2, Y1, Y2, lambda2)
            else:
                lambda2 = 10 ** self.lambda2
                feature_2 = np.multiply(KX1X1.dot(J), KO1O2_stage2)
                part2 = np.linalg.solve(np.multiply(J.T.dot(KX1X1.dot(J)), KO2O2_stage2) + N * lambda2 * np.eye(N), Y2)
                Y_hat = feature_2.dot(part2)
            # logger.info(f"Yhat: {Y_hat[:5].T}")
            # logger.info(f"Y1: {Y1[:5].T}")
            eta_dict[eta]['Y_hat'] = Y_hat
            eta_dict[eta]['lambda2'] = lambda2
            eta_dict[eta]['gammaX'] = gammaX
            eta_dict[eta]['gammaO'] = gammaO_stage2
            eta_dict[eta]['score'] = np.linalg.norm(Y1 - eta_dict[eta]['Y_hat'])

            logger.info(f"eta: {eta}, gammaX: {gammaX}, gammaO: {gammaO_stage2}, score: {np.linalg.norm(Y1 - eta_dict[eta]['Y_hat'])}")
        # Tune eta 
        eta = min(eta_dict, key=lambda idx: eta_dict[idx]['score'])
        lambda2 = eta_dict[eta]['lambda2']
        frac_X = (1 / d_X) / (1 + 2 * (s_X / d_X + eta) + d_O / s_O * (s_X / d_X + eta))
        frac_O = (1 / s_O * (s_X / d_X)) / (1 + 2 * (s_X / d_X + eta) + d_O / s_O * (s_X / d_X + eta))
        gammaX = 5 * N ** (-frac_X) 
        gammaO_stage2 = 5 * N ** (-frac_O)
        # gammaX = eta_dict[eta]['gammaX']
        # gammaO_stage2 = eta_dict[eta]['gammaO']
        if verbose > 0:
            logger.info("end stage1 and stage 2")
        logger.info(f"eta:{eta}, gammaX:{gammaX}, gammaO:{gammaO_stage2}, lambda2:{lambda2}")
        logger.info(f"median X1: {get_median(X1)}, median O1: {get_median(O1)}")
        mdl = KernelIVAdaptBandModel(X1, O2, Y2, J, gammaX, gammaO_stage2, lambda2)
        return mdl.evaluate(test_data)

    def stage1_tuning(self, KX1X1, KX1X2, KZ1Z1, KO1O1, KZ1Z2, KO1O2, lambda1):
        M = KX1X1.shape[0]
        KZO1ZO1 = np.multiply(KZ1Z1, KO1O1)
        KZO1ZO2 = np.multiply(KZ1Z2, KO1O2)
        J_list = [np.linalg.solve(KZO1ZO1 + M * lam1 * np.eye(M), KZO1ZO2) for lam1 in lambda1]
        score = [np.trace(J.T.dot(KX1X1.dot(J)) - 2 * KX1X2.T.dot(J)) for J in J_list]
        return J_list[np.argmin(score)]

    def stage2_tuning(self, feature_2, J, KX1X1, KO2O2, Y1, Y2, lambda2):
        N = feature_2.shape[1]
        alpha_list = [np.linalg.solve(np.multiply(J.T.dot(KX1X1.dot(J)), KO2O2) + N * lam2 * np.eye(N), Y2) for lam2 in lambda2]
        score = [np.linalg.norm(Y1 - feature_2.dot(alpha)) for alpha in alpha_list]
        Y_hat = feature_2.dot(alpha_list[np.argmin(score)])
        return lambda2[np.argmin(score)], Y_hat