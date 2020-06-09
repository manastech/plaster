import pandas as pd
import numpy as np
from plaster.run.base_result import BaseResult
from plaster.run.test_nn.test_nn_params import TestNNParams


class TestNNResult(BaseResult):
    name = "test_nn"
    filename = "test_nn.pkl"

    required_props = dict(
        params=TestNNParams,
        test_true_pep_iz=np.ndarray,
        test_dt_mat=np.ndarray,
        test_dyetracks_df=pd.DataFrame,
        test_dt_pep_sources_df=pd.DataFrame,
        test_true_dt_iz=np.ndarray,
        test_pred_dt_iz=np.ndarray,
        test_dt_scores=np.ndarray,
        test_pred_pep_iz=np.ndarray,
        test_pep_scores=np.ndarray,
        test_scores=np.ndarray,
        test_vdists=np.ndarray,
        test_peps_pr=(type(None), pd.DataFrame),
        test_peps_pr_abund=(type(None), pd.DataFrame),
        train_true_pep_iz=(type(None), np.ndarray),
        train_dt_mat=(type(None), np.ndarray),
        train_dyetracks_df=(type(None), pd.DataFrame),
        train_dt_pep_sources_df=(type(None), pd.DataFrame),
        train_pred_dt_iz=(type(None), np.ndarray),
        train_dt_scores=(type(None), np.ndarray),
        train_pred_pep_iz=(type(None), np.ndarray),
        train_pep_scores=(type(None), np.ndarray),
        train_scores=(type(None), np.ndarray),
        train_vdists=(type(None), np.ndarray),
        train_peps_pr=(type(None), pd.DataFrame),
        train_peps_pr_abund=(type(None), pd.DataFrame),
    )

    def includes_train_results(self):
        return self.train_dt_mat is not None

    def __repr__(self):
        try:
            return (
                f"TestNNResult with average score {np.mean(self.test_scores)} "
                f"({'includes' if self.includes_train_results else 'does not include'} train results)"
            )
        except:
            return "TestNNResult"
