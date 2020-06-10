from munch import Munch
import numpy as np
import pandas as pd
from plaster.tools.utils import utils
from plaster.run.call_bag import CallBag
from plaster.run.test_nn.test_nn_result import TestNNResult
from plaster.run.nn import nn
from plaster.tools.log.log import debug, prof


def test_nn(test_nn_params, prep_result, sim_result, progress=None, pipeline=None):
    n_channels, n_cycles = sim_result.params.n_channels_and_cycles

    n_phases = 2 if test_nn_params.include_training_set else 1
    if pipeline is not None:
        pipeline.set_phase(0, n_phases)

    test_result = nn(
        sim_result,
        sim_result.unflat("test_dyemat"),
        sim_result.unflat("test_radmat"),
        test_nn_params,
        progress,
    )
    test_result.true_pep_iz = sim_result.test_true_pep_iz

    call_bag = CallBag(
        true_pep_iz=test_result.true_pep_iz,
        pred_pep_iz=test_result.pred_pep_iz,
        scores=test_result.scores,
        prep_result=prep_result,
        sim_result=sim_result,
    )
    test_result.peps_pr = call_bag.pr_curve_by_pep()

    # If there is abundance information, compute the abundance-adjusted PR
    # This call returns None if there is no abundance info avail.
    test_result.peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance()

    if test_nn_params.include_training_set:
        # Permit testing for over-fitting by classifying on the train data

        if pipeline is not None:
            pipeline.set_phase(1, n_phases)

        real_pep_iz = prep_result.peps__no_decoys().pep_i.values
        keep_rows = np.isin(sim_result.train_true_pep_iz, real_pep_iz)
        train_radmat = sim_result.train_radmat[keep_rows]
        train_dyemat = sim_result.train_dyemat[keep_rows]

        train_result = nn(
            sim_result,
            utils.mat_lessflat(train_dyemat, n_channels, n_cycles),
            utils.mat_lessflat(train_radmat, n_channels, n_cycles),
            test_nn_params.use_gmm,
            progress,
        )
        train_result.true_pep_iz = sim_result.train_true_pep_iz
        call_bag = CallBag(
            true_pep_iz=train_result.true_pep_iz,
            pred_pep_iz=train_result.pred_pep_iz,
            scores=train_result.scores,
            prep_result=prep_result,
            sim_result=sim_result,
        )
        train_result.peps_pr = call_bag.pr_curve_by_pep()
        train_result.peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance()

    else:
        train_result = {k: None for k in test_result.keys()}

    def rename(d, prefix):
        return {f"{prefix}{k}": v for k, v in d.items()}

    return TestNNResult(
        params=test_nn_params,
        **rename(test_result, "test_"),
        **rename(train_result, "train_"),
    )
