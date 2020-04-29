"""
Train the Random Forest classifier.

TASK:
    This is to be deprecated once the Nearest-Neighbor classifier
    improves.  The RF is excellent but its memory demands grow
    quadratically with the number of classes and becomes unwieldy
    at proteome scales.

"""

from plumbum import local
from plaster.run.sklearn_rf import SciKitLearnRandomForestClassifier
from plaster.run.train_rf.train_rf_result import TrainRFResult
from plaster.tools.log.log import important


def train_rf(train_rf_params, sim_result, progress=None):
    if sim_result.params.n_samples_train > 1000:
        important(
            "Warning: RF does not memory-scale well when the n_samples_train is > 1000."
        )
    X = sim_result.train_radmat
    y = sim_result.train_true_pep_iz
    classifier = SciKitLearnRandomForestClassifier(**train_rf_params)
    classifier.train(X, y, progress)
    return TrainRFResult(params=train_rf_params, classifier=classifier)
