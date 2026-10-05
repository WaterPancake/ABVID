"""Shared numeric functions from experiments/h1/evaluate.py."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fixed_fit(features, labels, config):
    if sorted(np.unique(labels).tolist()) != [0,1]: raise ValueError('Both training classes required')
    clf=LogisticRegression(C=config['C'],solver=config['solver'],tol=config['tol'],max_iter=config['max_iter'],
                           fit_intercept=config['fit_intercept'],class_weight=config['class_weight'],penalty='l2')
    model=make_pipeline(StandardScaler(),clf)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='scipy.optimize: The .*',category=DeprecationWarning)
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(features,labels)
    assert np.array_equal(model.classes_,[0,1])
    np.testing.assert_allclose(model[0].mean_,features.mean(axis=0),rtol=1e-6,atol=1e-7)
    return model
