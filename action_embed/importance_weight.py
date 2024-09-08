from typing import Callable

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import ClassifierMixin


def vanilla_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
    w_x_a = action_dist[:, data["action"]] / data["pscore"]
    return w_x_a


def clipped_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
    w_x_a = vanilla_weight(data=data, action_dist=action_dist)
    return np.minimum(w_x_a, kwargs["lambda"])


def marginal_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
    action_embed_dim = kwargs["action_embed_dim"] if "action_embed_dim" in kwargs else np.arange(data["p_e_d_a"].shape[-1])
    p_e_a = []
    for d in action_embed_dim:
        p_e_a.append(data["p_e_d_a"][:, data["action_context"][:, d], d])
    
    p_e_a = np.array(p_e_a).T.prod(axis=2)
    p_e_x_pi_e = (action_dist * p_e_a).sum(axis=1)
    p_e_x_pi_b = (data["pi_b"] * p_e_a).sum(axis=1)
    w_x_e = p_e_x_pi_e / p_e_x_pi_b
    
    return w_x_e


def estimated_marginal_weight(data: dict, action_dist: np.ndarray, **kwargs) -> np.ndarray:
    pi_a_x_e_estimator: ClassifierMixin = kwargs["weight_estimator"]
    action_embed_dim = kwargs["action_embed_dim"] if "action_embed_dim" in kwargs else np.arange(data["p_e_d_a"].shape[-1])
    
    estimator_name = pi_a_x_e_estimator.__class__.__name__
    if estimator_name == "LogisticRegression":
        encoder = OneHotEncoder(sparse=False, drop="first")
        e = encoder.fit_transform(data["action_context"][:, action_embed_dim])
    else:
        e = data["action_context"][:, action_embed_dim]
    
    x_e = np.c_[data["context"], e]
    pi_a_x_e_estimator.fit(x_e, data["action"])
    
    w_x_a = action_dist / data["pi_b"]
    pi_a_x_e_hat = np.zeros_like(w_x_a)
    pi_a_x_e_hat[:, np.unique(data["action"])] = pi_a_x_e_estimator.predict_proba(x_e)
    w_x_e_hat = (w_x_a * pi_a_x_e_hat).sum(axis=1)
    
    return w_x_e_hat
