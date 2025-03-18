import numpy as np


def gen_eps_greedy(q_s_a: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Markov policy based on epsilon-greedy rule."""
    N, A = q_s_a.shape
    bast_pi = np.zeros_like(q_s_a)
    a = np.argmax(q_s_a, axis=1)
    bast_pi[np.arange(N), a] = 1
    pi = (1.0 - eps) * bast_pi
    pi += eps / A

    return pi
