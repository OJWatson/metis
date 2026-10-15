"""SIRDS simulator used in the original documentation example.

This preserves the discrete-step multinomial approximation in the original
tutorial. It is an example data generator, not a newly inferred epidemic model.
"""

import numpy as np


def simulate_sirds(
    I0=0.005, beta=0.4, gamma=0.1, omega=0.02, mu=0.01,
    T=365, dt=1.0, N=1000, seed=None,
):
    """Return S,I,R,D counts including initial state on a fixed time grid."""
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not all(np.isfinite(v) for v in [I0, beta, gamma, omega, mu, T, dt]):
        raise ValueError("Simulator parameters must be finite")
    if not 0 <= I0 <= 1 or min(beta, gamma, omega, mu) < 0 or T <= 0 or dt <= 0:
        raise ValueError("Require 0<=I0<=1, nonnegative rates, and positive T/dt")
    p_rec, p_die, p_sus = 1-np.exp(-gamma*dt), 1-np.exp(-mu*dt), 1-np.exp(-omega*dt)
    if p_rec+p_die > 1:
        raise ValueError("The discrete competing-risk approximation needs a smaller dt")
    rng = np.random.default_rng(seed)
    n = int(T/dt)+1
    S, I, R, D = [np.zeros(n, dtype=np.int32) for _ in range(4)]
    I[0] = int(round(I0*N))
    S[0] = N-I[0]
    for k in range(n-1):
        new_I = rng.binomial(S[k], 1-np.exp(-beta*I[k]/N*dt))
        new_R, new_D, _ = rng.multinomial(I[k], [p_rec, p_die, 1-p_rec-p_die])
        new_S = rng.binomial(R[k], p_sus)
        S[k+1] = S[k]-new_I+new_S
        I[k+1] = I[k]+new_I-new_R-new_D
        R[k+1] = R[k]+new_R-new_S
        D[k+1] = D[k]+new_D
    return S, I, R, D
