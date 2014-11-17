import numpy as np
from convergence import ConvergenceTester

def alm(Xini, F, g_list, h_list, g_lambda_ini, h_lambda_ini, rp_ini, gamma, rp_max, unconstrained_minimizer, abs_tolerance=1e-4, rel_tolerance=1e-3, max_iters=100):
    if len(h_lambda_ini) != len(h_list):
        raise Exception('There must be one lambda for each equalty constraint.')
    if len(g_lambda_ini) != len(g_list):
        raise Exception('There must be one lambda for each inequalty constraint.')

    g_lambda = g_lambda_ini
    h_lambda = h_lambda_ini
    rp = rp_ini
    X = Xini

    def phi(lmbda, g, Xcur):
        return max(g(*Xcur), -lmbda / (2.0 * rp))

    def A(*Xcur):
        ret = F(*Xcur)

        for l, g in zip(g_lambda, g_list):
            gval = phi(l, g, Xcur)
            ret += l * gval + rp * gval * gval

        for l, h in zip(h_lambda, h_list):
            hval = h(*Xcur)
            ret += l * hval + rp * hval * hval
        return ret

    conv_test = ConvergenceTester(F, abs_tolerance, rel_tolerance)

    history = [F(*X)]

    for i in range(1, max_iters+1):
        Xprev = X
        X = unconstrained_minimizer(X, A)[-1]
        history.append(F(*X))

        if conv_test.has_converged(Xprev, X) and conv_test.test_constaints(h_list, g_list, X):
            break
        
        g_lambda = g_lambda + 2.0 * rp * np.array([phi(l, g, X) for l,g in zip(g_lambda, g_list)])
        h_lambda = h_lambda + 2.0 * rp * np.array([h(*X) for h in h_list])
        rp = min(rp_max, rp * gamma)

    return history, i, X, g_lambda, h_lambda
