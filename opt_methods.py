from au_section import min_1d
from convergence import ConvergenceTester
import numpy as np
import numpy.linalg as linalg

def steepest_descent(Xini, F, gradF, search_radius=10, abs_tolerance=1e-4, rel_tolerance=1e-3, max_iters=100):
    conv_test = ConvergenceTester(F, abs_tolerance, rel_tolerance)
    X = Xini
    history = [F(*X)]
    for i in range(1, max_iters+1):
        S = -gradF(*X)
        Xprev = X
        X = min_1d(X, S, F, -search_radius, search_radius)
        history.append(F(*X))

        if conv_test.has_converged(Xprev, X):
            break

    return history, i, X

def powell(Xini, F, abs_tolerance=1e-4, rel_tolerance=1e-3, max_iters=100):
    conv_test = ConvergenceTester(F, abs_tolerance, rel_tolerance)
    H = list(np.identity(len(Xini)))
    X = Xini
    history = [F(*X)]
    for i in range(1, max_iters+1):
        for S in H:
            X = min_1d(X, S, F)
        S = X - Xini
        Xprev = X
        X = min_1d(X, S, F)
        history.append(F(*X))

        if conv_test.has_converged(Xprev, X):
            break
        #if conv_test.first_test_passed:
        #    H = list(np.identity(len(Xini)))

        H = H[1:]
        H.append(S)

    return history, i, X

def newton(Xini, F, gradF, hessF, search_radius=10, abs_tolerance=1e-4, rel_tolerance=1e-3, max_iters=100):
    conv_test = ConvergenceTester(F, abs_tolerance, rel_tolerance)
    X = Xini
    history = [F(*X)]
    for i in range(1, max_iters+1):
        S = linalg.solve(hessF(*X), -gradF(*X))
        Xprev = X
        X = min_1d(X, S, F, -search_radius, search_radius)
        history.append(F(*X))

        if conv_test.has_converged(Xprev, X):
            break

    return history, i, X
