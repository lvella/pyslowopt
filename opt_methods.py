from au_section import min_1d
from convergence import ConvergenceTester
import numpy as np
import numpy.linalg as linalg

def numerical_grad(F, delta):
	half_delta = delta * 0.5
	def ret(*X):
		X = list(X)
		grad = []
		for i in range(len(X)):
			a = X[:]
			a[i] -= half_delta
			b = X[:]
			b[i] += half_delta
			grad.append((F(*b) - F(*a)) / delta)
		return np.array(grad)
	return ret

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

def variable_matrix(Xini, F, gradF=None, theta=0.0, search_radius=3, abs_tolerance=1e-4, rel_tolerance=1e-3, max_iters=1000):
    if gradF is None:
        gradF = numerical_grad(F, 0.01)

    conv_test = ConvergenceTester(F, abs_tolerance, rel_tolerance)
    X = Xini
    history = [F(*X)]
    inv_hess = np.identity(len(X))
    curr_grad = gradF(*X)
    for i in range(1, max_iters+1):
        S = -np.dot(inv_hess, curr_grad)
        Xprev = X
        X = min_1d(X, S, F, -search_radius, search_radius)

        history.append(F(*X))

        if conv_test.has_converged(Xprev, X):
            break

        new_grad = gradF(*X)
        y = new_grad - curr_grad
        p = X - Xprev
        sigma = np.dot(p, y)
        iHy = np.dot(inv_hess, y)
        tau = np.dot(y, iHy)
        D = (((sigma + theta * tau)/(sigma * sigma)) * np.outer(p,p)
            + ((theta - 1.0)/tau) * np.outer(iHy, iHy)
            - (theta/sigma) * (np.outer(iHy, p) + np.outer(p, iHy)))
        inv_hess = inv_hess + D
        curr_grad = new_grad

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

def newton(Xini, F, gradF=None, hessF=None, search_radius=10, abs_tolerance=1e-4, rel_tolerance=1e-3, max_iters=100):
    if gradF is None:
        gradF = numerical_grad(F, 0.01)
    if hessF is None:
        hessF = numerical_grad(gradF, 0.01)

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
