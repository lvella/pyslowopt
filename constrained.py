import numpy as np

def alm(Xini, F, eq_constraints, lambda_ini, rp_ini, gamma, rp_max, unconstrained_minimizer, max_iters=100):
	if len(lambda_ini) != len(eq_constraints):
		raise Exception('There must be one lambda for each equality constraint.')
	if len(lambda_ini) >= len(Xini):
		raise Exception('There must be more project variables than equality constraints.')

	lmbda = lambda_ini
	rp = rp_ini
	X = Xini

	def print_values():
		print('  λ:', lmbda)
		print('  rp:', rp)
		print('  X:', X)
	print('Initial values:')
	print_values()

	def A(*X):
		ret = F(*X)
		for l, h in zip(lmbda, eq_constraints):
			hval = h(*X)
			ret += l * hval + rp * hval * hval
		return ret

	for i in range(1, max_iters+1):
		print('Iteration',i)

		X = unconstrained_minimizer(X, A)[-1]

		lmbda = lmbda + 2.0 * rp * np.array([h(*X) for h in eq_constraints])
		rp = min(rp_max, rp * gamma)

		print_values()

	return X
