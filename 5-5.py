from functools import partial
import numpy as np
from constrained import alm
from opt_methods import newton

def F(X1, X2):
	return (X1 - 1.0)**2 + (X2 - 1.0)**2

def h(X1, X2):
	return X1 - X2 - 2.0

def run():
	Xini = np.array([0.0, 0.0])
	for l in (0.0, 1.0, -4.0):
		print('Initial λ =', l)
		X = alm(Xini, F, [h], [l], 1, 2, 1024, newton, 3)
		print('------------------')


if __name__ == '__main__':
	run()
