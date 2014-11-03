#!/usr/bin/env python3

import numpy as np
import math
import itertools
from opt_methods import newton, steepest_descent, variable_matrix, powell, numerical_grad

def dist(X0, Y0, X1, Y1):
	return math.sqrt((X1 - X0)**2 + (Y1 - Y0)**2)

def get_coords(coords):
	if len(coords) % 2 != 0:
		raise "O número de coordenadas X deve ser igual ao número de coordenadas Y."

	N = len(coords) // 2

	X = [0.0] + list(coords[::2]) + [10.0*(N+1)]
	Y = [0.0] + list(coords[1::2]) + [0.0]
	return N, X, Y

def K(N, i):
	return 500.0 + 200.0 * (N/3.0 - i + 1.0)**2

W = 50.0
L0 = 10.0

def PE(*args):
	"""A entrada é um vetor de tamanho par, contendo as coordenadas X e Y
	das massas:
		[X0, Y0, X1, Y1, X2, Y2, ..., Xn, Yn]
	"""

	N, X, Y = get_coords(args)

	ret = 0.0
	for i in range(N+1):
		dL = dist(X[i], Y[i], X[i+1], Y[i+1]) - L0
		ret += K(N, i) * dL * dL * 0.5

	for i in range(1,N+1):
		ret += W * i * Y[i]

	return ret

import csv
import sys

def main():
	Xini = np.array([
		10.0, 0.0,
		20.0, 0.0,
		30.0, 0.0,
		40.0, 0.0,
		50.0, 0.0
		])

	grad = numerical_grad(PE, 0.01)
	hessian = numerical_grad(grad, 0.01)

	def print_result(i, X):
		print(' Iterações:', i)
		print(' Função objetivo:', PE(*X))
		print(' Gradiente:', ', '.join(('{:.3}'.format(e) for e in grad(*X))))
		print(' Projeto:', ', '.join(('{:.3}'.format(e) for e in X)))

	print('Newton:')
	newton_hist, i, X = newton(Xini, PE, grad, hessian)
	print_result(i, X)

	print('Máxima descida:')
	sd_hist, i, X = steepest_descent(Xini, PE, grad)
	print_result(i, X)

	print('Powell:')
	powell_hist, i, X = powell(Xini, PE)
	print_result(i, X)

	print('DFP:')
	dfp_hist, i, X = variable_matrix(Xini, PE, grad)
	print_result(i, X)

	print('BFGS:')
	bfgs_hist, i, X = variable_matrix(Xini, PE, grad, 1.0)
	print_result(i, X)

	try:
		with open(sys.argv[1], 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			size = max(len(newton_hist), len(sd_hist), len(powell_hist), len(dfp_hist), len(bfgs_hist))
			csvwriter.writerow(['Iteração', 'Newton', 'Máxima descida', 'Powell', 'DFP', 'BFGS'])
			for row in itertools.zip_longest(range(1, size+1), newton_hist, sd_hist, powell_hist, dfp_hist, bfgs_hist, fillvalue=''):
				csvwriter.writerow(row)
	except IndexError:
		pass

if __name__ == '__main__':
	main()
