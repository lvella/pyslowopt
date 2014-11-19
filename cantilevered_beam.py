#!/usr/bin/env python3

import numpy as np
from functools import partial
from opt_methods import variable_metric, powell
from constrained import alm

def get_sizes(sizes):
    if len(sizes) % 2 != 0:
        raise "O número de bs deve ser igual ao número de hs."

    N = len(sizes) // 2

    b = list(sizes[::2])
    h = list(sizes[1::2])
    return N, b, h

P = 50000.0
E = 2e7
L = 500.0
l_i = 100.0
sigma_ = 14000.0
y_ = 2.5

def I(b,h):
    return (b * h**3)/12.0

def g_y(*args):
    N, b, h = get_sizes(args)

    y_ant = dy_ant = 0.0
    for i in range(N):
        Pl_EI = P*l_i / (I(b[i], h[i]) * E)
        dy = Pl_EI * (L + l_i/2.0 - (i+1) * l_i) + dy_ant
        y = (l_i / 2.0) * Pl_EI * (L - (i+1) * l_i + 2*l_i/3.0) + dy_ant * l_i + y_ant
        dy_ant = dy
        y_ant = y

    return y/y_ - 1.0

def g_sigma(i, *args):
    N, b, h = get_sizes(args)

    M = P * (L - i * l_i)
    return (((M * h[i]) / (2.0 * I(b[i], h[i]))) / sigma_) - 1.0

#g_sigma = [partial(g_sigma_i, i) for i in range(5)]

def g_hb(i, *args):
    N, b, h = get_sizes(args)
    return h[i] - 20.0*b[i]

def g_b(i, *args):
    N, b, h = get_sizes(args)
    return 1.0 - b[i]

def g_h(i, *args):
    N, b, h = get_sizes(args)
    return 5.0 - h[i]

def V(*args):
    """
    [b0, h0, b1, h1, b2, h2, ..., bn, hn]
    """
    N, b, h = get_sizes(args)

    return sum([b[i] * h[i] * l_i for i in range(N)])

import csv
import sys

def main():
    N = 5
    Xini = np.array([5.0, 40.0] * N)

    g_list = [g_y]
    for g in (g_sigma, g_hb, g_b, g_h):
        for i in range(N):
            g_list.append(partial(g, i))

    lambda_ini = np.zeros(len(g_list))

    history, iters, X, g_lambda, h_lambda = alm(Xini, V, g_list, [], lambda_ini, [], 5, 2, 100000, variable_metric, abs_tolerance=100, rel_tolerance=1e-4, max_iters=1000)
    print('Projeto final:', ', '.join(('{:.3}'.format(e) for e in X)))
    print('Número de iterações MMLA:', iters)

    print('Restrições:')
    print('  ',g_y.__name__, '(', g_lambda[0],')', g_y(*X))
    li = 1
    for g in (g_sigma, g_hb, g_b, g_h):
        print('  ',g.__name__)
        for i in range(N):
            print('    ', i+1, '(', g_lambda[li],')', g(i, *X))
            li += 1

    try:
        with open(sys.argv[1], 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            size = len(history)
            csvwriter.writerow(['Iteração', 'Volume'])
            for row in zip(range(0, size+1), history):
                csvwriter.writerow(row)
    except IndexError:
        pass

if __name__ == '__main__':
	main()
