# Implementação do método da seção áurea
# seguido de interpolação polinomial

import numpy as np
import numpy.linalg

def F(X1, X2):
    return 10 * X1**4 - 20 * X1**2 * X2 + 10 * X2**2 + X1**2 - 2 * X1 + 5

Xini = np.array([-0.5, -0.5]) # Projeto inicial
S = np.array([np.sin(1.0), np.cos(1.0)])  # Direção de busca

a = -10.0 # a inicial (a0)
b = 10.0  # b inicial (b0)

steps = 10 # número de iterações no método da seção áurea

# Restringe F a uma função de alfa e avalia
evaluations = 0
def omega(alpha):
    global evaluations
    evaluations += 1

    X = Xini + alpha * S
    return F(*X)

tau = (np.sqrt(5.0) - 1.0) / 2.0 # Número de ouro

### Método da seção áurea:

# Valores de x e y iniciais
y = a + tau * (b - a)
x = b - tau * (b - a)

# Avalia omega para os valores de x e y:
ox, oy = map(omega, (x, y))

# Valores de omega para a e b ainda não são necessários:
oa = ob = None

print("Iterações do método da seção áurea:")

def print_info(i):
    print("i =", i)
    print("α:    {:10.5f} {:10.5f} {:10.5f} {:10.5f}".format(a, x, y, b))
    out = ['{:>10s}'.format('???') if e is None else '{:10.5f}'.format(e)
            for e in (oa, ox, oy, ob)]
    print("ω(α): {}\n".format(' '.join(out)))
print_info(0)

for i in range(steps):
    if ox < oy:
        b, ob = y, oy
        y, oy = x, ox
        x = b - tau * (b - a)
        ox = omega(x)
    else:
        a, oa = x, ox
        x, ox = y, oy
        y = a + tau * (b - a)
        oy = omega(y)
    print_info(i + 1)

### Interpolação cúbica:

A = np.array([
    [1.0, a, a**2, a**3],
    [1.0, x, x**2, x**3],
    [1.0, y, y**2, y**3],
    [1.0, b, b**2, b**3]
    ])

# Valores de omega(a) e omega(b) são necessários agora:
if oa is None:
    oa = omega(a)
if ob is None:
    ob = omega(b)

rhs = np.array([oa, ox, oy, ob])

p = np.linalg.solve(A, rhs)
print("Polinômio interpolação cúbica:")
print("P = {:.6f} + {:.6f} α + {:.6f} α² + {:.6f} α³".format(*p))

# Fazendo dP/d alfa = 0, temos 2 pontos de inflexão do polinômio:
def bhaskara(a, b, c):
    delta = b*b - 4.0*a*c

    x1 = (-b + np.sqrt(delta)) / (2.0 * a)
    x2 = (-b - np.sqrt(delta)) / (2.0 * a)

    return x1, x2

r1, r2 = bhaskara(3.0*p[3], 2.0*p[2], p[1])

def is_valid(alpha):
    return a <= alpha <= b

if is_valid(r1) and is_valid(r2):
    alpha = r1 if omega(r1) < omega(r2) else r2
elif is_valid(r1):
    alpha = r1
elif is_valid(r2):
    alpha = r2
else:
    raise "Nenhum ponto de inflexão encontrado dentro do intervalo inicial."

print("\nα mínimo encontrado: {}".format(alpha))
print("número de avaliações de ω:", evaluations)
print("\nω(α) = ω({}) = {}".format(alpha, omega(alpha)))
print("Novo projeto: Xini + α * S = ({}, {})".format(*(Xini + alpha * S)))
