# Implementação do método da seção áurea
# seguido de interpolação polinomial

import numpy as np
import numpy.linalg

def bhaskara(a, b, c):
    delta = b*b - 4.0*a*c

    x1 = (-b + np.sqrt(delta)) / (2.0 * a)
    x2 = (-b - np.sqrt(delta)) / (2.0 * a)

    return x1, x2

tau = (np.sqrt(5.0) - 1.0) / 2.0 # Número de ouro

def min_1d(Xini, S, F, a=-10, b=10, steps=30):

    def omega(alpha):
        Xnew = Xini + alpha * S
        return F(*Xnew)

    ### Método da seção áurea:

    # Valores de x e y iniciais
    y = a + tau * (b - a)
    x = b - tau * (b - a)

    # Avalia omega para os valores de x e y:
    ox, oy = map(omega, (x, y))

    # Valores de omega para a e b ainda não são necessários:
    oa = ob = None

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
        #print(a, x, y, b)

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

    rhs = [oa, ox, oy, ob]

    p = np.linalg.solve(A, rhs)

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
	# Sover error is bigger than the difference between the values,
	# take the smaller of them
        alpha = [a,x,y,b][rhs.index(min(rhs))]

    return Xini + alpha * S
