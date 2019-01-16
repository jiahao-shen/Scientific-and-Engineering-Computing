"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file interpolation.py
@ide: PyCharm
@time: 2018-12-05 20:05:12
@blog: https://jiahaoplus.com
"""
from sympy import Symbol, simplify, lambdify


x = Symbol('x')


def lagrange(X, Y):
    """Lagrange Interpolation
    :param X:
    :param Y:
    :return: lambda(x, y)
    """
    n = len(X)
    y = 0

    for i in range(n):
        l = 1
        for j in range(n):
            if j != i:
                l *= (x - X[j]) / (X[i] - X[j])
        y += l * Y[i]

    y = simplify(y)
    return y


def newton(X, Y):
    """Newton Interpolation
    :param X:
    :param Y:
    :return:
    """
    n = len(X)

    f = []
    y = Y[0]

    for k in range(n):
        s = 0
        for i in range(k + 1):
            t = Y[i]
            for j in range(k + 1):
                if j != i:
                    t /= X[i] - X[j]
            s += t
        f.append(s)

    for k in range(1, n):
        s = f[k]
        for i in range(k):
            s *= (x - X[i])
        y += s

    y = simplify(y)
    return y


def test():
    X = [-2, -1, 1, 2]
    Y = [5, 3, 17, 21]
    # -1.0*x**3 + 1.0*x**2 + 8.0*x + 9.0
    print(newton(X, Y))
    # -x**3 + x**2 + 8*x + 9
    print(lagrange(X, Y))


if __name__ == '__main__':
    test()
