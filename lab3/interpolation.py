"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file linear_function.py
@ide: PyCharm
@time: 2018-12-05 20:05:12
@blog: https://jiahaoplus.com
"""
import sympy as sy

x = sy.Symbol('x')


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

    y = sy.simplify(y)
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

    y = sy.simplify(y)
    return y


def test():
    pass
    # X = [-2, -1, 1, 2]
    # Y = [5, 3, 17, 21]
    # newton(X, Y)

    # lagrange(X, Y)


if __name__ == '__main__':
    test()
