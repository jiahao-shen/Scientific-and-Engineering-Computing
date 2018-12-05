"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file linear_function.py
@ide: PyCharm
@time: 2018-12-05 20:05:12
@blog: https://jiahaoplus.com
"""
import sympy as sy
import numpy as np

x = sy.Symbol('x')


def lagrange(X, Y):
    """Lagrange Interpolation
    :param X:
    :param Y:
    :return:
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

    print(y)


def newton(X, Y):
    """Newton Interpolation
    :param X:
    :param Y:
    :return:
    """
    # n = len(X)
    #
    # f = np.zeros(n + 1)
    #
    # for k in range(n):
    #     s = 0
    #     for i in range(k):
    #         t = Y[i]
    #         for j in range(k):
    #             if i != j:
    #                 t /= (X[i] - X[j])
    #         s += t
    #     f[k] = s
    #
    # y = 0
    #
    # for i in range(n):
    #     s = f[i]
    #     for j in range(i):
    #         s *= (x - X[j])
    #     y += s
    #
    # y = sy.simplify(y)
    #
    # y = sy.lambdify(x, y)
    #
    # print(y(8.4))


def test():
    X = [3, 6, 9]
    Y = [10, 8, 4]
    lagrange(X, Y)

    # X = [8.1, 8.3, 8.6, 8.7]
    # Y = [16.94410, 17.56492, 18.50515, 18.82091]

    # lagrange(X, Y)
    # newton(X, Y)


if __name__ == '__main__':
    test()
