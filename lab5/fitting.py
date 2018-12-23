"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file fitting.py
@ide: PyCharm
@time: 2018-12-23 15:30:44
@blog: https://jiahaoplus.com
"""
import sympy as sy

x = sy.Symbol('x')


def inner_product(f1, f2, X):
    """Inner Product
    :param f1: function1
    :param f2: function2
    :param X:
    :return:
    """
    s = 0
    for item in X:
        s += f1(item) * f2(item)
    return s


def least_squares(X, Y, n):
    """Least Square Method
    :param X:
    :param Y:
    :param n: order
    :return: expression
    """
    def f(item):
        return Y[X.index(item)]

    p = []
    a = []
    alpha = []
    beta = []

    p.append(1)
    alpha.append(inner_product(sy.lambdify(x, x * p[0]), sy.lambdify(x, p[0]), X) / inner_product(sy.lambdify(x, p[0]),
                                                                                                  sy.lambdify(x, p[0]),
                                                                                                  X))
    p.append(x - alpha[0])

    for i in range(1, n):
        alpha.append(
            inner_product(sy.lambdify(x, x * p[i]), sy.lambdify(x, p[i]), X) / inner_product(sy.lambdify(x, p[i]),
                                                                                             sy.lambdify(x, p[i]), X))
        beta.append(
            inner_product(sy.lambdify(x, p[i]), sy.lambdify(x, p[i]), X) / inner_product(sy.lambdify(x, p[i - 1]),
                                                                                         sy.lambdify(x, p[i - 1]), X))
        p.append((x - alpha[i]) * p[i] - beta[i - 1] * p[i - 1])

    for i in range(n + 1):
        a.append(
            inner_product(sy.lambdify(x, p[i]), f, X) / inner_product(sy.lambdify(x, p[i]), sy.lambdify(x, p[i]), X))

    S = 0
    for i in range(n + 1):
        S += p[i] * a[i]

    S = sy.simplify(S)

    return S


def test():
    X = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Y = [1.00, 1.75, 1.96, 2.19, 2.44, 2.71, 3.00]
    S = least_squares(X, Y, 2)
    print(S)

    X = [0, 0.25, 0.50, 0.75, 1.00]
    Y = [1.0, 1.284, 1.6487, 2.1170, 2.7183]
    S = least_squares(X, Y, 2)
    print(S)


if __name__ == '__main__':
    test()
