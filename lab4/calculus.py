"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file calculus.py
@ide: PyCharm
@time: 2018-12-13 10:36:13
@blog: https://jiahaoplus.com
"""
from sympy import Symbol, lambdify


x = Symbol('x')
EPS = 1e-5


def simpson(y, a, b):
    """Simpson Method
    :param y: expression
    :param a: lower
    :param b: upper
    :return: integral
    """
    f = lambdify(x, y)
    result = (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))
    return result


def complex_simpson(y, a, b, n=10):
    """
    :param y: expression
    :param a: lower
    :param b: upper
    :param n: int
    :return: integral
    """
    f = lambdify(x, y)
    h = (b - a) / n

    s1 = 0
    for i in range(0, n):
        tmp = a + h * i + 0.5 * h
        s1 += f(tmp)

    s2 = 0
    for i in range(1, n):
        tmp = a + h * i
        s2 += f(tmp)

    result = (h / 6) * (f(a) + 4 * s1 + 2 * s2 + f(b))
    return result


def variable_step_simpson(y, a, b, eps=EPS):
    """Variable Step Simpson Method
    :param y: expression
    :param a: lower
    :param b: upper
    :param eps: default 1e-5
    :return: integral
    """
    f = lambdify(x, y)
    h = b - a
    n = 1
    T = (h / 2) * (f(a) + f(b))
    H = h * f((a + b) / 2)
    S = (1 / 3) * (T + 2 * H)
    while True:
        S_old = S
        n *= 2
        h /= 2
        T = (1 / 2) * (T + H)
        H = 0
        for j in range(1, n + 1):
            H += f(a + (j - 0.5) * h)
        H *= h
        S = (1 / 3) * (T + 2 * H)
        if abs(S - S_old) < eps:
            return S


def test():
    y = 1 / (1 + x)
    # 0.6944444444444443
    print(simpson(y, 0, 1))
    # 0.6931473746651161
    print(complex_simpson(y, 0, 1, 10))
    # 0.6931476528194189
    print(variable_step_simpson(y, 0, 1))


if __name__ == '__main__':
    test()
