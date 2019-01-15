"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file main.py
@ide: PyCharm
@time: 2018-11-20 00:00:00
@blog: https://jiahaoplus.com
"""
from math import fabs
from sympy import Symbol, lambdify, diff, exp


EPS = 1e-8
x = Symbol('x')


def bisection(y):
    """Bisection Method
    :param y: function expression
    :return:
    """
    f = lambdify(x, y)
    l, r, cnt = 0, 100, 0
    while True:
        cnt += 1
        mid = (l + r) / 2
        if f(mid) * f(l) <= 0:
            r = mid
        elif f(mid) * f(r) <= 0:
            l = mid
        if fabs(l - r) <= EPS:
            print('x =', mid)
            print('cnt =', cnt)
            break


def newton(y):
    """Newton Method
    :param y: function expression
    :return:
    """
    f = lambdify(x, y)
    d = lambdify(x, diff(y, x))

    x0, cnt = 0, 0
    while True:
        cnt += 1
        new_x = x0 - f(x0) / d(x0)
        if fabs(x0 - new_x) <= EPS:
            print('x =', x0)
            print('cnt =', cnt)
            break
        x0 = new_x


def newton_downhill(y):
    """Newton Downhill Method
    :param y: function expression
    :return:
    """
    f = lambdify(x, y)
    d = lambdify(x, diff(y, x))

    x0, cnt = 0, 0
    while True:
        cnt += 1
        k = 1
        new_x = x0 - k * f(x0) / d(x0)
        while fabs(f(new_x)) >= fabs(f(x0)):
            k /= 2
            new_x = x0 - k * f(x0) / d(x0)
        if fabs(x0 - new_x) <= EPS:
            print('x =', x0)
            print('cnt =', cnt)
            break
        x0 = new_x


def secant(y):
    """Secant Method
    :param y: function expression
    :return:
    """
    f = lambdify(x, y)
    old_x, x0, cnt = 0, 5, 0
    while True:
        cnt += 1
        new_x = x0 - f(x0) * (x0 - old_x) / (f(x0) - f(old_x))
        if fabs(x0 - new_x) <= EPS:
            print('x =', x0)
            print('cnt =', cnt)
            break
        old_x, x0 = x0, new_x
        x0 = new_x


def main():
    y1 = x ** 2 - 3 * x + 2 - exp(x)
    y2 = x ** 3 - x - 1

    print('Bisection Method')
    print('Function1')
    bisection(y1)
    print('Function2')
    bisection(y2)

    print('--------------------')
    print('Newton Method')
    print('Function1')
    newton(y1)
    print('Function2')
    newton(y2)

    print('--------------------')
    print('Newton Downhill Method')
    print('Function1')
    newton_downhill(y1)
    print('Function2')
    newton_downhill(y2)

    print('--------------------')
    print('Secant Method')
    print('Function1')
    secant(y1)
    print('Function2')
    secant(y2)


if __name__ == '__main__':
    main()
