"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file main.py
@ide: PyCharm
@time: 2018-12-05 20:05:02
@blog: https://jiahaoplus.com
"""
from interpolation import *


def get_XY(n):
    """Get points(x, y), -1 <= x <= 1, y = 1 / (1 + 25 * x ** 2)
    :param n:
    :return: x, y
    """
    X = []
    Y = []
    for i in range(n + 1):
        tmp = -1 + 2 * i / n
        X.append(tmp)
        Y.append(1 / (1 + 25 * tmp * tmp))

    return X, Y


def main():
    """
    :return:
    """
    print('Test 1')
    X = [0.2, 0.4, 0.6, 0.8, 1.0]
    Y = [0.98, 0.92, 0.81, 0.64, 0.38]
    I = [0, 1, 10, 11]
    print('Lagrange Method')
    y = sy.lambdify(x, lagrange(X, Y))
    for i in I:
        xi = 0.2 + 0.08 * i
        print('f(' + str(xi) + ') =', y(xi))
    print('----------------------------------')
    print('Newton Method')
    y = sy.lambdify(x, newton(X, Y))
    for i in I:
        xi = 0.2 + 0.08 * i
        print('f(' + str(xi) + ') =', y(xi))
    print('----------------------------------')

    print('Test 2')
    for n in range(2, 10):
        print('n =', n)
        X, Y = get_XY(n)
        y = lagrange(X, Y)
        sy.plot(y, (x, -1, 1), title='Lagrange(n=' + str(n) + ')')
        print('Lagrange Method:', y)
        y = newton(X, Y)
        sy.plot(y, (x, -1, 1), title='Newton(n=' + str(n) + ')')
        print('Newton Method:', y)
        print()


if __name__ == '__main__':
    main()
