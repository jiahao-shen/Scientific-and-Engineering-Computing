"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file main.py
@ide: PyCharm
@time: 2018-12-23 15:25:13
@blog: https://jiahaoplus.com
"""
import sys
sys.path.append('..')

from fitting import *
from lab3.interpolation import *
import numpy as np
import matplotlib.pyplot as plt


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
    X = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    Y = [1.0, 0.41, 0.50, 0.61, 0.91, 2.02, 2.46]

    y = least_squares(X, Y, 2)
    print('Least Squares Method:', y)

    x_val = np.linspace(0, 1, 100)
    y_val = sy.lambdify(x, y)(x_val)
    plt.figure(figsize=(5, 15))
    plt.subplot(1, 3, 1)
    plt.title('Least Squares Method')
    plt.plot(x_val, y_val)
    plt.scatter(X, Y)

    y = lagrange(X, Y)
    print('Lagrange Method:', y)

    y_val = sy.lambdify(x, y)(x_val)
    plt.subplot(1, 3, 2)
    plt.title('Lagrange Method')
    plt.plot(x_val, y_val)
    plt.scatter(X, Y)

    y = newton(X, Y)
    print('Newton Method:', y)

    y_val = sy.lambdify(x, y)(x_val)
    plt.subplot(1, 3, 3)
    plt.title('Newton Method')
    plt.plot(x_val, y_val)
    plt.scatter(X, Y)

    plt.show()

    print('----------------------------------')

    print('Test 2')
    for n in range(2, 10):
        print('n =', n)
        X, Y = get_XY(n)

        y = lagrange(X, Y)
        print('Lagrange Method: f(0.98) =', sy.lambdify(x, y)(0.98))

        x_val = np.linspace(-1, 1, 100)
        y_val = sy.lambdify(x, y)(x_val)
        plt.subplot(2, 2, 1)
        plt.title('Lagrange Method(n=' + str(n) + ')')
        plt.scatter(X, Y)
        plt.plot(x_val, y_val)

        y = newton(X, Y)
        print('Newton Method: f(0.98) =', sy.lambdify(x, y)(0.98))

        y_val = sy.lambdify(x, y)(x_val)
        plt.subplot(2, 2, 2)
        plt.title('Newton Method(n=' + str(n) + ')')
        plt.plot(x_val, y_val)
        plt.scatter(X, Y)

        y = least_squares(X, Y, 3)
        print('Least Squares Method: f(0.98) =', sy.lambdify(x, y)(0.98))

        y_val = sy.lambdify(x, y)(x_val)
        plt.subplot(2, 2, 3)
        plt.title('Least Squares Method(n=' + str(n) + ')')
        plt.plot(x_val, y_val)
        plt.scatter(X, Y)

        y = 1 / (1 + 25 * x ** 2)
        print('y = 1 / (1 + 25 * x ^ 2): f(0.98) =', sy.lambdify(x, y)(0.98))
        y_val = sy.lambdify(x, y)(x_val)
        plt.subplot(2, 2, 4)
        plt.plot(x_val, y_val)
        plt.scatter(X, Y)
        plt.title('1/(1+25x^2) (n=' + str(n) + ')')
        plt.show()

        print()


if __name__ == '__main__':
    main()
