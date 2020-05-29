"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file: main.py
@ide: PyCharm
@time: 2018-12-25 10:21:57
@blog: https://jiahaoplus.com
"""
from numpy import zeros
from sympy import Symbol, exp, integrate


def equation_1():
    """I_n = 1 - n * I_{n-1}
    :return:
    """
    I = zeros(20)
    I[0] = 0.63212056
    for i in range(1, 14):
        I[i] = 1 - i * I[i - 1]
    print(I)


def equation_2():
    """I_n = 1 - n * I_{n-1}
    :return:
    """
    I = zeros(20)
    I[15] = 0.042746233
    for i in reversed(range(15)):
        I[i] = (1 / (i + 1)) * (1 - I[i + 1])
    print(I)


def equation_3():
    """Using calculus in Sympy
    :return:
    """
    x = Symbol('x')
    I = zeros(20)
    for i in range(20):
        tmp = (1 / exp(1)) * integrate((x ** i) * (exp(x)), (x, 0, 1))
        I[i] = tmp

    print(I)


def main():
    equation_1()
    equation_2()
    equation_3()


if __name__ == '__main__':
    main()
