"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file main.py
@ide: PyCharm
@time: 2018-11-30 01:41:19
@blog: https://jiahaoplus.com
"""
from linear_equations import *
from time import time


def get_equation_set_2(n):
    """Get equation 2
    :param n: int
    :return: matrix A, b
    """
    A = eye(n)
    b = zeros(n)
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
            b[i] += A[i][j]
    return A, b


def main():
    A1 = array([[10, -7, 0, 1], [-3, 2.099999, 6, 2],
                   [5, -1, 5, -1], [2, 1, 0, 2]])
    b1 = array([8, 5.900001, 5, 1])

    A2, b2 = get_equation_set_2(4)

    """---------------------------------------------"""
    print('Pivot Gauss')

    t = time()
    x1 = pivot_gauss(A1, b1)
    t = (time() - t) * 1000
    print('x1 =', x1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time()
    x2 = pivot_gauss(A2, b2)
    t = (time() - t) * 1000
    print('x2 =', x2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('LU Decomposition——Doolittle')

    t = time()
    x1 = lu_decomposition_doolittle(A1, b1)
    t = (time() - t) * 1000
    print('x1 =', x1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time()
    x2 = lu_decomposition_doolittle(A2, b2)
    t = (time() - t) * 1000
    print('x2 =', x2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('Jacobi')

    x0 = random.rand(4)
    print('x0 =', x0)

    t = time()
    x1, cnt1 = jacobi(A1, b1, x0)
    t = (time() - t) * 1000
    print('x1 =', x1)
    print('cnt1 =', cnt1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time()
    x2, cnt2 = jacobi(A2, b2, x0)
    t = (time() - t) * 1000
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('Successive Over Relaxation(SOR)')

    W = [1.1, 1.25, 1.5]

    x0 = random.rand(4)
    print('x0 =', x0)

    for w in W:
        print('w =', w)

        t = time()
        x1, cnt1 = successive_over_relaxation(A1, b1, w, x0)
        t = (time() - t) * 1000
        print('x1 =', x1)
        print('cnt1 =', cnt1)
        print('Time:' + str(t) + 'ms')

        print()

        t = time()
        x2, cnt2 = successive_over_relaxation(A2, b2, w, x0)
        t = (time() - t) * 1000
        print('x2 =', x2)
        print('cnt2 =', cnt2)
        print('Time:' + str(t) + 'ms')

        print()
    print('----------------------------------')

    """---------------------------------------------"""
    print('Conjugate Gradient')

    x0 = random.rand(4)
    print('x0 =', x0)

    t = time()
    x1, cnt1 = conjugate_gradient(A1, b1, x0)
    t = (time() - t) * 1000
    print('x1 =', x1)
    print('cnt1 =', cnt1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time()
    x2, cnt2 = conjugate_gradient(A2, b2, x0)
    t = (time() - t) * 1000
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')

    print('----------------------------------')


if __name__ == '__main__':
    main()
