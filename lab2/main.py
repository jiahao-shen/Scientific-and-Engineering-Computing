from lab2.algorithm import *
import time


def main():
    """
    目前使用迭代法计算方程组1时不收敛
    以后会进行修改
    """
    A1 = array([[10, -7, 0, 1], [-3, 2.099999, 6, 2],
                [5, -1, 5, -1], [2, 1, 0, 2]])
    b1 = array([8, 5.900001, 5, 1])

    A2, b2 = get_equation_set_2(4)

    """---------------------------------------------"""
    print('Pivot Gauss')

    t = time.time()
    x1 = pivot_gauss(A1, b1)
    t = (time.time() - t) * 1000
    print('x1 =', x1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time.time()
    x2 = pivot_gauss(A2, b2)
    t = (time.time() - t) * 1000
    print('x2 =', x2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('LU Decomposition——Doolittle')

    t = time.time()
    x1 = lu_decomposition_doolittle(A1, b1)
    t = (time.time() - t) * 1000
    print('x1 =', x1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time.time()
    x2 = lu_decomposition_doolittle(A2, b2)
    t = (time.time() - t) * 1000
    print('x2 =', x2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('Jacobi')

    x0 = random.rand(4)

    t = time.time()
    x1, cnt1 = jacobi(A1, b1, x0)
    t = (time.time() - t) * 1000
    print('x1 =', x1)
    print('cnt1 =', cnt1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time.time()
    x2, cnt2 = jacobi(A2, b2, x0)
    t = (time.time() - t) * 1000
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('Successive Over Relaxation(SOR)')

    W = [1.1, 1.25, 1.5]

    x0 = random.rand(4)

    for w in W:
        print('w =', w)

        t = time.time()
        x1, cnt1 = successive_over_relaxation(A1, b1, w, x0)
        t = (time.time() - t) * 1000
        print('x1 =', x1)
        print('cnt1 =', cnt1)
        print('Time:' + str(t) + 'ms')

        print()

        t = time.time()
        x2, cnt2 = successive_over_relaxation(A2, b2, w, x0)
        t = (time.time() - t) * 1000
        print('x2 =', x2)
        print('cnt2 =', cnt2)
        print('Time:' + str(t) + 'ms')

        print()
    print('----------------------------------')

    """---------------------------------------------"""
    print('Conjugate Gradient')

    x0 = random.rand(4)

    t = time.time()
    x1, cnt1 = conjugate_gradient(A1, b1, x0)
    t = (time.time() - t) * 1000
    print('x1 =', x1)
    print('cnt1 =', cnt1)
    print('Time:' + str(t) + 'ms')

    print()

    t = time.time()
    x2, cnt2 = conjugate_gradient(A2, b2, x0)
    t = (time.time() - t) * 1000
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')

    print('----------------------------------')


if __name__ == '__main__':
    main()
