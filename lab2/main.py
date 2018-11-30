from numpy import *
import time

eps = 1e-6
set_printoptions(suppress=True, precision=6)


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


def pivot_gauss(A, b):
    """Pivot Gauss Method
    :param A: matrix
    :param b: vector
    :return: vector x
    """
    A, b = A.copy(), b.copy()
    n = len(A)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if A[j, i] > A[i, i]:
                A[[i, j], :] = A[[j, i], :]
                b[[i, j]] = b[[j, i]]
            if A[j, i] != 0.0:
                m = A[j, i] / A[i, i]
                A[j, i:n] = A[j, i:n] - m * A[i, i:n]
                b[j] = b[j] - m * b[i]
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - dot(A[k, (k + 1):n], b[(k + 1):n])) / A[k, k]

    x = b
    return x


def lu_decomposition_doolittle(A, b):
    """LU——Doolittle Method
    :param A: matrix
    :param b: vector
    :return: vector x
    """
    A, b = A.copy(), b.copy()
    n = len(A)
    l = eye(n)
    u = zeros((n, n))
    for k in range(n):
        for j in range(k, n):
            s = 0
            for t in range(0, k):
                s += l[k, t] * u[t, j]
            u[k, j] = A[k, j] - s
        for i in range(k + 1, n):
            s = 0
            for t in range(0, k):
                s += l[i, t] * u[t, k]
            l[i, k] = (A[i, k] - s) / u[k, k]

    y = zeros(n)
    for i in range(n):
        s = 0
        for j in range(0, i):
            s += l[i, j] * y[j]
        y[i] = b[i] - s

    x = zeros(n)
    for i in reversed(range(n)):
        s = 0
        for j in range(i + 1, n):
            s += u[i, j] * x[j]
        x[i] = (y[i] - s) / u[i, i]

    return x


def check_jacobi_convergency(A):
    """
    :param A: matrix
    :return: True or False
    """
    n = len(A)
    D = eye(n)
    R = A.copy()
    for i in range(n):
        D[i, i] = A[i, i]
        R[i, i] = 0
    B = dot(linalg.inv(D), R)

    eigenvalues = linalg.eigvals(B)
    spectral_radius = linalg.norm(eigenvalues, ord=inf)
    if spectral_radius > 1:
        return False
    else:
        return True



def jacobi(A, b, x0):
    """Jacobi Method
    :param A: matrix
    :param b: vector
    :param x0: vector
    :return: vector x, iterations cnt
    """
    A, b = A.copy(), b.copy()

    if not check_jacobi_convergency(A):
        print("Spectral radius > 1, not convergency")
        return NaN, NaN

    n = len(A)

    x = x0.copy()
    cnt = 0
    while True:
        y = x.copy()
        for i in range(n):
            s = 0
            for j in range(0, n):
                if j != i:
                    s += A[i, j] * y[j]
            x[i] = (b[i] - s) / A[i, i]
        cnt += 1
        if linalg.norm(x - y) <= eps:
            break

    return x, cnt


def successive_over_relaxation(A, b, w, x0):
    """Successive Over Relaxation Method
    :param A: matrix
    :param b: vector
    :param w: float(0 < w < 2)
    :param x0: vector
    :return: vector x, iterations cnt
    """
    A, b = A.copy(), b.copy()

    n = len(A)

    x = x0.copy()
    cnt = 0
    while True:
        y = x.copy()
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x[i] = (1 - w) * x[i] + w * (b[i] - s) / A[i, i]
        cnt += 1
        if linalg.norm(x - y) <= eps:
            break

    return x, cnt


def check_symmetric_and_positive_definite_matrix(A):
    """Check whether matrix A is symmetric and positive definite
    :param A: matrix
    :return: True or False
    """
    if not (A.T == A).all():
        return False

    values = linalg.eigvals(A)
    if any(values < 0):
        return False

    return True


def conjugate_gradient(A, b, x0):
    """Conjugate Gradient Method
    :param A: matrix
    :param b: vector
    :param x0: vector
    :return: vector x, iterations cnt
    """
    A, b = A.copy(), b.copy()

    if not check_symmetric_and_positive_definite_matrix(A):
        print("A must be symmetric and positive definite matrix")
        return NaN, NaN

    n = len(A)

    x = x0.copy()
    r = b - dot(A, x)
    p = r
    cnt = 1
    while True:
        y = x
        alpha = dot(r.T, r) / dot(dot(p.T, A), p)
        x = x + dot(alpha, p)
        old_r = r
        r = r - dot(dot(alpha, A), p)
        beta = dot(r.T, r) / dot(old_r.T, old_r)
        p = r + dot(beta, p)
        cnt += 1
        if linalg.norm(x - y) <= eps:
            break

    return x, cnt


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

    t = time.time()
    x2, cnt2 = jacobi(A2, b2, x0)
    t = (time.time() - t) * 1000
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')
    print('----------------------------------')

    """---------------------------------------------"""
    print('Successive Over Relaxation(SOR)')

    W = [1.0, 1.25, 1.5]

    x0 = random.rand(4)

    for w in W:
        print('w =', w)
        t = time.time()
        x2, cnt2 = successive_over_relaxation(A2, b2, w, x0)
        t = (time.time() - t) * 1000
        print('x2 =', x2)
        print('cnt2 =', cnt2)
        print('Time:' + str(t) + 'ms')
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

    t = time.time()
    x2, cnt2 = conjugate_gradient(A2, b2, x0)
    t = (time.time() - t) * 1000
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')

    print('----------------------------------')


if __name__ == '__main__':
    main()
