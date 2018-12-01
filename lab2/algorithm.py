"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file algorithm.py
@ide: PyCharm
@time: 2018-12-01 01:41:19
@blog: https://jiahaoplus.com
"""
from numpy import *

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
    L = eye(n)
    U = zeros((n, n))
    for k in range(n):
        for j in range(k, n):
            s = 0
            for t in range(0, k):
                s += L[k, t] * U[t, j]
            U[k, j] = A[k, j] - s
        for i in range(k + 1, n):
            s = 0
            for t in range(0, k):
                s += L[i, t] * U[t, k]
            L[i, k] = (A[i, k] - s) / U[k, k]

    y = zeros(n)
    for i in range(n):
        s = 0
        for j in range(0, i):
            s += L[i, j] * y[j]
        y[i] = b[i] - s

    x = zeros(n)
    for i in reversed(range(n)):
        s = 0
        for j in range(i + 1, n):
            s += U[i, j] * x[j]
        x[i] = (y[i] - s) / U[i, i]

    return x


def lu_decomposition_cholesky(A, b):
    """LU——Cholesky Method
    :param A: matrix
    :param b: vector
    :return: vector x
    """
    A, b = A.copy(), b.copy()

    if not check_symmetric_and_positive_definite_matrix(A):
        return NaN

    n = len(A)
    L = eye(n)

    for j in range(n):
        s = 0
        for k in range(j):
            s += L[j, k] ** 2
        L[j, j] = sqrt(A[j, j] - s)

        for i in range(j + 1, n):
            s = 0
            for k in range(j):
                s += L[i, k] * L[j, k]
            L[i, j] = (A[i, j] - s) / L[j, j]

    y = zeros(n)

    for i in range(n):
        s = 0
        for j in range(0, i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]

    x = zeros(n)
    for i in reversed(range(n)):
        s = 0
        for j in range(i + 1, n):
            s += L[j, i] * x[j]
        x[i] = (y[i] - s) / L[i, i]

    return x


def check_spectral_radius(B):
    """Check Spectral Radius of matrix B
    :param B: matrix
    :return: True or False
    """
    eigenvalues = linalg.eigvals(B)
    spectral_radius = linalg.norm(eigenvalues, ord=inf)
    if spectral_radius - 1 >= eps or 1 - spectral_radius <= eps:
        print('Spectral radius >= 1, not convergency')
        return False
    else:
        return True


def check_gauss_seidel_convergency(A):
    """Check whether matrix A is convergency in Gauss Seidel Method
    :param A: matrix
    :return: True or False
    """
    n = len(A)
    D = eye(n)
    L = tril(A, -1)
    U = triu(A, 1)
    for i in range(n):
        D[i, i] = A[i, i]
    B = -dot(linalg.inv(D + L), U)

    return check_spectral_radius(B)


def gauss_seidel(A, b, x0):
    """Gauss Seidel Method
    :param A: matrix
    :param b: vector
    :param x0: vector
    :return: vector x, iterations cnt
    """
    A, b = A.copy(), b.copy()

    if not check_gauss_seidel_convergency(A):
        return NaN, NaN

    n = len(A)

    x = x0.copy()
    cnt = 0
    while True:
        y = x.copy()
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]
        cnt += 1
        if linalg.norm(x - y) <= eps:
            break

    return x, cnt


def check_jacobi_convergency(A):
    """Check whether matrix A is convergency in Jacobi Method
    :param A: matrix
    :return: True or False
    """
    n = len(A)
    D = eye(n)
    L = tril(A, -1)
    U = triu(A, 1)
    for i in range(n):
        D[i, i] = A[i, i]
    B = -dot(linalg.inv(D), L + U)

    return check_spectral_radius(B)


def jacobi(A, b, x0):
    """Jacobi Method
    :param A: matrix
    :param b: vector
    :param x0: vector
    :return: vector x, iterations cnt
    """
    A, b = A.copy(), b.copy()

    if not check_jacobi_convergency(A):
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


def check_successive_over_relaxation_convergency(A, w):
    """Check whether matrix A is convergency in SOR Method
    :param A: matrix
    :param w: float(1 < w < 2)
    :return: True or False
    """
    n = len(A)
    D = eye(n)
    L = tril(A, -1)
    U = triu(A, 1)
    for i in range(n):
        D[i, i] = A[i, i]

    B = dot(linalg.inv(D + dot(w, L)), (1 - w) * D - dot(w, U))

    return check_spectral_radius(B)


def successive_over_relaxation(A, b, w, x0):
    """Successive Over Relaxation Method
    :param A: matrix
    :param b: vector
    :param w: float(1 < w < 2)
    :param x0: vector
    :return: vector x, iterations cnt
    """
    A, b = A.copy(), b.copy()

    if not check_successive_over_relaxation_convergency(A, w):
        return NaN, NaN

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
        print('A must be symmetric and positive definite matrix')
        return False

    values = linalg.eigvals(A)
    if any(values < 0):
        print('A must be symmetric and positive definite matrix')
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


def test():
    """Just test
    :return:
    """
    A = array([[1, 1, 2],
               [1, 2, 0],
               [2, 0, 11]])
    b = array([5, 8, 7])
    x = lu_decomposition_cholesky(A, b)
    # [-2, 5, 1]
    print(x)

    x = lu_decomposition_doolittle(A, b)
    # [-2, 5, 1]
    print(x)

    A = array([[2, -1, 0],
               [-1, 3, -1],
               [0, -1, 2]])
    b = array([1, 8, -5])
    x = lu_decomposition_doolittle(A, b)
    # [2, 3, -1]
    print(x)

    x0 = array([-0.5, 2.6667, -2.5000])
    x, cnt = jacobi(A, b, x0)
    # [2, 3, -1]
    print(x, cnt)

    x0 = array([0.5, 2.8333, -1.0833])
    x, cnt = gauss_seidel(A, b, x0)
    # [2, 3, -1]
    print(x, cnt)

    x0 = array([0.55, 3.1350, -1.0257])
    x, cnt = successive_over_relaxation(A, b, 1.1, x0)
    # [2, 3, -1]
    print(x, cnt)

    A = array([[0.76, -0.01, -0.14, -0.16],
               [-0.01, 0.88, -0.03, 0.06],
               [-0.14, -0.03, 1.01, -0.12],
               [-0.16, 0.06, -0.12, 0.72]])
    b = array([0.68, 1.18, 0.12, 0.74])
    x0 = array([0.0, 0, 0, 0])
    x, cnt = successive_over_relaxation(A, b, 1.05, x0)
    # [1.271497 1.284355 0.485795 1.284269] 7
    print(x, cnt)

    A = array([[2, -1, 1],
               [1, 1, 1],
               [1, 1, -2]])
    # Spectral radius >= 1, not convergency
    # False
    print(check_jacobi_convergency(A))
    # True
    print(check_gauss_seidel_convergency(A))

    A = array([[31.0, -13, 0, 0, 0, -10, 0, 0, 0],
               [-13, 35, -9, 0, -11, 0, 0, 0, 0],
               [0, -9, 31, -10, 0, 0, 0, 0, 0],
               [0, 0, -10, 79, -30, 0, 0, 0, -9],
               [0, 0, 0, -30, 57, -7, 0, -5, 0],
               [0, 0, 0, 0, -7, 47, -30, 0, 0],
               [0, 0, 0, 0, 0, -30, 41, 0, 0],
               [0, 0, 0, 0, -5, 0, 0, 27, -2],
               [0.0, 0, 0, -9, 0, 0, 0, -2, 29]])
    b = array([-15, 27, -23, 0, -20, 12, -7, 7, 10])
    x0 = random.rand(9)
    x, cnt = jacobi(A, b, x0)
    # [-0.289233  0.345436 - 0.712811 - 0.220608 - 0.4304    0.154309 - 0.057822 0.201054  0.290229]
    # 42
    print(x, cnt)

    A = array([[10, -7, 0, 1], [-3, 2.099999, 6, 2],
               [5, -1, 5, -1], [2, 1, 0, 2]])
    b = array([8, 5.900001, 5, 1])
    # Spectral radius >= 1, not convergency
    # False
    print(check_successive_over_relaxation_convergency(A, 1.0))


if __name__ == '__main__':
    test()
