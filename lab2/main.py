from numpy import *
import time

eps = 1e-6
set_printoptions(suppress=True, precision=6)


"""
input: int n
return: matrix A, b
"""
def get_equation_set_2(n):
    A = eye(n)
    b = zeros(n)
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)
            b[i] += A[i][j]
    return A, b


"""
Pivot Gauss Method
input: matrix A, vector b
return: vector x
"""
def pivot_gauss(A, b):
    A, b = A.copy(), b.copy()
    n = len(b)
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


"""
LU——Doolittle Method
input: matrix A, vector b
return: vector x
"""
def lu_decomposition_doolittle(A, b):
    A, b = A.copy(), b.copy()
    n = len(b)
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


"""
Jacobi Method
input: matrix A, vector b, vector x0
return: vector x, int cnt(iterations)
"""
def jacobi(A, b, x0):
    A, b = A.copy(), b.copy()
    n = len(b)

    x = x0
    cnt = 0
    while True:
        y = x.copy()
        for i in range(n):
            s = 0
            for j in range(0, n):
                if j != i:
                    s += A[i, j] * x[j]
            x[i] = (b[i] - s) / A[i, i]
        cnt += 1
        if linalg.norm(x - y) <= eps:
            break

    return x, cnt


"""
Successive Over Relaxation Method
input: matrix A, vector b, float w, vector x0
return: vector x, int cnt(iterations)
"""
def successive_over_relaxation(A, b, w, x0):
    A, b = A.copy(), b.copy()
    n = len(b)

    x = x0
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


"""
Conjugate Gradient Method
input: matrix A, vector b, vector x0
return: vector x, int cnt(iterations)
"""
def conjugate_gradient(A, b, x0):
    A, b = A.copy(), b.copy()
    n = len(b)

    x = x0
    r = b - dot(A, x)
    p = r
    cnt = 1
    while True:
        y = x
        alpha = dot(r.T, r) / dot(dot(p.T, A), p)
        x = x + dot(alpha, p)
        new_r = r
        r = r - dot(dot(alpha, A), p)
        beta = dot(r.T, r) / dot(new_r.T, new_r)
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
    # A1[[0, 2], :] = A1[[2, 0], :]
    # b1[0], b1[2] = b1[2], b1[0]
    # A1[[1, 2], :] = A1[[2, 1], :]
    # b1[1], b1[2] = b1[2], b1[1]

    A2, b2 = get_equation_set_2(4)

    print('Pivot Gauss')
    t = time.time()
    x1 = pivot_gauss(A1, b1)
    x2 = pivot_gauss(A2, b2)
    t = (time.time() - t) * 1000
    print('x1 =', around(x1, 4))
    print('x2 =', x2)
    print('Time:' + str(t) + 'ms')
    print('--------------------')

    print('LU Decomposition——Doolittle')
    t = time.time()
    x1 = lu_decomposition_doolittle(A1, b1)
    x2 = lu_decomposition_doolittle(A2, b2)
    t = (time.time() - t) * 1000
    print('x1 =', x1)
    print('x2 =', x2)
    print('Time:' + str(t) + 'ms')
    print('--------------------')

    print('Jacobi')
    t = time.time()
    # x1, cnt1 = jacobi(A1, b1)
    x0 = random.rand(4)
    x2, cnt2 = jacobi(A2, b2, x0)
    t = (time.time() - t) * 1000
    # print('x1 =', x1)
    # print('cnt1 =', cnt1)
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')
    print('--------------------')

    print('Successive Over Relaxation(SOR)')
    W = [1.0, 1.25, 1.5]
    for w in W:
        print('w =', w)
        t = time.time()
        x0 = random.rand(4)
        x2, cnt2 = successive_over_relaxation(A2, b2, w, x0)
        t = (time.time() - t) * 1000
        print('x2 =', x2)
        print('cnt2 =', cnt2)
        print('Time:' + str(t) + 'ms')
    print('--------------------')

    print('Conjugate Gradient')
    t = time.time()
    # x1, cnt1 = conjugate_gradient(A1, b1)
    x0 = random.rand(4)
    x2, cnt2 = conjugate_gradient(A2, b2, x0)
    t = (time.time() - t) * 1000
    # print('x1 =', x1)
    # print('cnt1 =', cnt1)
    print('x2 =', x2)
    print('cnt2 =', cnt2)
    print('Time:' + str(t) + 'ms')
    print('--------------------')


if __name__ == '__main__':
    main()
