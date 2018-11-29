from numpy import *
import time


# set_printoptions(precision=2, suppress=True)

# 列主元高斯消去
def pivot_gauss(a, b):
    a, b = a.copy(), b.copy()
    n = len(b)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if a[j, i] > a[i, i]:
                a[[i, j], :] = a[[j, i], :]
                b[[i, j]] = b[[j, i]]
            if a[j, i] != 0.0:
                m = a[j, i] / a[i, i]
                a[j, i:n] = a[j, i:n] - m * a[i, i:n]
                b[j] = b[j] - m * b[i]
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - dot(a[k, (k + 1):n], b[(k + 1):n])) / a[k, k]
    
    x = b
    return x 


# LU分解法——Doolittle分解
def lu_decomposition(a, b):
    a, b = a.copy(), b.copy()
    n = len(b)
    l = eye(n)
    u = zeros((n, n))
    for k in range(n):
        for j in range(k, n):
            sum = 0
            for t in range(0, k):
                sum += l[k, t] * u[t, j]
            u[k, j] = a[k, j] - sum
        for i in range(k + 1, n):
            sum = 0
            for t in range(0, k):
                sum += l[i, t] * u[t, k]
            l[i, k] = (a[i, k] - sum) / u[k, k]
    
    y = zeros(n)
    for i in range(n):
        sum = 0
        for j in range(0, i):
            sum += l[i, j] * y[j]
        y[i] = b[i] - sum
    
    x = zeros(n)
    for i in reversed(range(n)):
        sum = 0
        for j in range(i + 1, n):
            sum += u[i, j] * x[j]
        x[i] = (y[i] - sum) / u[i, i]
    
    return x
    

# 雅克比迭代法
def jacobi(a, b):
    a, b = a.copy(), b.copy()
    # a = array([[10, 3, 1], [2, -10, 3], [1, 3, 10]])
    # b = array([14, -5, 14])

    n = len(b)
    x = zeros(n)
    cnt = 10
    while cnt >= 0:
        y = x
        for i in range(n):
            sum = 0
            for j in range(0, i):
                sum += a[i, j] * y[j]
            for j in range(i + 1, n):
                sum += a[i, j] * y[j]
            x[i] = (b[i] - sum) / a[i, i]
        cnt -= 1
        print(x)
    


# 生成方程组2
def get_equation_set_2(n):
    a = eye(n)
    b = zeros(n)
    for i in range(n):
        for j in range(n):
            a[i, j] = 1 / (i + j + 1)
            b[i] += a[i][j]
    return a, b


def main():
    a1 = array([[10, -7, 0, 1], [-3, 2.099999, 6, 2],
              [5, -1, 5, -1], [2, 1, 0, 2]])
    b1 = array([8, 5.900001, 5, 1])
    
    a2, b2 = get_equation_set_2(4)

    # print('Pivot Gauss')
    # t = time.time()
    # x1 = pivot_gauss(a1, b1)
    # x2 = pivot_gauss(a2, b2)
    # print('x1 =', x1)
    # print('x2 =', x2)
    # t = (time.time() - t) * 1000
    # print('Time:' + str(t) + 'ms')
    # print('--------------------')

    # print('LU Decomposition——Doolittle')
    # t = time.time()
    # x1 = lu_decomposition(a1, b1)
    # x2 = lu_decomposition(a2, b2)
    # print('x1 =', x1)
    # print('x2 =', x2)
    # t = (time.time() - t) * 1000
    # print('Time:' + str(t) + 'ms')
    # print('--------------------')

    jacobi(a1, b1)

    




if __name__ == '__main__':
    main()
