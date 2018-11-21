import math

eps = 10 ** (-8)


def f1(x):
    return x ** 2 - 3 * x + 2 - math.exp(x)


def d1(x):
    return 2 * x - 3 - math.exp(x)


def f2(x):
    return x ** 3 - x - 1


def d2(x):
    return 3 * x ** 2 - 1


def binary(f):
    l, r, cnt = 0, 100, 0
    while True:
        cnt += 1
        mid = (l + r) / 2
        if f(mid) * f(l) <= 0:
            r = mid
        elif f(mid) * f(r) <= 0:
            l = mid
        if (math.fabs(l - r) <= eps):
            print('x =', mid)
            print('cnt =', cnt)
            break


def newton(f, d):
    x, cnt = 0, 0
    while True:
        cnt += 1
        new_x = x - f(x) / d(x)
        if (math.fabs(x - new_x) <= eps):
            print('x =', x)
            print('cnt =', cnt)
            break
        x = new_x


def newton_downhill(f, d):
    x, cnt = 0, 0
    while True:
        cnt += 1
        k = 1
        new_x = x - k * f(x) / d(x)
        while math.fabs(f(new_x)) >= math.fabs(f(x)):
            k /= 2
            new_x = x - k * f(x) / d(x)
        if math.fabs(x - new_x) <= eps:
            print('x =', x)
            print('cnt =', cnt)
            break
        x = new_x


def secant(f):
    old_x, x, cnt = 0, 5, 0
    while True:
        cnt += 1
        new_x = x - f(x) * (x - old_x) / (f(x) - f(old_x))
        if math.fabs(x - new_x) <= eps:
            print('x =', x)
            print('cnt =', cnt)
            break
        old_x, x = x, new_x
        x = new_x


def main():
    print('二分法')
    print('Function1')
    binary(f1)
    print('Function2')
    binary(f2)

    print('--------------------')
    print('牛顿法')
    print('Function1')
    newton(f1, d1)
    print('Function2')
    newton(f2, d2)

    print('--------------------')
    print('牛顿下山法')
    print('Function1')
    newton_downhill(f1, d1)
    print('Function2')
    newton_downhill(f2, d2)

    print('--------------------')
    print('弦截法')
    print('Function1')
    secant(f1)
    print('Function2')
    secant(f2)


if __name__ == '__main__':
    main()
