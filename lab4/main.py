"""
@project: Scientific-and-Engineering-Computing
@author: sam
@file main.py
@ide: PyCharm
@time: 2018-12-13 10:35:34
@blog: https://jiahaoplus.com
"""
from calculus import *


def main():
    y = 1 / (1 + x)
    print('Simpson Method')
    result = simpson(y, 0, 1)
    print('I =', result)

    print('----------------------------------')

    print('Complex Simpson Method')
    result = complex_simpson(y, 0, 1, 5)
    print('I =', result)

    print('----------------------------------')

    print('Variable Step Simpson Method')
    result = variable_step_simpson(y, 0, 1)
    print('I =', result)


if __name__ == '__main__':
    main()
