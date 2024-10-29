def numsChain(num):
    n  = int(num)
    if n == 1:
        print('1->1')
        return None
    elif n == 89:
        print('{}->'.format(n), end = '')
        n = 0
        for c in num:
            n += (int(c) ** 2)
        num = str(n)
    print('{}'.format(n), end = '')
    while n != 1 and n != 89:
        n = 0
        for c in num:
            n += (int(c) ** 2)
        num = str(n)
        print('->{}'.format(n), end = '')

if __name__ == '__main__':
    num = input()
    numsChain(num)