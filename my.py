def isprime(x):
    if x == 2:
        return 1
    for i in range(2, int(x)):
        if x % i == 0:
            return 0
    else:
        return 1

if __name__ == '__main__':
    x = eval(input())
    print('{}='.format(x), end = '')
    if isprime(x):
        print(x)
    else:
        p = []
        for i in range(2, int(x)):
            if isprime(i):
                p.append(i)
        while isprime(x) != 1:
            for num in p:
                if x % num == 0:
                    print('{}*'.format(num), end = '')
                    x /= num
                    break
        print(int(x))