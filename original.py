def m(s, n):
    maxp = 0
    maxs = ''
    for i in range(len(s) - n + 1):
        p = 1
        ds = s[i:i + n]
        for j in ds:
            p *= int(j)
        if p > maxp:
            maxp = p
            maxs = ds
    print('最大的{0}位数的数值是{1}，这个{0}位数整数是{2}'.format(n, maxp, maxs))

if __name__ == "__main__":
    s = input()
    n = eval(input())
    m(s, n)