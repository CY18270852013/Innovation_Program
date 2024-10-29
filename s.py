def m(x, y):
    if y % x == 0:
        return x
    else:
        return m(y % x, x)

if __name__ == '__main__':
    x, y = eval(input())
    print(m(x, y))