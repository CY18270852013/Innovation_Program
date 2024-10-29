def twonumSum(n, lst):
    for num in lst:
        if n - num in lst:
            return (lst.index(num), lst.index(n - num))
    else:
        return -1

if __name__ == '__main__':
    lst = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 19, 20, 21, 29, 34, 54, 65]
    n = eval(input())
    result = twonumSum(n, lst)
    if result == -1:
        print('not found')
    else:
        print(result)