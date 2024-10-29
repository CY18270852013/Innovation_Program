def minNum(*args):
    l1 = list(args)
    l2 =(ar listgs)
    l1.sort()
    l2.sort()
    ind = 0
    num = 0
    for i in range(len(l2[:])):
        if l2[i] != 0:
            ind = i
            break
    l2.insert(0, l1[ind])
    l2.pop(ind + 1)
    num = int(''.join(map(str, l2)))
    return num

if __name__ == '__main__':
    args = eval(input())
    print(minNum(*args))