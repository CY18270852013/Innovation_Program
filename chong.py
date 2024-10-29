def find_QQ(a, name):
    if name in a.keys():
        return a[name]
    else:
        return None

def find_luckyguys(a):
    l = []
    for key, value in a.items():
        if len(str(value)) <= 5:
            l.append(key)
    return l

if __name__ == '__main__':
    a = dict([('xiaoming', 88888), ('ahua', 5555555), ('dazhuang', 11111), ('damao', 1234321), ('xiaomao', 1212121)])
    count = 1
    while count < 4:
        name = input()
        if find_QQ(a, name) != None:
            print(find_QQ(a, name))
            break
        elif count != 3:
            print('This guy is not in the list, you can still search {} times now.'.format(3 - count))
            count += 1
        else:
            print('Sorry, sir, this guy is also not in this list, you have no chance.')
            break
    
    l = find_luckyguys(a)
    for names in l:
        print(names)