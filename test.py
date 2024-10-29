aStr = 'Python!'
print(aStr.center(20))
bStr = 'No pains, No gains.'
print(bStr.count('No'))
print(bStr.find('s'))
print(bStr.find('d'))
print(bStr.index('No', 2))
l = ['I', 'want', 'to', 'make', 'friends', 'with', 'you.']
print('->'.join(l))
cStr = ' '.join(l)
cStr = cStr.replace('friends', 'a friend') 
print(cStr)
print(cStr.split(' '))
dStr = '            odsauf\n  asjf      '
print(dStr.strip())
print(bStr.capitalize())
print(bStr.endswith('.'))
print(cStr.format())
print(cStr.isalnum())
print(''.join(l).isalpha())
print(dStr.islower())
print(dStr.isspace())
print(bStr.istitle(), cStr.istitle(), 'HOHBHIHGYG'.isupper())
print(dStr.rjust(100))
print(bStr.lower())
print(bStr.lower().capitalize())
print(dStr.lstrip())
print(list(cStr.partition('make')))
print(bStr.rfind('No'), bStr.rindex('s'), bStr.rpartition('No'))
print(cStr.split('t'))
print(dStr.splitlines())
print(bStr.swapcase())
print(cStr.title())
print(cStr.upper())
print(aStr.zfill(20))
l2 = [2, 'hsaf', 98, 'iau', "uoiwqu"]
l3 = [1, 2, 3, 4]
l3.append(5)
print(l3)  #print(l3.append(5))会返回None, 因为列表是可变的
l2.extend(l3)
print(l2)
a = [7218, 'qiur', 72389, "wuiqry", [3, 'ue', 898, '83198']]
b = a.copy()
b[0] = 1
b[4][2] = 1
print(b)
print(a)
a.pop()
print(a)
a.pop(0)
print(a)
a.remove(72389)
print(a)
b.reverse()
print(b)
c = [3, 4, 89, 23, 12, 0, 289317]
c.sort(reverse = True)
print(c)
c.sort()
print(c)
print(c.count(0))
print(c.index(0))
c.insert(0, 1987)
num = c.index(0, 1, 6)
print(num)
print(c)



























