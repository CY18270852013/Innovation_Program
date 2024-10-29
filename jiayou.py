a = {'yang': 1894, 'chen': 8218, 'xie': 89847}
b = dict([('ma', 17), ('li', 19), ('wang', 819)])
c = dict([['ma', 17], ['li', 19], ['wang', 819]])
d = dict(impossible = 89, jiangjun = 91, kajf= 194)
e = dict((('ma', 17), ('li', 19), ('wang', 819)))
print(a, b, c, d, e, sep = '\n')
f = {}.fromkeys(('fang', 'yu', 'wu'), 192347)
print(f)
print(sorted(a))
print(a['chen'])
a['chen'] =198647902938
print(a['chen'])
a['jiang'] = 987654321
print(a)
a['xiao'] = [198247, 102938]
print(a)
a['xiao'] += [238489038]
print(a)
print('he' not in a)
del a['jiang']
print(a)
print(len(a), len(b), len(c), len(d), len(e), len(f))
print(hash('yang'), hash('fang'))
print(a.keys())
print(a.values())
print(a.items())
print(a.get('chen'))
print(a.copy())
print(b.pop('ma'))
print(b)
b.clear()
print(b)
b.update(c)
print(b)
print(b.setdefault('mao', 'sb'))
names = [1, 3, 234, 987, 3, 2, 0, 19, 0, 0, 2, 2]
names = set(names)
names = list(names)
print(names)
g = {83, 1, 19, 10, 2, 1, 9187654}
print(g)
h = {1123, 19, 83, 9187654}
i = frozenset('poiuytrrhjkjhgfdgyuiofdghuikiuytfgthyu')
print(i)
print(hash(i))
print(2 not in h, h == g, i != g, h <= g, h > g)
print(g & h, g | h, g - h, g ^ h)
print(h.issubset(g), h.issuperset(g), h.union(g), h.intersection(g), g.difference(h), g.symmetric_difference(h), g.copy(), sep = '\n')
g.update(h)
print(g)
j = {19, 10, 0, 9876543}
#j.intersection_update(g)
print(j)
j.difference_update(g)
print(j)
g.symmetric_difference_update(h)
print(g)
g.add(100)
print(g)
g.remove(2)
print(g)
g.discard(0)
g.discard(10)
print(g)
print(g.pop())
print(g)
g.clear()
print(g)
del g

























































