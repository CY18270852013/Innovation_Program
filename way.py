y = input()
l = []

for word in y:
    l.append(word)
x = int(''.join(l))
l1 = []
l2 = []
if len(set(l)) == 1:
    print('{0}-{1}=0000'.format(x, x))
else:
    while x != 6174:
        l1 = sorted(l)
        x1 = int(''.join(l1))
        l2 = reversed(l1)
        x2 = int(''.join(l2))
        x = x2 - x1
        print('{0:0>4d}-{1:0>4d}={2:0>4d}'.format(x2, x1, x))
        l = []
        for w in str(x):
            l.append(w)
        if x < 1000:
            l.append(str(0))