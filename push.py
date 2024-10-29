n = eval(input())
l = []
for i in range(1, 101):
    if i % n != 0 and str(n) not in str(i):
        l.append(i)
num = len(l)
count = 1
for i in l:
    if count % 10 != 0 and count != num:
        print(i, end = ',')
        count += 1
    else:
        print(i)
        count += 1