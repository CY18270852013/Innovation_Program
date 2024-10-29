count = 0
for i in range(1, 10):
    for j in range(0, 10):
        for k in range(0, 10):
            n1 = i * 100 + j * 10 + k
            if n1 % 37 == 0:
                n2 = j * 100 + k * 10 + i
                n3 = k * 100 + i * 10 + j
                if n2 % 37 != 0 or n3 % 37 != 0:
                    count += 1

if count == 0:
    print('这是一个真命题')
else:
    print('这是一个假命题')