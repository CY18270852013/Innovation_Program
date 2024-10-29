n = 0
for i in range(1, 5):
    for j in range(1, 5):
        for k in range(1, 5):
            print(i * 100 + j * 10 + k)
            n += 1
print('个数为{}'.format(n))