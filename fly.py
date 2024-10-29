def isprime(x):
    if x == 2:
        return 1
    for i in range(2, x):
        if x % i == 0:
            return 0
    else:
        return 1

for i in range(1, 10):
    for j in range(0, 10):
        for k in range(0, 10):
            n = i * 100 + j * 10 + k
            if isprime(n) and (j + k) % 10 == i:
                print(n)
            