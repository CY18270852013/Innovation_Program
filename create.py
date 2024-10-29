import random
import math
l = random.sample(range(100), 2)
print(l)
m = min(l)
M = max(l)

def isprime(x):
    if x == 1:
        return 0
    if x == 2:
        return 1
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return 0
    return 1

y = []
for i in range(2, m + 1):
    if m % i == 0:
        y.append(i)
for num in y:
    if M % num == 0:
        print('这两个数不互质！')
        break
else:
    print('这两个数互质！')