count = 0
import math
def isprime(x):
    if x == 1:
        return 0
    if x == 2:
        return 1
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return 0
    return 1

m = []
i = 1
while count < 5:
    if isprime(i) and isprime(2 ** i - 1):
        m.append(2 ** i - 1)
        count += 1
    i += 1
print(m)