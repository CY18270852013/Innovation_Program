from math import sqrt
a, b, c = eval(input())
if a + b > c and a + c > b and b + c > a:
    C = a + b + c
    s1 = C / 2
    s2 = s1 - a
    s3 = s1 - b
    s4 = s1 - c
    S = sqrt(s1 * s2 * s3 * s4)
    print(C, S)
else:
    print('Wrong input.')