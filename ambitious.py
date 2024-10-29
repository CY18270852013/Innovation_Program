from functools import reduce
l = [3, 4, 5, 2, 7, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 987654]
print(list(map(lambda x: x**2, l)))
print(list(filter(lambda x: x % 2 == 0, l)))
print(reduce(lambda x, y: x + y, l))


def g(args1, *names):
    print(args1)
    print(names)
names = ['w', 'l', 'j']
g('h', *names)

x = 3
def f(x):
    #global x
    print(x ** 2)
    x= 5
    print(x ** 2)
f(x)

def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment()
g = counter()
print(g)
        