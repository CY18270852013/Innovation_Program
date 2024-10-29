def foo(num, base):
    if num >= base:
        foo(num // base, base)
    print(num % base, end = '')

num = eval(input())
base = eval(input())
foo(num, base)