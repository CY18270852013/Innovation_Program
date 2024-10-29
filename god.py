def max_product(s, n):
    max_product = 0
    max_digits = ""
    for i in range(len(s)-n+1):
        digits = s[i:i+n]  # 枚举连续n个数字
        product = 1
        for digit in digits:
            product *= int(digit)  # 计算乘积
        if product > max_product:  # 更新最大乘积
            max_product = product
            max_digits = digits
    print(f"乘积最大的连续{n}个数字是{max_digits}，乘积为{max_product}")      

s = input()
n = eval(input())

max_product(s, n)