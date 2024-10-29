s = input()
result = [0] * 26
s = s.lower()
s = ''.join(filter(str.isalpha, s))
for i in s:
    result[ord(i) - 97] += 1
print(result)
