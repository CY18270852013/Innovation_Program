f = {'li', 'wang', 'jiang', 'qian', 'wan', 'zhang', 'liu', 'qiu', 'zhao'}
b = {'yao', 'zhou', 'zhang', 'ren', 'guo', 'chen', 'fan', 'liu', 'wang'}
print(f & b)
print(len(f & b))
print((f | b) - (f & b))
print(len((f | b) - (f & b)))