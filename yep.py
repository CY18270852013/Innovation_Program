def count_str(s):
    l = s.split(' ')
    d = {}
    for word in l:
        d[word] = d.get(word, 0) + 1
    return d

if __name__ == '__main__':
    s = input()
    d = count_str(s)
    d = sorted(d.items(), key = lambda x: x[0])
    for key, value in d:
        print(key, value)
    d = sorted(d, key = lambda x: x[1])
    for key, value in d:
        print(key, value)