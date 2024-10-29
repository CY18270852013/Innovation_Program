def playCard(l):
    l = list(set(l))
    l.sort()
    count = 0
    for i in l:
        if i == len(l) - 4:
            break
        if i + 1 in l:
            if i + 2 in l:
                if i + 3 in l:
                    if i + 4 in l:
                        print(i, i + 1, i + 2, i + 3, i + 4, sep = ',')
                        count += 1
    if count == 0:
        print('Not exist.')
        
if __name__ == '__main__':
    l = list(eval(input()))
    playCard(l)