def isIdentifier(s):
    if ord(s[0].lower()) < 48 or (ord(s[0].lower()) > 57 and ord(s[0].lower()) < 97) or ord(s[0].lower()) > 123:
        print('Error. First char must be alphas or number.')
    else:
        flag = 1
        for i in range(1, len(s)):
            if ord(s[i].lower()) != '_' and (ord(s[i].lower()) < 48 or (ord(s[i].lower()) > 57 and ord(s[i].lower()) < 97) or ord(s[i].lower()) > 123):
                flag = 0
                break
        if flag == 1:
            print('Valid identifier.')
        else:
            print('Error. Other chars must be alphas number or _.')

if __name__ == '__main__':
    s = input()
    isIdentifier(s)