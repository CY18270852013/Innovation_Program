rate = eval(input('Enter the exchange rate from dollars to RMB:'))
flag = eval(input('Enter 0 to convert dollars to RMB and 1 vice versa:'))
if flag == 0:
    dollar = eval(input('Enter the dollar amount:'))
    print('$ {0} is {1} yuan'.format(float(dollar), rate * dollar))
elif flag == 1:
    rmb = eval(input('Enter the RMB amount:'))
    print('{0} yuan is $ {1}'.format(float(rmb), rmb / rate))
else:
    print('Incorrect input')