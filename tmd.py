def clean_list(coffee):
    for i in range(len(coffee)):
        coffee[i] = ''.join(filter(str.isalpha, coffee[i]))
    return coffee  
if __name__ == '__main__':
    coffee = ['32Latte', '_Americano30', '/34Cappuccino', 'Mocha35']      
    coffee = clean_list(coffee)
    for i in range(1, len(coffee) + 1):
        print('{0} {1}'.format(i, coffee[i - 1]))
