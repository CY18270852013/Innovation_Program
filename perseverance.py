minutes = eval(input())
years = minutes // (365 * 24 * 60)
m1 = minutes % (365 * 24 * 60)
days = m1 // (24 * 60)
m2 = m1 % (24 * 60)
hours = m2 // 60
m3 = m2 % 60
print('years = {0}, days = {1}, hours = {2}, minutes = {3}'.format(years, days, hours, m3))