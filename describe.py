import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(2, 1)

ax0.plot(range(7), [3, 4, 7, 6, 2, 8, 9], color = 'r', marker = 'o')
ax0.set_title('subplot1')
plt.xlabel('yes')
plt.ylabel('no')

plt.subplots_adjust(hspace = 0.5)
ax1.plot(range(7), [5, 1, 8, 2, 6, 9, 4], color = 'green', marker = 'o')
ax1.set_title('subplot2')