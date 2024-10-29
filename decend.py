import numpy as np

# Create a 5x4 2D array with random numbers from U[-10, 10)
array = np.random.uniform(-10, 10, (5, 4))

# Find the two largest and two smallest numbers and set them to 0
largest_indices = np.argpartition(array.flatten(), -2)[-2:]
smallest_indices = np.argpartition(array.flatten(), 2)[:2]
array[np.unravel_index(largest_indices, array.shape)] = 0
array[np.unravel_index(smallest_indices, array.shape)] = 0

# Count the number of elements with an absolute value between 4 and 6
count = np.sum((4 < np.abs(array)) & (np.abs(array) < 6))

print("Modified array:")
print(array)
print("Count of elements with absolute value between 4 and 6:", count)