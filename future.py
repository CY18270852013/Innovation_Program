import numpy as np

# Define a scalar function to calculate the sum of digits of a numeric string
def sum_of_digits(s: str) -> int:
    return sum(int(digit) for digit in s)

# Vectorize the scalar function
vectorized_sum_of_digits = np.vectorize(sum_of_digits)

# Create an ndarray of integers from 1 to 1000
array = np.arange(1, 1001)

# Apply the vectorized function to the ndarray
result = vectorized_sum_of_digits(array.astype(str))

print("Resulting array:")
print(result)