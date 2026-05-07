import numpy as np

# Q1. Reverse a NumPy array
print("Q1. Reverse a NumPy Array")
arr = np.array([1, 2, 3, 6, 4, 5])
print("Original array:", arr)
reversed_arr = arr[::-1]
print("Reversed array:", reversed_arr)

print()

# Q2. Compare two NumPy arrays
print("Q2. Compare Two NumPy Arrays")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
print("Array 1:\n", arr1)
print("Array 2:\n", arr2)
print("Element-wise comparison (arr1 == arr2):\n", arr1 == arr2)
print("Arrays are equal:", np.array_equal(arr1, arr2))

print()

# Q3. Find the most frequent value and their indices
print("Q3. Most Frequent Value and Indices")

# Part i
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
vals_x, counts_x = np.unique(x, return_counts=True)
most_freq_x = vals_x[np.argmax(counts_x)]
indices_x = np.where(x == most_freq_x)[0]
print("Array x:", x)
print(f"Most frequent value: {most_freq_x} (appears {np.max(counts_x)} times)")
print("Indices:", indices_x)

print()

# Part ii
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
vals_y, counts_y = np.unique(y, return_counts=True)
most_freq_y = vals_y[np.argmax(counts_y)]
indices_y = np.where(y == most_freq_y)[0]
print("Array y:", y)
print(f"Most frequent value: {most_freq_y} (appears {np.max(counts_y)} times)")
print("Indices:", indices_y)

print()

# Q4. Matrix operations
print("Q4. Matrix Sum Operations")
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print("Matrix gfg:\n", gfg)
print("Sum of all elements:", np.sum(gfg))
print("Sum row-wise:\n", np.sum(gfg, axis=1))
print("Sum column-wise:\n", np.sum(gfg, axis=0))
