# 1. Function to count vowels and consonants in a string
def count_vowels_and_consonants(input_string):
    vowel_count = 0
    consonant_count = 0
    index = 0
    while True:
        try:
            character = input_string[index]
        except IndexError:
            break

        # Check if character is an alphabet
        if ('a' <= character <= 'z') or ('A' <= character <= 'Z'):
            if character in 'aeiouAEIOU':
                vowel_count += 1
            else:
                consonant_count += 1
        index += 1
    return vowel_count, consonant_count


# 2. Function to multiply two matrices
def multiply_two_matrices(matrix_a, matrix_b):
    rows_a = 0
    while True:
        try:
            _ = matrix_a[rows_a]
            rows_a += 1
        except IndexError:
            break

    cols_a = 0
    while True:
        try:
            _ = matrix_a[0][cols_a]
            cols_a += 1
        except IndexError:
            break

    rows_b = 0
    while True:
        try:
            _ = matrix_b[rows_b]
            rows_b += 1
        except IndexError:
            break

    cols_b = 0
    while True:
        try:
            _ = matrix_b[0][cols_b]
            cols_b += 1
        except IndexError:
            break

    if cols_a != rows_b:
        return None 

    result_matrix = []
    i = 0
    while i < rows_a:
        result_row = []
        j = 0
        while j < cols_b:
            cell_sum = 0
            k = 0
            while k < cols_a:
                cell_sum += matrix_a[i][k] * matrix_b[k][j]
                k += 1
            result_row.append(cell_sum)
            j += 1
        result_matrix.append(result_row)
        i += 1

    return result_matrix


# 3. Function to count common elements between two lists
def count_common_elements_in_lists(list_one, list_two):
    common_count = 0
    index_one = 0
    while True:
        try:
            item = list_one[index_one]
        except IndexError:
            break

        index_two = 0
        found = False
        while True:
            try:
                if item == list_two[index_two]:
                    found = True
                    break
                index_two += 1
            except IndexError:
                break

        if found:
            common_count += 1
        index_one += 1

    return common_count


# 4. Function to transpose a matrix
def transpose_of_matrix(original_matrix):
    row_count = 0
    while True:
        try:
            _ = original_matrix[row_count]
            row_count += 1
        except IndexError:
            break

    column_count = 0
    while True:
        try:
            _ = original_matrix[0][column_count]
            column_count += 1
        except IndexError:
            break

    transposed_matrix = []
    i = 0
    while i < column_count:
        new_row = []
        j = 0
        while j < row_count:
            new_row.append(original_matrix[j][i])
            j += 1
        transposed_matrix.append(new_row)
        i += 1

    return transposed_matrix




# Test for vowel and consonant count
sample_string = "Hello World"
vowels, consonants = count_vowels_and_consonants(sample_string)
print("String:", sample_string)
print("Vowels:", vowels)
print("Consonants:", consonants)

# Test for matrix multiplication
matrix_A = [[1, 2], [3, 4]]
matrix_B = [[5, 6], [7, 8]]
product = multiply_two_matrices(matrix_A, matrix_B)
print("\nMatrix A:", matrix_A)
print("Matrix B:", matrix_B)
if product is None:
    print("Error: Matrices cannot be multiplied due to dimension mismatch.")
else:
    print("Product of A and B (Single Matrix):")
    for row in product:
        print(row)

# Test for counting common elements
list1 = [1, 2, 3, 4]
list2 = [3, 4, 5, 6]
common_count = count_common_elements_in_lists(list1, list2)
print("\nList 1:", list1)
print("List 2:", list2)
print("Number of common elements:", common_count)

# Test for matrix transpose
original_matrix = [[1, 2, 3], [4, 5, 6]]
transposed = transpose_of_matrix(original_matrix)
print("\nOriginal Matrix:")
for row in original_matrix:
    print(row)
print("Transposed Matrix (Single Matrix):")
for row in transposed:
    print(row)
