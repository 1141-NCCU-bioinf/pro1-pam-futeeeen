

def compare_matrices(mat1, mat2):
    differences = []
    rows = min(len(mat1), len(mat2))
    for i in range(rows):
        cols = min(len(mat1[i]), len(mat2[i]))
        for j in range(cols):
            if mat1[i][j] != mat2[i][j]:
                differences.append((i, j, mat1[i][j], mat2[i][j]))
    return differences


# 測試
'''
matrix1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

matrix2 = [
    [1, 2, 0],
    [4, 0, 6],
    [7, 8, 9]
]

diffs = compare_matrices(matrix1, matrix2)
'''
