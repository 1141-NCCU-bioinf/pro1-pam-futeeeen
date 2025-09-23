import hw1
import numpy as np
import compare_matrices

hw1.generate_pam(
    x=250, 
    input_path="example/mut.txt", 
    output_path="example/pam250_out.txt"
)

# 讀檔案
matrix = np.loadtxt("example/pam250.txt", skiprows=1, usecols=range(1, 21))
PAMx = np.loadtxt("example/pam250_out.txt", skiprows=1, usecols=range(1, 21))

diffs = compare_matrices.compare_matrices(
    mat1 = matrix,
    mat2 = PAMx
)
for d in diffs:
    print(f"位置 (row={d[0]}, col={d[1]}) 不同: {d[2]} vs {d[3]}")