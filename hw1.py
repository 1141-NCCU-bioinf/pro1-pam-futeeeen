import numpy as np


def generate_pam(x, input_path, output_path):
    # 讀檔案
    matrix = np.loadtxt(input_path, skiprows=2, usecols=range(1, 21))
    M = matrix/10000
    
    # 背景頻率
    frequencies = {
        'A': 0.087, 'R': 0.041, 'N': 0.040, 'D': 0.047, 'C': 0.033,
        'Q': 0.038, 'E': 0.050, 'G': 0.089, 'H': 0.034, 'I': 0.037,
        'L': 0.085, 'K': 0.081, 'M': 0.015, 'F': 0.040, 'P': 0.051,
        'S': 0.070, 'T': 0.058, 'W': 0.010, 'Y': 0.030, 'V': 0.065
    }
    # 取排序
    with open(input_path, "r") as f_m1:
        lines = f_m1.readlines()
        order = lines[1].strip().split()
    # 進行排序
    freq = np.array([frequencies[aa] for aa in order])
    #freq = freq/sum(freq)
    print(order)

    '''
    # 計算發生突變的總次數
    # 每列合計
    n_j = np.sum(matrix, axis=0)  # sum down rows for each column j

    # 取對角線值以外的欄位(i ≠ j)
    # 每列合計 - 對角線上的值 = 發生突變的總次數
    # 突變率 = 發生突變的次數/總次數
    diag = np.diag(matrix)
    off_diag_sum = n_j - diag
    m = off_diag_sum / n_j

    # 計算 lambda，M1 99%不變，1%突變
    target_mutation_rate = 0.01
    # sum over j of f[j] * m[j]
    weighted = np.dot(freq, m)  # this is Σ_j f_j * m_j
    lam = target_mutation_rate / weighted
 
    # 建立 PAM1 transition 矩陣 M
    M = np.zeros_like(matrix, dtype=float)
    # 非對角
    for j in range(len(freq)):
        for i in range(len(freq)):
            if i != j:
                M[i, j] = lam * (matrix[i, j] / n_j[j])
        # 對角
        M[j, j] = 1.0 - lam * m[j]
    '''
    # PAMx
    PAMx = np.linalg.matrix_power(M, x)
    
    ##print(PAMx.sum(axis=0))
  
    # log-odds score
    with np.errstate(divide='ignore', invalid='ignore'):
    
        ratio = PAMx / freq[:, None]
        
        score_matrix = 10 * np.log10(ratio)
        # ChatGPT > 可選：將無定義或 -inf 的地方設成很小的值或某個預設 score


    # 產檔案
    with open(output_path, "w") as f_out:
        f_out.write("   " + " ".join(order) + "\n")
        for i, aa in enumerate(order):
            row = " ".join(f"{int(np.round(val)):3d}" for val in score_matrix[i])
            f_out.write(f"{aa} {row}\n")
   
    #idx_R = order.index('R')
    #idx_V = order.index('Y')

    #p_RV = PAMx[idx_R, idx_V]
    #fR = freq[idx_R]
    #ratio_RV = p_RV / fR
    #score_float = 10 * np.log10(ratio_RV)
       
    #print("DEBUG R->Y:")
    #print("PAMx[R,Y] =", p_RV)
    #print("freq[R] =", fR)
    #print("ratio =", ratio_RV)
    #print("score (float) =", score_float)
    #print("rounded score =", np.round(score_float))

