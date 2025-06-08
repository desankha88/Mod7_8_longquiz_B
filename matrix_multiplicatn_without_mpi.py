import numpy as np
import time

N=1024

def create_matrix_from_user_input():
    m = int(input('enter rowsize of matrix: '))
    n = int(input('enter num of columns in matrix: '))
    matrix = []
    for i in range(m):
        row = []
        for j in range(n):
            elmnt = int(input(f'enter the elements in {i} row:'))
            row.append(elmnt)
        matrix.append(row)
    print(matrix)
    return matrix

def initialize_matrix_without_input():
    matrix = np.fromfunction(lambda i, j: i + j, (N, N), dtype=float)
    return matrix

def multiply_two_square_matrices(matrix, matrix2) :
    np_matrix1 = np.array(matrix)
    np_matrix2 = np.array(matrix2)
    if np_matrix1.shape[-1] != np_matrix2.shape[0]:
        print('multiplication not possible')
        quit(0)
    else:
        print('Matrixes can be multiplied')
        res = [[0] * np_matrix2.shape[1] for _ in range(np_matrix1.shape[0])]
        print('result matrix shape will be :',len(res),len(res[0]))
        #print(np_matrix1.dot(np_matrix2))

    for i in range(len(matrix)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                #print(f'i={i} j={j} k={k} | matrix1[{i}][{k}] :', matrix[i][k],f'matrix2[{k}][{j}]:',matrix2[k][j])
                res[i][j] += matrix[i][k] * matrix2[k][j]
    return res


if __name__ == '__main__':
    #matrix1 = create_matrix_from_user_input()
    matrix1 = initialize_matrix_without_input()
    #matrix2 = create_matrix_from_user_input()
    matrix2 = initialize_matrix_without_input()
    start_time = time.time()
    mat_res = multiply_two_square_matrices(matrix1,matrix2)
    end_time = time.time()
    time_loops = end_time - start_time
    print(f"Loop-based matmul time: {time_loops:.6f} seconds")
    print(np.array(mat_res))
    print('Validate results - Expected output calculated via numpy dot product :')
    print(np.array(matrix1).dot(np.array(matrix2)))
    with open("benchmarking2.csv",'w') as f :
        f.write('matrix_size,time_duration_secs,MPI,num_cores\n')
        f.write(f'{N},{time_loops:.6f},N,1\n')
