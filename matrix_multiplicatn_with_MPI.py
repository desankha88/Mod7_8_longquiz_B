from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1024

if rank == 0:
    A = np.fromfunction(lambda i, j: i + j, (N, N), dtype=float)
    B = np.fromfunction(lambda i, j: i + j, (N, N), dtype=float)
else:
    A = None
    B = np.empty((N, N), dtype=float)

# Broadcast matrix B
comm.Bcast(B, root=0)

# Scatter matrix A rows
rows_per_proc = N // size
local_A = np.empty((rows_per_proc, N), dtype=float)
comm.Scatter(A, local_A, root=0)

comm.Barrier()
start = time.time()

local_C_loops = np.zeros((rows_per_proc, N), dtype=float)

for i in range(rows_per_proc):
    for j in range(N):
        sum_ = 0.0
        for k in range(N):
            sum_ += local_A[i][k] * B[k][j]
        local_C_loops[i][j] = sum_

comm.Barrier()
end = time.time()
time_loops = end - start

if rank == 0:
    print(f"Loop-based matmul time with MPI: {time_loops:.6f} seconds")
    with open('benchmarking2.csv','a') as f1:
        f1.write(f'512,{time_loops:.6f},Y,{size}\n')
