#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define BLOCK_SIZE 64
#define PHY_BLOCK 32
#define min(a, b) a < b ? a : b

__global__ void Phase1(int* dist, int Round, int N);
__global__ void Phase2(int* dist, int Round, int N);
__global__ void Phase3(int* dist, int Round, int N, int offset_y);

//#define DEV_NO 0
//cudaDeviceProp prop;

int main(int argc, char* argv[]) {

    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    const int INF = ((1 << 30) - 1);
    int n, m, N;


    FILE* file = fopen(argv[1], "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    if (n % BLOCK_SIZE == 0) N = n;
    else N = n + BLOCK_SIZE - n % BLOCK_SIZE;

    __align__(8) int* Dist = (int*)malloc(N * N * sizeof(int));
    for (int i = 0; i < N * N; ++i) {
        if (i % (N + 1) == 0) Dist[i] = 0;
        else Dist[i] = INF;
    }

    __align__(8) int* pair = (int*)malloc(3 * m * sizeof(int));
    fread(pair, sizeof(int), 3 * m, file);
    for (int i = 0; i < 3 * m; i += 3) Dist[pair[i] * N + pair[i + 1]] = pair[i + 2];
    free(pair);
    fclose(file);
    printf(" N = %d\n", N);
    cudaHostRegister(Dist, N * N * sizeof(int), cudaHostRegisterDefault);

    const int round = N / BLOCK_SIZE;
    dim3 block_dim(PHY_BLOCK, PHY_BLOCK);

    const int num_threads = 2;
    int* dst[num_threads];

    #pragma omp parallel
    {
        const int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);
        cudaMalloc(&dst[gpu_id], N * N * sizeof(int));

        int round_y = round / 2;
        const int offset_y = gpu_id == 0 ? 0 : round_y;
        if (gpu_id == 1 && round % 2 == 1) round_y++;
        dim3 grid_dim(round, round_y);

        const int row_size = BLOCK_SIZE * N;
        //const int small_row_size = BLOCK_SIZE * n;
        const int halfBlockSize = row_size * round_y;
        const int Offsety_Size = offset_y * row_size;

        cudaMemcpy(dst[gpu_id] + Offsety_Size, Dist + Offsety_Size, halfBlockSize * sizeof(int), cudaMemcpyHostToDevice);

        for (int r = 0; r < round; r++) {
            if (r >= offset_y && r < offset_y + round_y) cudaMemcpy(Dist + r * row_size, dst[gpu_id] + r * row_size, row_size * sizeof(int), cudaMemcpyDeviceToHost);
            #pragma omp barrier
            cudaMemcpy(dst[gpu_id] + r * row_size, Dist + r * row_size, row_size * sizeof(int), cudaMemcpyHostToDevice);
            Phase1 << <1, block_dim >> > (dst[gpu_id], r, N);
            Phase2 << <round - 1, block_dim >> > (dst[gpu_id], r, N);
            Phase3 << <grid_dim, block_dim >> > (dst[gpu_id], r, N, offset_y);
        }
        //cudaMemcpy2D(&Dist + small_row_size * offset_y, n * sizeof(int), &dst[gpu_id][Offsety_Size], N * sizeof(int), n * sizeof(int), round_y * small_row_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(Dist + Offsety_Size, dst[gpu_id] + Offsety_Size, halfBlockSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dst[gpu_id]);
        #pragma omp barrier
    }
    FILE* outfile = fopen(argv[2], "wb");
    //fwrite(&Dist[0], sizeof(int), n * n, outfile);
    for (int i = 0; i < n; i++) fwrite(&Dist[i * N], sizeof(int), n, outfile);
    fclose(outfile);
    //for (int i = 0; i < n * n; i++) printf("%d ", Dist[i/n*N + i%n]);
    cudaFreeHost(Dist);
    return 0;
}

__global__ void Phase1(int* dist, int Round, int N) {
    __shared__ int ROUND_BLOCK[BLOCK_SIZE][BLOCK_SIZE];
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;
    const int offset = BLOCK_SIZE * Round * (N + 1);

    ROUND_BLOCK[i][j] = dist[offset + i * N + j];
    ROUND_BLOCK[i][j2] = dist[offset + i * N + j2];
    ROUND_BLOCK[i2][j] = dist[offset + i2 * N + j];
    ROUND_BLOCK[i2][j2] = dist[offset + i2 * N + j2];
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        ROUND_BLOCK[i][j] = min(ROUND_BLOCK[i][j], ROUND_BLOCK[i][k] + ROUND_BLOCK[k][j]);
        ROUND_BLOCK[i][j2] = min(ROUND_BLOCK[i][j2], ROUND_BLOCK[i][k] + ROUND_BLOCK[k][j2]);
        ROUND_BLOCK[i2][j] = min(ROUND_BLOCK[i2][j], ROUND_BLOCK[i2][k] + ROUND_BLOCK[k][j]);
        ROUND_BLOCK[i2][j2] = min(ROUND_BLOCK[i2][j2], ROUND_BLOCK[i2][k] + ROUND_BLOCK[k][j2]);
    }

    dist[offset + i * N + j] = ROUND_BLOCK[i][j];
    dist[offset + i * N + j2] = ROUND_BLOCK[i][j2];
    dist[offset + i2 * N + j] = ROUND_BLOCK[i2][j];
    dist[offset + i2 * N + j2] = ROUND_BLOCK[i2][j2];
}

__global__ void Phase2(int* dist, int Round, int N) {
    __shared__ int ROUND_BLOCK[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int ROW[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int COL[BLOCK_SIZE][BLOCK_SIZE];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;
    const int offset = BLOCK_SIZE * Round;
    const int block_i = blockIdx.x >= Round ? (blockIdx.x + 1) * BLOCK_SIZE : blockIdx.x * BLOCK_SIZE;

    ROW[i][j] = dist[block_i * N + offset + i * N + j];
    ROW[i][j2] = dist[block_i * N + offset + i * N + j2];
    ROW[i2][j] = dist[block_i * N + offset + i2 * N + j];
    ROW[i2][j2] = dist[block_i * N + offset + i2 * N + j2];

    ROUND_BLOCK[i][j] = dist[offset * (N + 1) + i * N + j];
    ROUND_BLOCK[i][j2] = dist[offset * (N + 1) + i * N + j2];
    ROUND_BLOCK[i2][j] = dist[offset * (N + 1) + i2 * N + j];
    ROUND_BLOCK[i2][j2] = dist[offset * (N + 1) + i2 * N + j2];

    COL[i][j] = dist[offset * N + block_i + i * N + j];
    COL[i][j2] = dist[offset * N + block_i + i * N + j2];
    COL[i2][j] = dist[offset * N + block_i + i2 * N + j];
    COL[i2][j2] = dist[offset * N + block_i + i2 * N + j2];
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        ROW[i][j] = min(ROW[i][j], ROW[i][k] + ROUND_BLOCK[k][j]);
        ROW[i][j2] = min(ROW[i][j2], ROW[i][k] + ROUND_BLOCK[k][j2]);
        ROW[i2][j] = min(ROW[i2][j], ROW[i2][k] + ROUND_BLOCK[k][j]);
        ROW[i2][j2] = min(ROW[i2][j2], ROW[i2][k] + ROUND_BLOCK[k][j2]);

        COL[i][j] = min(COL[i][j], ROUND_BLOCK[i][k] + COL[k][j]);
        COL[i][j2] = min(COL[i][j2], ROUND_BLOCK[i][k] + COL[k][j2]);
        COL[i2][j] = min(COL[i2][j], ROUND_BLOCK[i2][k] + COL[k][j]);
        COL[i2][j2] = min(COL[i2][j2], ROUND_BLOCK[i2][k] + COL[k][j2]);
    }

    dist[block_i * N + offset + i * N + j] = ROW[i][j];
    dist[block_i * N + offset + i * N + j2] = ROW[i][j2];
    dist[block_i * N + offset + i2 * N + j] = ROW[i2][j];
    dist[block_i * N + offset + i2 * N + j2] = ROW[i2][j2];

    dist[offset * N + block_i + i * N + j] = COL[i][j];
    dist[offset * N + block_i + i * N + j2] = COL[i][j2];
    dist[offset * N + block_i + i2 * N + j] = COL[i2][j];
    dist[offset * N + block_i + i2 * N + j2] = COL[i2][j2];
}

__global__ void Phase3(int* dist, int Round, int N, int offset_y) {
    __shared__ int ROW[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int COL[BLOCK_SIZE][BLOCK_SIZE];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;
    const int block_i = (blockIdx.y + offset_y) * BLOCK_SIZE;
    const int block_j = blockIdx.x * BLOCK_SIZE;
    //const int block_i = (blockIdx.y + offset_y) >= Round ? (blockIdx.y + offset_y + 1) * BLOCK_SIZE : (blockIdx.y + offset_y) * BLOCK_SIZE;
    //const int block_j = blockIdx.x >= Round ? (blockIdx.x + 1) * BLOCK_SIZE : (blockIdx.x) * BLOCK_SIZE;
    const int offset = BLOCK_SIZE * Round;
    int L1, L2, L3, L4;

    if (block_i / BLOCK_SIZE == Round || block_j / BLOCK_SIZE == Round) return;

    ROW[i][j] = dist[block_i * N + offset + i * N + j];
    ROW[i][j2] = dist[block_i * N + offset + i * N + j2];
    ROW[i2][j] = dist[block_i * N + offset + i2 * N + j];
    ROW[i2][j2] = dist[block_i * N + offset + i2 * N + j2];

    COL[i][j] = dist[offset * N + block_j + i * N + j];
    COL[i][j2] = dist[offset * N + block_j + i * N + j2];
    COL[i2][j] = dist[offset * N + block_j + i2 * N + j];
    COL[i2][j2] = dist[offset * N + block_j + i2 * N + j2];

    L1 = dist[block_i * N + block_j + i * N + j];
    L2 = dist[block_i * N + block_j + i * N + j2];
    L3 = dist[block_i * N + block_j + i2 * N + j];
    L4 = dist[block_i * N + block_j + i2 * N + j2];
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        L1 = min(L1, ROW[i][k] + COL[k][j]);
        L2 = min(L2, ROW[i][k] + COL[k][j2]);
        L3 = min(L3, ROW[i2][k] + COL[k][j]);
        L4 = min(L4, ROW[i2][k] + COL[k][j2]);
    }

    dist[block_i * N + block_j + i * N + j] = L1;
    dist[block_i * N + block_j + i * N + j2] = L2;
    dist[block_i * N + block_j + i2 * N + j] = L3;
    dist[block_i * N + block_j + i2 * N + j2] = L4;
}
