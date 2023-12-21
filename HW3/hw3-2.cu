#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_pipeline_primitives.h>
//#include <chrono>

#define BLOCK_SIZE 64
#define PHY_BLOCK 32
#define min(a, b) a < b ? a : b

__global__ void Phase1(int* dist, int Round, int n);
__global__ void Phase2(int* dist, int Round, int n);
__global__ void Phase3(int* dist, int Round, int n);

//clock_t begin, end;
//#define DEV_NO 0
//cudaDeviceProp prop;

int main(int argc, char* argv[]) {

    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    //double IO = 0;

    const int INF = ((1 << 30) - 1);
    int n, m, N;

    //begin = clock();
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
    //end = clock();
    //IO += (double)(end - begin) / CLOCKS_PER_SEC;

    for (int i = 0; i < 3 * m; i += 3) Dist[pair[i] * N + pair[i + 1]] = pair[i + 2];

    __align__(8) int* dst = NULL;
    cudaHostRegister(Dist, N * N * sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&dst, N * N * sizeof(int));

    cudaMemcpyAsync(dst, Dist, N * N * sizeof(int), cudaMemcpyHostToDevice);
    const int round = N / BLOCK_SIZE;
    dim3 block_dim(PHY_BLOCK, PHY_BLOCK);
    dim3 grid_dim(round - 1, round - 1);

    for (int r = 0; r < round; ++r) {
        Phase1 << < 1, block_dim >> > (dst, r, N);
        Phase2 << < round - 1, block_dim >> > (dst, r, N);
        Phase3 << < grid_dim, block_dim >> > (dst, r, N);
    }

    free(pair);
    fclose(file);
    FILE* outfile = fopen(argv[2], "wb");

    cudaMemcpy2D(&Dist[0], n * sizeof(int), dst, N * sizeof(int), n * sizeof(int), n, cudaMemcpyDeviceToHost);

    //begin = clock();
    fwrite(&Dist[0], sizeof(int), n * n, outfile);
    fclose(outfile);
    //end = clock();
    //IO += (double)(end - begin) / CLOCKS_PER_SEC;
    //printf("IO = %.3lf\n", IO);

    cudaFreeHost(Dist);
    cudaFree(dst);
    return 0;
}

__global__ void Phase1(int* dist, int Round, int N) {
    __shared__ int ROUND_BLOCK[BLOCK_SIZE][BLOCK_SIZE];
    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;
    const int offset = BLOCK_SIZE * Round * (N + 1) + i * N + j;

    ROUND_BLOCK[i][j] = dist[offset];
    ROUND_BLOCK[i][j2] = dist[offset + PHY_BLOCK];
    ROUND_BLOCK[i2][j] = dist[offset + PHY_BLOCK * N];
    ROUND_BLOCK[i2][j2] = dist[offset + PHY_BLOCK * (N + 1)];
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        ROUND_BLOCK[i][j] = min(ROUND_BLOCK[i][j], ROUND_BLOCK[i][k] + ROUND_BLOCK[k][j]);
        ROUND_BLOCK[i][j2] = min(ROUND_BLOCK[i][j2], ROUND_BLOCK[i][k] + ROUND_BLOCK[k][j2]);
        ROUND_BLOCK[i2][j] = min(ROUND_BLOCK[i2][j], ROUND_BLOCK[i2][k] + ROUND_BLOCK[k][j]);
        ROUND_BLOCK[i2][j2] = min(ROUND_BLOCK[i2][j2], ROUND_BLOCK[i2][k] + ROUND_BLOCK[k][j2]);
        __syncthreads();
    }

    dist[offset] = ROUND_BLOCK[i][j];
    dist[offset + PHY_BLOCK] = ROUND_BLOCK[i][j2];
    dist[offset + PHY_BLOCK * N] = ROUND_BLOCK[i2][j];
    dist[offset + PHY_BLOCK * (N + 1)] = ROUND_BLOCK[i2][j2];
}

__global__ void Phase2(int* dist, int Round, int N) {
    __shared__ int ROUND_BLOCK[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int ROW[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int COL[BLOCK_SIZE][BLOCK_SIZE];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int block_i = blockIdx.x >= Round ? (blockIdx.x + 1) * BLOCK_SIZE : blockIdx.x * BLOCK_SIZE;

    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;
    const int ROUND_offset = BLOCK_SIZE * Round * (N + 1) + i * N + j;
    const int ROW_offset = block_i * N + BLOCK_SIZE * Round + i * N + j;
    const int COL_offset = BLOCK_SIZE * Round * N + block_i + i * N + j;

    __pipeline_memcpy_async(&ROW[i][j], &dist[ROW_offset], sizeof(int));
    __pipeline_memcpy_async(&ROW[i][j2], &dist[ROW_offset + PHY_BLOCK], sizeof(int));
    __pipeline_memcpy_async(&ROW[i2][j], &dist[ROW_offset + PHY_BLOCK * N], sizeof(int));
    __pipeline_memcpy_async(&ROW[i2][j2], &dist[ROW_offset + PHY_BLOCK * (N + 1)], sizeof(int));

    __pipeline_memcpy_async(&ROUND_BLOCK[i][j], &dist[ROUND_offset], sizeof(int));
    __pipeline_memcpy_async(&ROUND_BLOCK[i][j2], &dist[ROUND_offset + PHY_BLOCK], sizeof(int));
    __pipeline_memcpy_async(&ROUND_BLOCK[i2][j], &dist[ROUND_offset + PHY_BLOCK * N], sizeof(int));
    __pipeline_memcpy_async(&ROUND_BLOCK[i2][j2], &dist[ROUND_offset + PHY_BLOCK * (N + 1)], sizeof(int));

    __pipeline_memcpy_async(&COL[i][j], &dist[COL_offset], sizeof(int));
    __pipeline_memcpy_async(&COL[i][j2], &dist[COL_offset + PHY_BLOCK], sizeof(int));
    __pipeline_memcpy_async(&COL[i2][j], &dist[COL_offset + PHY_BLOCK * N], sizeof(int));
    __pipeline_memcpy_async(&COL[i2][j2], &dist[COL_offset + PHY_BLOCK * (N + 1)], sizeof(int));

    __syncthreads();

    #pragma unroll 32
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

    dist[ROW_offset] = ROW[i][j];
    dist[ROW_offset + PHY_BLOCK] = ROW[i][j2];
    dist[ROW_offset + PHY_BLOCK * N] = ROW[i2][j];
    dist[ROW_offset + PHY_BLOCK * (N + 1)] = ROW[i2][j2];

    dist[COL_offset] = COL[i][j];
    dist[COL_offset + PHY_BLOCK] = COL[i][j2];
    dist[COL_offset + PHY_BLOCK * N] = COL[i2][j];
    dist[COL_offset + PHY_BLOCK * (N + 1)] = COL[i2][j2];
}

__global__ void Phase3(int* dist, int Round, int N) {
    __shared__ int ROW[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int COL[BLOCK_SIZE][BLOCK_SIZE];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int block_i = blockIdx.y >= Round ? (blockIdx.y + 1) * BLOCK_SIZE : blockIdx.y * BLOCK_SIZE;
    const int block_j = blockIdx.x >= Round ? (blockIdx.x + 1) * BLOCK_SIZE : blockIdx.x * BLOCK_SIZE;
    int L1, L2, L3, L4;

    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;
    const int ROW_offset = block_i * N + BLOCK_SIZE * Round + i * N + j;
    const int COL_offset = BLOCK_SIZE * Round * N + block_j + i * N + j;
    const int CUR_offset = block_i * N + block_j + i * N + j;

    __pipeline_memcpy_async(&ROW[i][j], &dist[ROW_offset], sizeof(int));
    __pipeline_memcpy_async(&ROW[i][j2], &dist[ROW_offset + PHY_BLOCK], sizeof(int));
    __pipeline_memcpy_async(&ROW[i2][j], &dist[ROW_offset + PHY_BLOCK * N], sizeof(int));
    __pipeline_memcpy_async(&ROW[i2][j2], &dist[ROW_offset + PHY_BLOCK * (N + 1)], sizeof(int));

    __pipeline_memcpy_async(&COL[i][j], &dist[COL_offset], sizeof(int));
    __pipeline_memcpy_async(&COL[i][j2], &dist[COL_offset + PHY_BLOCK], sizeof(int));
    __pipeline_memcpy_async(&COL[i2][j], &dist[COL_offset + PHY_BLOCK * N], sizeof(int));
    __pipeline_memcpy_async(&COL[i2][j2], &dist[COL_offset + PHY_BLOCK * (N + 1)], sizeof(int));
    
    L1 = dist[CUR_offset];
    L2 = dist[CUR_offset + PHY_BLOCK];
    L3 = dist[CUR_offset + PHY_BLOCK * N];
    L4 = dist[CUR_offset + PHY_BLOCK * (N + 1)];

    __syncthreads();

    #pragma unroll 32
    for (int k = 0; k < BLOCK_SIZE; k++) {
        L1 = min(L1, ROW[i][k] + COL[k][j]);
        L2 = min(L2, ROW[i][k] + COL[k][j2]);
        L3 = min(L3, ROW[i2][k] + COL[k][j]);
        L4 = min(L4, ROW[i2][k] + COL[k][j2]);
    }

    dist[CUR_offset] = L1;
    dist[CUR_offset + PHY_BLOCK] = L2;
    dist[CUR_offset + PHY_BLOCK * N] = L3;
    dist[CUR_offset + PHY_BLOCK * (N + 1)] = L4;
}
