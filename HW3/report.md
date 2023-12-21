# HW3 All-Pairs Shortest Path
張瀚 / 111064528
---
## Implementation
1. Which algorithm do you choose in hw3-1?
BLOCKED FLOYD-WARSHALL ALGORITHM，用openmp對for loop做平行運算

2. How do you divide your data in hw3-2, hw3-3?
![image](https://hackmd.io/_uploads/BkALJ5t8T.png)
按照BLOCKED FLOYD-WARSHALL ALGORITHM的流程切Block，一個Block大小為64x64，若n無法被64整除，就padding成64的倍數並存在N
    ```
    if (n % BLOCK_SIZE == 0) N = n;
    else N = n + BLOCK_SIZE - n % BLOCK_SIZE;
    ```

3. What’s your configuration in hw3-2, hw3-3? And why? (e.g. blocking factor, blocks, threads)
    * 把一個Block能有的thread數目跟shared memory開到最大最有效率
    * 一個Block開32 x 32個threads (thread limit = 1024)
    * 這32 x 32threads會處理大小為64 x 64的Region (因為shared memor limit = 64 x 64 x 3 bytes)
    * 一個thread會處理四格(如下圖)，分別處理(thread.idx, thread,idy)在4個32 x 32的Region對應的位置，至於為何這樣分配後面**Optimization**的部分會講
![image](https://hackmd.io/_uploads/BJ9q4qF8a.png)
    * 下圖為各個Phase的Block數
        ```
        const int round = N / BLOCK_SIZE;
        dim3 block_dim(PHY_BLOCK, PHY_BLOCK);
        dim3 grid_dim(round - 1, round - 1);

        for (int r = 0; r < round; ++r) {
            Phase1 <<< 1, block_dim >>> (dst, r, N);
            Phase2 <<< round - 1, block_dim >>> (dst, r, N);
            Phase3 <<< grid_dim, block_dim >>> (dst, r, N);
        }
        ```

4. How do you implement the communication in hw3-3?
    * 一台GPU負責一半的Row
    * 進迴圈之前，先把一半的Host Dist copy進Device dst
    * 之後每一round，都只需要把current round的row copy到device就好，因為device在ith round中，只需用到current round row的資料，下圖舉device1為例，在ith round只需知道到黃色部分的資料就可完成phase1, 2, 3的計算。
    ![image](https://hackmd.io/_uploads/rkv30oKUT.png)
    * 最後分別把自己那半copy回Host Dist即可

## Profiling Results (hw3-2)
### Profiling Results Metrics
* occupancy (Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor)
* sm efficiency (The percentage of time at least one warp is active on a multiprocessor averaged over all multiprocessors on the GPU)
* shared memory load/store throughput
* global load/store throughput

Testcase P21K1

### Phase1

| Metric               | Max        | Min        | Avg        |
| -------------------- | ---------- | ---------- | ---------- |
| sm efficiency        | 4.34%      | 4.24%      | 4.33%      |
| occupancy            | 0.496486   | 0.495919   | 0.496234   |
| shared memory  load  | 126.31GB/s | 112.04GB/s | 115.02GB/s |
| shared memory  store | 56.559GB/s | 63.902GB/s | 62.912GB/s |
| global load          | 405.54MB/s | 485.84MB/s | 467.59MB/s |
| global store         | 887.73MB/s | 1.0233GB/s | 953.99MB/s |


### Phase2

| Metric               | Min        | Max        | Avg        |
| -------------------- | ---------- | ---------- | ---------- |
| sm efficiency        | 95.29%     | 96.91%     | 95.85%     |
| occupancy            | 0.970919   | 0.981051   | 0.976396   |
| shared memory  load  | 2180.0GB/s | 2448.7GB/s | 2216.4GB/s |
| shared memory  store | 1478.2GB/s | 1661.6GB/s | 1626.4GB/s |
| global load          | 13.181GB/s | 14.049GB/s | 13.674GB/s |
| global store         | 22.498GB/s | 26.529GB/s | 23.889GB/s |

### Phase3
| Metric               | Min        | Max        | Avg        |
| -------------------- | ---------- | ---------- | ---------- |
| sm efficiency        | 99.95%     | 99.97%     | 99.97%     |
| occupancy            | 0.926313   | 0.929451   | 0.927155   |
| shared memory  load  | 3149.3GB/s | 3412.0GB/s | 3176.8GB/s |
| shared memory  store | 131.07GB/s | 142.11GB/s | 140.76GB/s |
| global load          | 22.503GB/s | 26.497GB/s | 23.434GB/s |
| global store         | 65.548GB/s | 72.892GB/s | 68.252GB/s |


## Experiment & Analysis
### System Spec
hw3-1 : apollo server
hw3-2, 3-3: hades02(GTX1080)

### Blocking Factor 
Testcase P11K1
Bolcking Factor (a, b) means  a x a threads handle b x b region
| Bolcking Factor     | (8, 16)    | (16, 32)   | (32, 64)   |
| ------------------- | ---------- | ---------- | ---------- |
| shared memory load  | 2225.4GB/s | 3926.2GB/s | 6052GB/s   |
| shared memory store | 1662.5GB/s | 2267.7GB/s | 1841.8GB/s |
| global load         | 224.8GB/s  | 264.8GB/s  | 256GB/s    |
| global store        | 191GB/s    | 102.3GB/s  | 98.5GB/s   |
| integer instruction | 4273248256 | 7431945280 | 1.3325e+10 |
| time                | 9.923s     | 4.826s     | 4.215s     |

![image](https://hackmd.io/_uploads/SkikuX3U6.png)
![image](https://hackmd.io/_uploads/HyNqum38T.png)

* 以時間來看，Bolcking Factor = (32, 64)的表現最好
* integer instruction也是(32, 64)最高，代表GPU被用於計算的地方越多
* Bolcking Factor越大，shared memory使用率越好
* Bolcking Factor越大，global memory的使用率越差，這是好的，因為global memory access時間長，應盡量少用

### Optimization 
我用32 x 32個threads去處理64 x 64的區域，每個threads負責的格子如下圖，這樣分配可以達到Larger blocking factor, Coalesced memory access以及Reduce Bank Conflict
![image](https://hackmd.io/_uploads/BJ9q4qF8a.png)
* Coalesced memory access
如下圖所示，這樣分配可以讓同個wrap讀的記憶體位置是連續的
![image](https://hackmd.io/_uploads/r18HeR5UT.png)
* Reduce Bank Conflict
從下方的程式可以看出，在round k，因為同一個wrap都在同一列，所以都會需要相同ROW[i][k]以及不同COL[k][j] (COL[k][j], COL[k][j+1], .... COL[k][j+31])。因此COL的部分不會有Bank Conflict，而ROW的部分可以透過Broadast
    ```
    for (int k = 0; k < BLOCK_SIZE; k++)
        L1 = min(L1, ROW[i][k] + COL[k][j]);
    ```
* Shared memory
    ```
    __shared__ int ROW[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int COL[BLOCK_SIZE][BLOCK_SIZE];

    const int i = threadIdx.y;
    const int j = threadIdx.x;
    const int i2 = i + PHY_BLOCK;
    const int j2 = j + PHY_BLOCK;

    ROW[i][j] = dist[block_i * N + offset + i * N + j];
    ROW[i][j2] = dist[block_i * N + offset + i * N + j2];
    ROW[i2][j] = dist[block_i * N + offset + i2 * N + j];
    ROW[i2][j2] = dist[block_i * N + offset + i2 * N + j2];

    COL[i][j] = dist[offset * N + block_j + i * N + j];
    COL[i][j2] = dist[offset * N + block_j + i * N + j2];
    COL[i2][j] = dist[offset * N + block_j + i2 * N + j];
    COL[i2][j2] = dist[offset * N + block_j + i2 * N + j2];
    ```

* Larger Chuck of File Read/Write
Read
    ```
    __align__(8) int* pair = (int*)malloc(3 * m * sizeof(int));
    fread(pair, sizeof(int), 3 * m, file);
    ```

    Write

    ```
    cudaMemcpy2D(&Dist[0], n * sizeof(int), dst, N * sizeof(int), n * sizeof(int), n, cudaMemcpyDeviceToHost);
    fwrite(&Dist[0], sizeof(int), n * n, outfile);
    ```
* unroll
#pragma unroll 32 will be slightly faster

*  Asynchronous Copy from Global Memory to Shared Memory
如下圖， __pipeline_memcpy_async() 不用經過L1及Register就可以直接寫到shared_memory
![image](https://hackmd.io/_uploads/ryBmv1nIT.png)
但我實測下來沒有很大的進步
 


### Weak scalability
Testcase p21k1(N = 20960 / n = 20959)
For half data(N = 14832 / n = 14820)
```
if (n % BLOCK_SIZE == 0) N = n;
else N = n + BLOCK_SIZE - n % BLOCK_SIZE;
printf("origin N = %d / n = %d\n", N, n);

n /= sqrt(2);
if (n % BLOCK_SIZE == 0) N = n;
else N = n + BLOCK_SIZE - n % BLOCK_SIZE;
printf("half N = %d / n = %d\n", N, n);
```

| Weak Scability | 1 GPU | 2GPU         |
| -------------- | ----- | ------------ |
| Time           | 9.02s | 11.28s       |
| Relative Time  | 1     | 1.25         |
| Speedup        | 1     | 1.6 (2/1.25) |

![image](https://hackmd.io/_uploads/HkBtmyo8T.png)

### Time Distribution
Bottleneck is Computing time
| Testcase   | p11k1 | p15k1 | p21k1 | p25k1 | p30k1 |
| ---------- | ----- | ----- | ----- | ----- | ----- |
| Compute    | 1.14  | 2.89  | 7.89  | 13.69 | 23.39 |
| I/O        | 0.99  | 1.85  | 3.59  | 5.09  | 7.35  |
| Cudamemcpy | 0.3   | 0.66  | 0.9   | 2.05  | 6.43  |

![image](https://hackmd.io/_uploads/SJWphzoIT.png)



## Experience & conclusion
CUDA GOOD
