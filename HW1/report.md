# HW1 Odd-Even Sort
張瀚 / 111064528

## Implementation
### How do you handle an arbitrary number of input items and processes?
平均分配，若有餘數則由rank小的依次分派
```
// local buffer, buffer size and start index
int base = n / size;
int data_size = base + (rank < (n% size) ? 1 : 0);
int index = base * rank + (rank >= (n % size) ? (n % size) : rank);
float* data = (float*)malloc(data_size * sizeof(float));
```
### How do you sort in your program?
1. 每個process先locally sort各自的element
2. 在odd-even phase時把較**小的**elements丟給**process i**，把較**大的**丟給 **process i+1**，若已sort好，則sorted = 1，反之則為0
3. 用MPI_Allreduce()來決定是否要終止，如果sorted數達到min(n, size)，則程式結束

### Other efforts you’ve made in your program
1. 在odd-even phase時，**資料由單向傳輸改為雙向**，節省了3k的時間(k = N/m)

Before

![image](https://github.com/107061130/Parallel-Programming/assets/79574369/512b99a5-626c-4ad6-a6cd-319e69a19ce1)

After

![image](https://github.com/107061130/Parallel-Programming/assets/79574369/d018092c-1b0c-4389-9387-3d90a9a1a4ca)

2. Quick sort to **boost::spreadsort**(good performance on large size floating point array)
3. 在送資料之前，**先互送一筆data確認是否sort過了**，如果**process i的尾**已經小於等於**process i+1的頭**，那就沒必要傳送資料了。這個做法會導致小測資時間增加，但在大測資上會有微小進步
```
MPI_Sendrecv(cur_data, 1, MPI_FLOAT, rank - 1, EVEN_PHASE1, recv_data, 
1, MPI_FLOAT, rank - 1, EVEN_PHASE1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```
4. 在Merge function中，減少while()迴圈中不必要的判斷

Before
```
while (i >= 0 && j >= 0 && k >= 0) {
    if (arr1[i] > arr2[j]) temp[k--] = arr1[i--];
    else temp[k--] = arr2[j--];
}
```
After
```
 while(k >= 0)
```
5. 在Merge function的最後一步，本來要花N/m的時間去把temp中的資料copy進data，如下
```
for (int i = 0; i < len1; i++) arr1[i] = temp[i];
```
後改成用**指標操作的方式去替換data跟temp**，把這步變constant time，做法是在每次call Merge()都去swap指到data跟temp的指標

進while()之前
```
float* cur_data = data, * cur_temp = temp;
```
每次Merge之後，進行swap
```
if (cur_data[0] < recv_data[0]) {
    MPI_Sendrecv(cur_data, data_size, MPI_FLOAT, rank - 1, EVEN_PHASE2, recv_data, left_data_size, MPI_FLOAT, rank - 1, EVEN_PHASE2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Merge_max(cur_data, recv_data, data_size, left_data_size, cur_temp);
    swap(float*, cur_data, cur_temp);
}
```

## Experiment & Analysis
### i. Methodology
#### (a) System Spec
apollo.cs.nthu.edu.tw server

#### (b) Performance Metrics
由於我用IPM static或dynamic mode去做srun時，跑出來的結果都不對，所以改用MPI_Wtime()去包指令，針對multi-process的case，我會用MPI_Reduce()去加總，然後在rank 0算出平均值並print
```
START = MPI_Wtime();
// read file and store values in local buffer
MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
END = MPI_Wtime();
FILE_OPEN_CLOSE += END - START;
```
```
MPI_Reduce(&CPU, &ans[0], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&COMM, &ans[1], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&FILE_READ_WRITE, &ans[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&FILE_OPEN_CLOSE, &ans[3], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
    for (int i = 0; i < 4; i++) ans[i] /= size;
    printf("CPU = %.3lf / COMM = %.3lf / FILE_READ_WRITE = %.3lf / FILE_OPEN_CLOSE = %.3lf\n", ans[0], ans[1], ans[2], ans[3]);
}
```

### ii. Plots: Speedup Factor & Profile
#### Experimental Method
* Test Case Description
/share/data/.testcases/hw1/38.in, contains 536831999 number
* Parallel Configurations
1 Node, Process = {1,2,4,6,8,10,12}
![](https://hackmd.io/_uploads/By85fIMMT.png)
3 Nodes, Process = {1,2,4,6,8,10,12}
![](https://hackmd.io/_uploads/BkVSIUMfp.png)

#### Performance Measurement
1. CPU: cpu times, basicly is Total time - Communication time - File I/O time
2. COMM: Communication time, includes MPI_Sendrecv() and MPI_Allreduce()
3. FILE_READ_WRITE: MPI_File_read/write_at() for input/output file
4. FILE_OPEN_CLOSE: MPI_File_open/close() for input & output file

#### Analysis of Results
![](https://hackmd.io/_uploads/H1sMnIfMa.png)
![](https://hackmd.io/_uploads/Hy4X3Izfp.png)
1. 可以看到當process數上升，I/O變成Bottleneck，但是I/O的時間其實有上限，並不會隨著Process數增加而上升
2. 我本來預期FILE_READ_WRITE的時間會隨著process數增加而減少，因為每個process要寫的資料變少，而且我採用的是Independent I/O，但結果顯示就算是Independent I/O，還是會互相衝突
3. 可以看到不管是Single Node或3 Nodes，10個process表現都是最好的，雖然增加process可以減少CPU time，但卻會增加Commnunication time
4. 我原本預期在multi node的模式下會比較久，因為不同node做Communication跟I/O感覺都會花比較多的時間，但結果顯示除了process num = 2，其他case都差不多

---

![](https://hackmd.io/_uploads/B1-UpLMG6.png)
![](https://hackmd.io/_uploads/HkFFa8Mfa.png)
1. 可以看到不管single或3 node，strong scability都非常差，最多只到2
2. 原因是因為I/O才是Bottleneck，以及Communication time會隨著process num增加而稱加，如果單看CPU time，strong scability是理想的

#### Experiences / Conclusion
1. 用local sort + Odd-Even sort的複雜度是O(N/m * lg(N/m) + m * (N/m))，前面是local sort的後面是communication，因為每次有(N/m)筆data要傳，而至多傳m次。
2. 當m = N時，複雜度為O(n)，平行度比上課教的Bitonic
mergesort還差
3. 這次作業除了mpi之外，還學到兩個打程式的規則，第一個是memory allocation不要放在loop裡，我一開始把temp放在Merge()裡宣告，結果就TLE了。第二個是當一個指標malloc完，就不能把它當一般指標操作了。
