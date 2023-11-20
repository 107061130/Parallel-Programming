# HW2 Mandelbrot Set
張瀚 / 111064528
---
## Implementation
### How do you partition the task?
I apply static allocation in both pthread and hybrid versions. I use hashing strategy to assign pixels to a thread. For example, *thread i* will handle indexes that mod *thread num* = *i*. Below figure shows how it works.   

(*process id*, *thread id*)
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/3807d634-f905-45c5-b193-65d871d1a0d5)


I found that this partition can achieve good load balancing becasue height * width is a lot greater than process number. Hence, i don't use dynamic allocation which will cause another time in communication or mutual exclusion section.

pthread
```
for (int i = *tid; i < height*width; i += ncpus)
```
hybrid
```
for (int i = *tid + rank*ncpus; i < height*width; i += size*ncpus)
```

### What technique do you use to reduce execution time and increase scalability?
1. use _m128d register to do two pixels at the same time
2. At first I make two pixels together, like the code below. However, this may cause a one pixel idle if it finishs first
```
for (int i = *tid; i < height; j += 2*ncpus) {
    pixels1 = i;
    pixels2 = i+1;
    /*Mandelbrot Set Caculation*/
    ...
    ...
}
```
3. Hence, in the final version, if a pixel is done, it will imediately change to another pixel. I use a variable called index to trace next pixel location.

```
pixel1 = *tid;
pixel2 = *tid + ncpus;
index = pixel2;
while (true) {
    if (pixel1_is_done) {
        pixel1 = index = index + ncpus;
        if (index >= BOUND) break;
    }
    if (pixel2_is_done) {
        pixel2 = index = index + ncpus;
        if (index >= BOUND) break;
    }
     /*Mandelbrot Set Caculation*/
    ...
    ...
}
```

### Other efforts you made in your program
1. I use _mm_comilt_sd(length_squared, four) to check whether length_squared is out of range, which save a lot of time compare to _mm_store_pd()
2. I also tried union and _mm_movemask_pd(), but they are not faster than _mm_comilt_sd()


## Experiment & Analysis
### i、Methodology
#### System Spec
apollo.cs.nthu.edu.tw server
#### Performance Metrics
Use MPI_Wtime() to compute the CPU, IO and Communication time

### ii、 Plots: Scalability & Load Balancing & Profile
#### Experimental Method
* Test Case Description
/share/data/.testcases/hw1/strict35.txt
* Parallel Configurations
1 Node, Process = {1,2,4,6,8,10,12}
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/8009d279-d41a-4eb4-943c-e54d44592fc7)

3 Nodes, Process = {1,2,4,6,8,10,12}
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/5f10bf95-ae24-44b5-a334-bad88cc3fc91)


#### Performance Measurement
1. CPU: CPU times, the time spend in Mandelbrot Set computaion
2. COMM: only in hybrid version, the time  processes reduce the result to process 0
3. IO: Write png file time

#### Analysis of Results
##### Load Balancing
For Load Balancing, I only record the time in the thread funtion, without considering I/O time and initial setup.

pthread(thread num = 12)
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/2dbeaecd-c98f-4b21-8fa3-7556963e2494)


hybrid(process num = 3, thread num = 12)
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/b76f30b8-7538-4542-aacd-e0e20c344d23)



Static allocation by hashing can achieve great balance.

##### Time profile
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/fad13c91-fb83-4220-9fbb-9db85a887f22)
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/7bd84f04-f4ba-425f-b3ed-10a395ade0f2)


* In pthread version, execution time significantly decrease with thread number, it's because IO time is fixed and only occupies a small portion of time
* In hybrid version, the scability is still good, because IO and Communication time is relatively small

##### Speedup
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/faba5169-d8d1-429d-923f-b8c2d6dd10a6)
![image](https://github.com/107061130/Parallel-Programming/assets/79574369/4dfd84af-89dc-4235-accd-1584b1ac37ce)


* In pthread version, the speedup is almost same as ideal at first, but gradually degrade when thread num increase due to the fixed IO time.
* In hybrid version, the thread number is from 3 to 36 because we use 3 process at the same time. Same as pthread version, scability gradually degrade due to the fixed IO time and Communication time.

### Experience & Conclusion
Except pthread and omp programming, the biggest gain from the HW is vectorization. I am glad that I have learned how to use m128 register to furthur speed up my program.
