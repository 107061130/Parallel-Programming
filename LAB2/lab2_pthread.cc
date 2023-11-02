#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long r, k, ncpus, pixels = 0;
pthread_mutex_t mutex;

void* count_pixels(void* threadid) {
	int* tid = (int*)threadid;
	unsigned long long temp = 0;
	for (unsigned long long x = *tid; x < r; x += ncpus) {
		unsigned long long y = ceil(sqrtl(r * r - x * x));
		temp += y;
	}
	pthread_mutex_lock(&mutex);
	pixels += temp;
	pixels %= k;
	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);

	pthread_t threads[ncpus];
	int ID[ncpus];
	int rc;
	pthread_mutex_init(&mutex, NULL);

	for (int t = 0; t < ncpus; t++) {
		ID[t] = t;
		rc = pthread_create(&threads[t], NULL, count_pixels, (void*)&ID[t]);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	for (int i = 0; i < ncpus; i++) pthread_join(threads[i], NULL);
	pthread_mutex_destroy(&mutex);
	printf("%llu\n", (4 * pixels) % k);
	return 0;
}