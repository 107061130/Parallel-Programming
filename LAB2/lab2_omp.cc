#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	int thread_num, thread_id;

	#pragma omp parallel reduction(+:pixels)
	{
		//thread_num = omp_get_num_threads();
		//thread_id = omp_get_thread_num();
		#pragma omp for
		for (unsigned long long x = 0; x < r; x++) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			pixels += y;
		}
		pixels %= k;
	}
	pixels %= k;
	printf("%llu\n", (4 * pixels) % k);
	return 0;
}
