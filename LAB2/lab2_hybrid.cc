#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int size, rank, rc;
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0, ans = 0;
	int thread_num, thread_id;

	#pragma omp parallel reduction(+:pixels)
	{
		/*
		thread_num = omp_get_num_threads();
		thread_id = omp_get_thread_num();
		for (unsigned long long x = rank*thread_num + thread_id; x < r; x += thread_num*size) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			pixels += y;
		}
		*/
		#pragma omp for
		for (unsigned long long x = rank; x < r; x += size) {
			unsigned long long y = ceil(sqrtl(r * r - x * x));
			pixels += y;
		}
		pixels %= k;
	}

	pixels %= k;
	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	ans %= k;
	if (rank == 0) printf("%llu\n", (4 * ans) % k);

	MPI_Finalize();
	return 0;
}
