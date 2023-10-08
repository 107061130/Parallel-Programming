#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	int numproc, rank, rc;
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long ans = 0;
	for (unsigned long long i = rank; i < r; i += numproc) {
		unsigned long long y = ceil(sqrtl(r * r - i * i));
		pixels += y;
	}

	//MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	ans %= k;

	if (rank == 0) printf("%llu\n", (ans*4) % k);

	MPI_Finalize();
}
