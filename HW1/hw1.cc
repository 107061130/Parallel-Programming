#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

#define EVEN_PHASE1 1
#define EVEN_PHASE2 2
#define ODD_PHASE1 3
#define ODD_PHASE2 4
#define min(a, b) (a < b ? a : b)
#define swap(type, a, b) {type c = a; a = b; b = c;}

inline void Merge_min(float* arr1, float* arr2, const int len1, const int len2, float* temp);
inline void Merge_max(float* arr1, float* arr2, const int len1, const int len2, float* temp);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_File input_file, output_file;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    char* input_filename = argv[2];
    char* output_filename = argv[3];

    size = min(size, 12);
    size = min(size, n);
    // local buffer, buffer size and start index
    int base = n / size;
    int data_size = base + (rank < (n % size) ? 1 : 0);
    int index = base * rank + (rank >= (n % size) ? (n % size) : rank);
    float* data = (float*)malloc(data_size * sizeof(float));
    //printf("rank %d got %d datas at location %d\n", rank, data_size, index);

    // receive buffer and neighbor's buffer size
    float* recv_data = recv_data = (float*)malloc((base + 1) * sizeof(float));;
    int right_data_size = base + (rank + 1 < (n % size) ? 1 : 0);
    int left_data_size = base + (rank - 1 < (n % size) ? 1 : 0);

    // merge buffer
    float* temp = (float*)malloc(data_size * sizeof(float));

    // read file and store values in local buffer
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (rank < size) MPI_File_read_at(input_file, sizeof(float) * index, data, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // sort locally
    if (data_size > 1) boost::sort::spreadsort::spreadsort(data, data + data_size);

    // Odd-Even Sort
    int sorted = 0, count = 0;
    float* cur_data = data, * cur_temp = temp;
    if (size > 1) {
        while (true) {
            if (rank < size) {
                sorted = 1;
                // Even Phase
                if (rank % 2 == 1) {
                    MPI_Sendrecv(cur_data, 1, MPI_FLOAT, rank - 1, EVEN_PHASE1, recv_data, 1, MPI_FLOAT, rank - 1, EVEN_PHASE1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (cur_data[0] < recv_data[0]) {
                        MPI_Sendrecv(cur_data, data_size, MPI_FLOAT, rank - 1, EVEN_PHASE2, recv_data, left_data_size, MPI_FLOAT, rank - 1, EVEN_PHASE2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_max(cur_data, recv_data, data_size, left_data_size, cur_temp);
                        swap(float*, cur_data, cur_temp);
                    }
                }
                if (rank % 2 == 0 && rank < size - 1) {
                    MPI_Sendrecv(cur_data + (data_size - 1), 1, MPI_FLOAT, rank + 1, EVEN_PHASE1, recv_data, 1, MPI_FLOAT, rank + 1, EVEN_PHASE1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (cur_data[data_size - 1] > recv_data[0]) {
                        sorted = 0;
                        MPI_Sendrecv(cur_data, data_size, MPI_FLOAT, rank + 1, EVEN_PHASE2, recv_data, right_data_size, MPI_FLOAT, rank + 1, EVEN_PHASE2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_min(cur_data, recv_data, data_size, right_data_size, cur_temp);
                        swap(float*, cur_data, cur_temp);
                    }
                }

                // Odd Phase
                if (rank % 2 == 0 && rank != 0) {
                    MPI_Sendrecv(cur_data, 1, MPI_FLOAT, rank - 1, ODD_PHASE1, recv_data, 1, MPI_FLOAT, rank - 1, ODD_PHASE1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (cur_data[0] < recv_data[0]) {
                        MPI_Sendrecv(cur_data, data_size, MPI_FLOAT, rank - 1, ODD_PHASE2, recv_data, left_data_size, MPI_FLOAT, rank - 1, ODD_PHASE2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_max(cur_data, recv_data, data_size, left_data_size, cur_temp);
                        swap(float*, cur_data, cur_temp);
                    }
                }
                if (rank % 2 == 1 && rank < size - 1) {
                    MPI_Sendrecv(cur_data + (data_size - 1), 1, MPI_FLOAT, rank + 1, ODD_PHASE1, recv_data, 1, MPI_FLOAT, rank + 1, ODD_PHASE1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (cur_data[data_size - 1] > recv_data[0]) {
                        sorted = 0;
                        MPI_Sendrecv(cur_data, data_size, MPI_FLOAT, rank + 1, ODD_PHASE2, recv_data, right_data_size, MPI_FLOAT, rank + 1, ODD_PHASE2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        Merge_min(cur_data, recv_data, data_size, right_data_size, cur_temp);
                        swap(float*, cur_data, cur_temp);
                    }
                }
            }
            MPI_Allreduce(&sorted, &count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (count == size) break;
        }
    }
    // write sorted outcome to out file
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (rank < size) MPI_File_write_at(output_file, sizeof(float) * index, cur_data, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    free(data);
    free(recv_data);
    free(temp);
    MPI_Finalize();
    return 0;
}

inline void Merge_min(float* arr1, float* arr2, const int len1, const int len2, float* temp) {
    int i = 0, j = 0, k = 0;
    while(k < len1 - 1) {
        if (arr1[i] <= arr2[j]) temp[k++] = arr1[i++];
        else temp[k++] = arr2[j++];
    }
    temp[k] = (j == len2 ? arr1[i] : min(arr1[i], arr2[j]));
}

inline void Merge_max(float* arr1, float* arr2, const int len1, const int len2, float* temp) {
    int i = len1 - 1, j = len2 - 1, k = len1 - 1;
    while(k >= 0) {
        if (arr1[i] > arr2[j]) temp[k--] = arr1[i--];
        else temp[k--] = arr2[j--];
    }
}
