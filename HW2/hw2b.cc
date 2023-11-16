#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <smmintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer);

int main(int argc, char** argv) {
    /* Init MPI */
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    const int iters = strtol(argv[2], 0, 10);
    const double left = strtod(argv[3], 0);
    const double right = strtod(argv[4], 0);
    const double lower = strtod(argv[5], 0);
    const double upper = strtod(argv[6], 0);
    const int width = strtol(argv[7], 0, 10);
    const int height = strtol(argv[8], 0, 10);
    const double dx = (right - left) / width;
    const double dy = (upper - lower) / height;

    /* allocate memory for image */
    int* image = (int*)calloc(width * height + size * 16, sizeof(int));
    assert(image);

    #pragma omp parallel
    {
        int ncpus = omp_get_num_threads();
        int id = omp_get_thread_num();
        int BOUND = width * height + ncpus * size;

        __m128d two = _mm_set_pd1(2), four = _mm_set_pd1(4);
        __m128d zero_one = _mm_castsi128_pd(_mm_set_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
        __m128d one_zero = _mm_castsi128_pd(_mm_set_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF));

        /* set first two points */
        int index1 = rank * ncpus + id, index2 = index1 + size * ncpus, index = index2;
        __m128d x0 = _mm_set_pd((index2 % width) * dx + left, (index1 % width) * dx + left);
        __m128d y0 = _mm_set_pd((index2 / width) * dy + lower, (index1 / width) * dy + lower);
        __m128d x = x0;
        __m128d y = y0;
        __m128d length_squared = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
        int repeats[2] = { 1, 1 };

        /* mandelbrot set computing */
        while (true) {
            /* update m128d[0 : 63] if it's done */
            if (repeats[0] >= iters || _mm_comige_sd(length_squared, four)) {
                image[index1] = repeats[0];
                repeats[0] = 0;
                index1 = index = index + size * ncpus;
                if (index >= BOUND) break;

                __m128d tempx = _mm_set_sd((index1 % width) * dx + left), tempy = _mm_set_sd((index1 / width) * dy + lower);
                x0 = _mm_move_sd(x0, tempx);
                y0 = _mm_move_sd(y0, tempy);
                x = _mm_and_pd(x, zero_one);
                y = _mm_and_pd(y, zero_one);
                length_squared = _mm_and_pd(length_squared, zero_one);
            }
            /* update m128d[64:127] if it's done */
            if (repeats[1] >= iters || _mm_comige_sd(_mm_shuffle_pd(length_squared, length_squared, 1), four)) {
                image[index2] = repeats[1];
                repeats[1] = 0;
                index2 = index = index + size * ncpus;
                if (index >= BOUND) break;

                __m128d tempx = _mm_set_pd((index2 % width) * dx + left, 0), tempy = _mm_set_pd((index2 / width) * dy + lower, 0);
                x0 = _mm_move_sd(tempx, x0);
                y0 = _mm_move_sd(tempy, y0);
                x = _mm_and_pd(x, one_zero);
                y = _mm_and_pd(y, one_zero);
                length_squared = _mm_and_pd(length_squared, one_zero);
            }
            repeats[0]++; repeats[1]++;
            __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), x0);
            y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, y), two), y0);
            x = temp;
            length_squared = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
        }
    }

    /* draw and cleanup */
    int* result = NULL;
    if (rank == 0) result = (int*)malloc(width * height * sizeof(int));
    MPI_Reduce(image, result, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) write_png(filename, iters, width, height, result);

    MPI_Finalize();
    free(result);
    free(image);
    return 0;
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}