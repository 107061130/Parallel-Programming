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
#include <pthread.h>
#include <smmintrin.h>
#include <math.h>
#define threshold1 1e-3
#define threshold2 1e-4
#define min(a, b) (a < b ? a : b)

void* task(void* threadid);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);

double left, right, lower, upper, dx, dy;
int iters, width, height, location = 0, *image, ncpus;

int main(int argc, char** argv) {
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    dx = (right - left) / width;
    dy = (upper - lower) / height;

    /* detect how many CPUs are available */
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ncpus = CPU_COUNT(&cpuset);
    //printf("%d cpus available\n", CPU_COUNT(&cpuset));

    /* allocate memory for image */
    image = (int*)malloc((width * height + +ncpus) * sizeof(int));
    assert(image);

    /* create threads */
    int ID[ncpus];
    pthread_t threads[ncpus];
    for (int i = 0; i < ncpus; i++) ID[i] = i;
    for (int i = 0; i < ncpus; i++) pthread_create(&threads[i], NULL, task, (void*)&ID[i]);
    for (int i = 0; i < ncpus; i++) pthread_join(threads[i], NULL);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    return 0;
}

void* task(void* threadid) {
    int* tid = (int*)threadid;
    int BOUND = width * height + ncpus;
    __m128d two = _mm_set_pd1(2), four = _mm_set_sd(4);
    __m128d zero_one = _mm_castsi128_pd(_mm_set_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    __m128d one_zero = _mm_castsi128_pd(_mm_set_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF));

    /* set first two points */
    int index1 = *tid, index2 = *tid + ncpus, index = index2;
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
            index1 = index = index + ncpus;
            if (index >= BOUND) break;

            __m128d tempx = _mm_set_sd((index1 % width) * dx + left), tempy = _mm_set_sd((index1 / width) * dy + lower);
            x0 = _mm_move_sd(x0, tempx);
            y0 = _mm_move_sd(y0, tempy);
            x = _mm_and_pd(x, zero_one);
            y = _mm_and_pd(y, zero_one);
            length_squared = _mm_and_pd(length_squared, zero_one);
        }
        /* update m128d[64 : 127] if it's done */
        if (repeats[1] >= iters || _mm_comige_sd(_mm_shuffle_pd(length_squared, length_squared, 1), four)) {
            image[index2] = repeats[1];
            repeats[1] = 0;
            index2 = index = index + ncpus;
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
    pthread_exit(NULL);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
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