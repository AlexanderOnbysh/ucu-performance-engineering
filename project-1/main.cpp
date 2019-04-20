#define LEN 33000
#define NTIMES 10000
#define MATRIX_DIM 100

#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <time.h>
#include <xmmintrin.h>
#include <smmintrin.h>


const double sec_const = 1000000.0;

// Task 1
float a[LEN] __attribute__((aligned(16)));
float b[LEN] __attribute__((aligned(16)));
float c[LEN] __attribute__((aligned(16)));
float d[LEN] __attribute__((aligned(16)));
float result[LEN] __attribute__((aligned(16)));

// Task 2
float matrix_a[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));
float matrix_b[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));
float matrix_c[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));


void timeit(void (*f)(), const unsigned n_times, const char *name) {
    clock_t start_t;
    clock_t end_t;

    start_t = clock();
    for (int n = 0; n < n_times; n++) {
        (*f)();
    }
    end_t = clock();

    printf("%-25s %-10.2f\n", name, (end_t - start_t) / sec_const);
}

// Task 1
// ------------------------------------------------------------
int nothing(float var[LEN]) {
    return (0);
}

void inline task1_not_vectorized() {
    for (int i = 0; i < LEN; i++) {
        result[i] = a[i] * b[i] + c[i] * d[i];
    }
    nothing(result);
}


void inline task1_vectorized() {
    __m128 rA, rB, rC, rD, rR;
    for (int i = 0; i < LEN; i += 4) {
        rA = _mm_load_ps(&a[i]);
        rB = _mm_load_ps(&b[i]);
        rC = _mm_load_ps(&c[i]);
        rD = _mm_load_ps(&d[i]);

        rR = _mm_add_ps(_mm_mul_ps(rA, rB), _mm_mul_ps(rC, rD));
        _mm_store_ps(&result[i], rR);
    }
    nothing(result);
}

// ------------------------------------------------------------
// Task 2
int nothing_matrix(float matrix[MATRIX_DIM][MATRIX_DIM]) {
    return 0;
}

void task2_matrix_multiplication_not_vectorized() {
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            matrix_c[i][j] = 0;
            for (int k = 0; k < MATRIX_DIM; k++) {
                matrix_c[i][j] = matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
    nothing_matrix(matrix_c);
}

void task2_matrix_multiplication_vectorized() {
    __m128 rA, rB, rR, rSum;

//     transpose matrix B for better cache access
    float matrix_b_transformed[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; i < MATRIX_DIM; i++) {
            matrix_b_transformed[i][j] = matrix_b[j][i];
        }
    }
//    multiply matrices using SSE
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; i < MATRIX_DIM; i++) {
            rSum = _mm_setzero_ps();
            for (int k = 0; k < MATRIX_DIM; k += 4) {
                rA = _mm_load_ps(&matrix_a[i][k]);
                rB = _mm_load_ps(&matrix_b_transformed[j][k]);
//                dot product
                rR = _mm_dp_ps(rA, rB, 0xff);
//                accumulate dot product
                rSum = _mm_add_ps(rSum, rR);
            }
            _mm_store_ps(&matrix_c[i][j], rSum);
        }
    }
    nothing_matrix(matrix_c);
}

// ------------------------------------------------------------
// Task 3
// ------------------------------------------------------------


int main() {
    printf("%-25s%-20s\n", "Function", "Time (Sec)\n");

//    timeit(&task1_not_vectorized, NTIMES, "Task 1: not vectorized");
//    timeit(&task1_vectorized, NTIMES, "Task 1: vectorized");
    timeit(&task2_matrix_multiplication_not_vectorized, NTIMES / 10, "Task 2: not vectorized");
    timeit(&task2_matrix_multiplication_vectorized, NTIMES / 10, "Task 2: vectorized");

    return 0;
}
