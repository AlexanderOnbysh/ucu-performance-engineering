#define LEN 33000
#define NTIMES 1000

#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <time.h>
#include <xmmintrin.h>

const double sec_const = 1000000.0;

float a[LEN] __attribute__((aligned(16)));
float b[LEN] __attribute__((aligned(16)));
float c[LEN] __attribute__((aligned(16)));
float d[LEN] __attribute__((aligned(16)));
float result[LEN] __attribute__((aligned(16)));


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

void task1_not_vectorized() {
    for (int i = 0; i < LEN; i++) {
        result[i] = a[i] * b[i] + c[i] * d[i];
    }
}


void task1_vectorized() {
    __m128 rA, rB, rC, rD, rR;
    for (int i = 0; i < LEN; i = i + 4) {
        rA = _mm_load_ps(&a[i]);
        rB = _mm_load_ps(&b[i]);
        rC = _mm_load_ps(&c[i]);
        rD = _mm_load_ps(&d[i]);

        rR = _mm_add_ps(_mm_mul_ps(rA, rB), _mm_mul_ps(rC, rD));
        _mm_store_ps(&result[i], rR);
    }
}


int main() {
    printf("%-25s%-20s\n", "Function", "Time (Sec)\n");

    timeit(&task1_not_vectorized, NTIMES, "Task 1: not vectorized");
    timeit(&task1_vectorized, NTIMES, "Task 2: vectorized");

    return 0;
}
