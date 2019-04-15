#define LEN 33000
#define LEN2 1024
#define NTIMES 100000

#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <time.h>
#include <xmmintrin.h>

const double sec_const = 1000000.0;

float a[LEN] __attribute__((aligned(16)));
float b[LEN] __attribute__((aligned(16)));
float c[LEN] __attribute__((aligned(16)));
float d[LEN2] __attribute__((aligned(16)));

float ma[LEN2][LEN2] __attribute__((aligned(16)));
float sum[LEN2] __attribute__((aligned(16)));

int nothing_00(float a[LEN], float b[LEN], float c[LEN]) {
    return (0);
}

int nothing_10(float a[LEN], float ma[LEN2][LEN2], float d[LEN2]) {
    return (0);
}

int e00() {
    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;

    start_t = clock();

    for (int n = 0; n < NTIMES; n++) {
        for (int i = 0; i < LEN; i++) {
            c[i] = a[i] * b[i];
        }
        nothing_00(a, b, c);
    }
    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = clock_delta / sec_const;
    printf("E00\t %.2f \t\n", clock_delta_sec);

    return 0;
}



int e01() {

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;

    start_t = clock();

    __m128 rA, rB, rC;
    for (int n = 0; n < NTIMES; n++) {
        for (int i = 0; i < LEN; i += 4) {
            rA = _mm_load_ps(&a[i]);
            rB = _mm_load_ps(&b[i]);
            rC = _mm_mul_ps(rA, rB);
            _mm_store_ps(&c[i], rC);
        }
        nothing_00(a, b, c);
    }
    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("E01\t %.2f \t\n", clock_delta_sec);
    return 0;
}

int e10() {

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    start_t = clock();

    for (int n = 0; n < NTIMES / 500; n++) {
        for (int i = 0; i < LEN2; i++) {
            float sum = 0;
            for (int j = 0; j < LEN2; j++) {
                sum += ma[j][i];
            }
            d[i] = sum;
        }
        nothing_10(a, ma, d);
    }

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("E10\t %.2f \t\n", clock_delta_sec);
    return 0;
}


int e11() {

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    start_t = clock();

    for (int nl = 0; nl < NTIMES / 500; nl++) {
        for (int i = 0; i < LEN2; i++) {
            sum[i] = 0;
            for (int j = 0; j < LEN2; j++) {
                sum[i] += ma[j][i];
            }
            d[i] = sum[i];
        }
        nothing_10(a, ma, d);
    }

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("E11\t %.2f \t\n", clock_delta_sec);
    return 0;
}


int main() {
    printf("Function \t Time(Sec) \n");
    e00();
    e01();
    e10();
    e11();
    return 0;
}
