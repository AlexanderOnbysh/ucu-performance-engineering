#define LEN 33000
#define NTIMES 10000
#define MATRIX_DIM 100

#define SEED 42
#define STRING_LEN 30000
#define SUBSTRING_LEN 1000

#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <sys/times.h>
#include <time.h>
#include <xmmintrin.h>
#include <cblas.h>
#include <smmintrin.h>


const double sec_const = 1000000.0;

// Task 1
float a[LEN] __attribute__((aligned(16)));
float b[LEN] __attribute__((aligned(16)));
float c[LEN] __attribute__((aligned(16)));
float d[LEN] __attribute__((aligned(16)));
float result[LEN] __attribute__((aligned(16)));

// Task 2
double matrix_a[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));
double matrix_b[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));
double matrix_c[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));


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
int nothing_matrix(double matrix[MATRIX_DIM][MATRIX_DIM]) {
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
    __m128d rA, rB, rR, rSum;

//     transpose matrix B for better cache access
    double matrix_b_transformed[MATRIX_DIM][MATRIX_DIM] __attribute__((aligned(16)));
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; i < MATRIX_DIM; i++) {
            matrix_b_transformed[i][j] = matrix_b[j][i];
        }
    }
//    multiply matrices using SSE
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; i < MATRIX_DIM; i++) {
            rSum = _mm_setzero_ps();
            for (int k = 0; k < MATRIX_DIM; k += 2) {
                rA = _mm_load_pd(&matrix_a[i][k]);
                rB = _mm_load_pd(&matrix_b_transformed[j][k]);
//                dot product
                rR = _mm_dp_pd(rA, rB, 0xff);
//                accumulate dot product
                rSum = _mm_add_pd(rSum, rR);
            }
            _mm_store_pd(&matrix_c[i][j], rSum);
        }
    }
    nothing_matrix(matrix_c);
}

void task2_matrix_multiplication_openblast() {
    double *matrix_a_flatten = &matrix_a[0][0];
    double *matrix_b_flatten = &matrix_b[0][0];
    double *matrix_c_flatten = &matrix_c[0][0];

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                (int) MATRIX_DIM, (int) MATRIX_DIM, (int) MATRIX_DIM, 1,
                matrix_a_flatten, (int) MATRIX_DIM,
                matrix_b_flatten, (int) MATRIX_DIM, (int) MATRIX_DIM,
                matrix_c_flatten, (int) MATRIX_DIM);
    nothing_matrix(matrix_c);
}

// ------------------------------------------------------------
// Task 3
// ------------------------------------------------------------
void gen_random_string(char *s, const int len) {
    static const char alphanum[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    s[len] = 0;
}

void gen_random_substring(char *s, const int len, char *subs, const int substring_len) {
    int n = rand() % (len - substring_len - 1);
    memset(subs, '\0', (size_t) substring_len);
    strncpy(subs, &s[n], (size_t) substring_len - 1);
}

void task3_strstr_search() {
    char str[STRING_LEN + 1];
    char substr[SUBSTRING_LEN + 1];
    gen_random_string(&str[0], (int) STRING_LEN);
    gen_random_substring(&str[0], STRING_LEN, &substr[0], SUBSTRING_LEN);
    strstr(str, substr);
}

void task3_brute_force_search() {
    char str[STRING_LEN];
    char substr[SUBSTRING_LEN];

    gen_random_string(&str[0], (int) STRING_LEN);
    gen_random_substring(&str[0], STRING_LEN, &substr[0], SUBSTRING_LEN);

    for (int i = 0; i < STRING_LEN - SUBSTRING_LEN - 1; i++) {
        for (int j = 0; j < SUBSTRING_LEN - 1; j++) {
            if (str[i + j] != substr[j]) {
                break;
            }
            if (j == SUBSTRING_LEN - 2) {
                return;
            }
        }
    }
}


void task3_karp_rabin_simd() {
// Karp-Rabin algorithm with weak hashing

    char str[STRING_LEN + 1];
    char substr[SUBSTRING_LEN + 1];

    gen_random_string(&str[0], (int) STRING_LEN);
    gen_random_substring(&str[0], STRING_LEN + 1, &substr[0], SUBSTRING_LEN + 1);

//    first and last characters in substring
    const __m128i first = _mm_set1_epi8(substr[0]);
    const __m128i last = _mm_set1_epi8(substr[SUBSTRING_LEN - 1]);

//    iterate over searching string in 16 bytes chunks
    for (size_t i = 0; i < STRING_LEN; i += 16) {

        const __m128i block_first = _mm_loadu_si128((__m128i *) (&str[0] + i));
        const __m128i block_last = _mm_loadu_si128((__m128i *) (&str[0] + i + SUBSTRING_LEN - 1));

        const __m128i eq_first = _mm_cmpeq_epi8(first, block_first);
        const __m128i eq_last = _mm_cmpeq_epi8(last, block_last);

//        mask by weak hashing
        uint16_t mask = static_cast<uint16_t>(_mm_movemask_epi8(_mm_and_si128(eq_first, eq_last)));

        while (mask != 0) {
//            index of most significant bit in mask
            int bitpos = __builtin_ctzl(mask);


            if (memcmp(str + i + bitpos + 1, substr + 1, SUBSTRING_LEN - 2) == 0) {
                return;
            }
//            clear most significant bit in mask
            mask = (uint16_t) (mask & (mask - 1));
        }
    }
}


int main() {
    printf("%-25s%-20s\n", "Function", "Time (Sec)\n");
//    Task 1
    srand(SEED);
    timeit(&task1_not_vectorized, NTIMES, "Task 1: not vectorized");
    srand(SEED);
    timeit(&task1_vectorized, NTIMES, "Task 1: vectorized");
    printf("--------------------\n");
//    Task 2
    srand(SEED);
    timeit(&task2_matrix_multiplication_not_vectorized, NTIMES / 10, "Task 2: not vectorized");
    srand(SEED);
    timeit(&task2_matrix_multiplication_vectorized, NTIMES / 10, "Task 2: vectorized");
    srand(SEED);
    timeit(&task2_matrix_multiplication_openblast, NTIMES / 10, "Task 2: openBLAST");
    printf("--------------------\n");
//    Task 3
    srand(SEED);
    timeit(&task3_strstr_search, NTIMES, "Task 3: strstr");
    srand(SEED);
    timeit(&task3_brute_force_search, NTIMES, "Task 3: brute force search");
    srand(SEED);
    timeit(&task3_karp_rabin_simd, NTIMES, "Task 3: Karp-Rabin simd");

    return 0;
}
