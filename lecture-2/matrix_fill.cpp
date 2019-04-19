#include <stdlib.h>
#include <stdio.h>
#include <sys/times.h>
#include <time.h>
#include <xmmintrin.h>

const float sec_const = 1000000.0;

int nothing(int sum, int a){
	return(0);
}

double matrix_sum_rows_malloc(size_t width)
{
  clock_t start_t;
  clock_t end_t;
  clock_t clock_delta;
  double clock_delta_sec;
  //size_t width = 1024*4*2;
  
  int **mat = (int **)malloc(width * sizeof(int *)); 
  for (size_t i=0; i<width; i++) 
    mat[i] = (int *)malloc(width * sizeof(int)); 
  
  //Init SEED. Why we need it?
  srand(1);  
  start_t = clock();

  //fill the matrix, avoid pre-optimization
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < width; j++) {
      mat[i][j] = rand();
    }    
  }  
  //
//Timer stop
  end_t = clock();
  clock_delta = end_t - start_t;
  clock_delta_sec = (double) (clock_delta / sec_const);
  printf("%.6f ;", clock_delta_sec);
  
    
  long sum = 0;
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < width; j++) {
      sum = mat[i][j] + sum;
    }    
  }
  
  
  for (size_t i=0; i<width; i++){
    free(mat[i]);
  }
  free(mat);
  
  return (sum);
}

double matrix_sum_cols_malloc(size_t width)
{
  clock_t start_t;
  clock_t end_t;
  clock_t clock_delta;
  double clock_delta_sec;
  //size_t width = 1024*4*2;
  
  int **mat = (int **)malloc(width * sizeof(int *)); 
    for (size_t i=0; i<width; i++) 
         mat[i] = (int *)malloc(width * sizeof(int)); 
  
  //Init SEED. Why we need it?
  srand(1);  
  start_t = clock();
  //Timer started
  
  //fill the matrix, avoid pre-optimization
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < width; j++) {
      mat[j][i] = rand();
    }    
  }  
  //
  //Timer stop
  end_t = clock();
  clock_delta = end_t - start_t;
  clock_delta_sec = (double) (clock_delta / sec_const);
  printf("%.6f\n", clock_delta_sec);
    
  long sum = 0;
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < width; j++) {
      sum = mat[j][i] + sum;
    }    
  }
  
    
  for (size_t i=0; i<width; i++){
    free(mat[i]);
  }
  free(mat);
  
  return (sum);
}




int main(){
  printf("SIZE;ROW;COL;\n");
  printf("32;");
  matrix_sum_rows_malloc(32);
  matrix_sum_cols_malloc(32);
  printf("64;");
  matrix_sum_rows_malloc(64);
  matrix_sum_cols_malloc(64);
  printf("128;");
  matrix_sum_rows_malloc(128);
  matrix_sum_cols_malloc(128);
  printf("1024;");
  matrix_sum_rows_malloc(1024);
  matrix_sum_cols_malloc(1024);
  printf("2048;");
  matrix_sum_rows_malloc(1024*2);
  matrix_sum_cols_malloc(1024*2);
  printf("4096;");
  matrix_sum_rows_malloc(1024*4);
  matrix_sum_cols_malloc(1024*4);
  printf("16384;");
  matrix_sum_rows_malloc(1024*16);
  matrix_sum_cols_malloc(1024*16);
  return 0;
}

