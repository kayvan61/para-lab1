#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "gen_matrix.h"
#include "my_malloc.h"

// sequential part of mm
void seq_mm(double *result, double *a, double *b, int dim_size) {
  int x, y, k;
  for (y = 0; y < dim_size; ++y) {
    for (x = 0; x < dim_size; ++x) {
      double r = 0.0;
      for (k = 0; k < dim_size; ++k) {
	r += a[y * dim_size + k] *  b[k * dim_size + x];
      }
      result[y * dim_size + x] = r;
    }
  }
}

void print_matrix(double *result, int x_b, int y_b, int row) {
  int x, y;
  for (y = 0; y < y_b; ++y) {
    for (x = 0; x < x_b; ++x) {
      printf("%f ", result[y * row + x]);
    }
    printf("\n");
  }
}

double dotProd(double* row, double* col, int dim, int matrix_size) {
  double ret = 0.0;
  for (int i = 0; i < matrix_size; i++) {
    ret += row[i] * col[dim*i];
  }
  return ret;
}

int main(int argc, char *argv[]) {
  double **r;
  double **result;
  int i;
  int num_arg_matrices;

  if (argc != 4) {
    printf("usage: debug_perf test_set matrix_dimension_size\n");
    exit(1);
  }
  int debug_perf = atoi(argv[1]);
  int test_set = atoi(argv[2]);
  matrix_dimension_size = atoi(argv[3]);
  num_arg_matrices = init_gen_sub_matrix(test_set);

  int rank, size;
  num_arg_matrices = init_gen_sub_matrix(test_set);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // determine the tile size type beat
  // current scheme is to do dumb tiling on whole rows and columns
  
  int blocksize = matrix_dimension_size / size;
  
  
  for(int i = 1; i < num_arg_matrices; i++) {
          
  }

  if(debug_perf == 0) {
    if(rank == 0) {
      printf("result matrix\n", 0);
      print_matrix(row_chunk, matrix_dimension_size, blocksize, matrix_dimension_size);
      double* p_buf = my_malloc(sizeof(double) * matrix_dimension_size * blocksize);
      for (int i = 1; i < size; i++) {
	MPI_Recv(p_buf, blocksize*matrix_dimension_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	print_matrix(p_buf, matrix_dimension_size, blocksize, matrix_dimension_size);
      }
      printf("\n");
      my_free(p_buf);
    }
    else{
      MPI_Send(row_chunk, blocksize*matrix_dimension_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
  }
  
  // MPI_Recv(void* data, int count, MPI_Datatype datatype, int from, int tag, MPI_Comm comm, MPI_Status* status);
  // MPI_Send(void* data, int count, MPI_Datatype datatype, int to,   int tag, MPI_Comm comm);
  
  MPI_Finalize();
  return 0;
}

