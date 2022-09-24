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
  
  double* results = my_malloc(sizeof(double) * matrix_dimension_size * blocksize);
  double* row_chunk = my_malloc(sizeof(double) * matrix_dimension_size * blocksize);
  double* col_chunk = my_malloc(sizeof(double) * matrix_dimension_size * blocksize);

  int send_buf_size = sizeof(double) * matrix_dimension_size * blocksize + MPI_BSEND_OVERHEAD;
  void* send_buf = my_malloc(send_buf_size);
  if(!send_buf) {
    printf("send_buffer not alloced.\n");
    exit(-1);
  }
  
  int buffer_res = MPI_Buffer_attach(send_buf, send_buf_size);
  if(buffer_res) {
    printf("rank %d failed to attach a send buffer\n", rank);
    exit(-1);
  }

  gen_sub_matrix(rank, // PID
		 test_set, // test set
		 0, // matrix number
		 row_chunk, // buffer
		 0, matrix_dimension_size-1, // x dims
		 1, // x stride
		 rank*blocksize, (rank+1)*blocksize-1, // y dims
		 1, // y stride
		 1); // is_row_maj order.
  if(debug_perf == 0) {
    if(rank == 0) {
      printf("argument matrix %d\n", 0);
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
  
  for(int i = 1; i < num_arg_matrices; i++) {
    // for each matrix: split up the columns and then multi. then shift and multi again.
    gen_sub_matrix(rank, // PID
		   test_set, // test set
		   i, // matrix number
		   col_chunk, // buffer
		   rank*blocksize, (rank+1)*blocksize-1, // x dims
		   1, // x stride
		   0, matrix_dimension_size-1, // y dims
		   1, // y stride
		   1); // is_row_maj order.
    
    if(debug_perf == 0) {
      if(rank == 0) {
	printf("argument matrix %d\n", i);
	double* p_buf = my_malloc(sizeof(double) * blocksize);
	for(int p_row = 0; p_row < matrix_dimension_size; p_row++) {
	  for(int p_node = 0; p_node < size; p_node++) {
	    if(p_node == 0) {
	      for (int p_i = 0; p_i < blocksize; p_i++) {
		printf("%f ", col_chunk[p_row * blocksize + p_i]);
	      }
	    }
	    else {
	      MPI_Recv(p_buf, blocksize, MPI_DOUBLE, p_node, 255, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      for (int p_i = 0; p_i < blocksize; p_i++) {
		printf("%f ", p_buf[p_i]);
	      }
	    }
	  }
	  printf("\n");
	}
	printf("\n");
      }
      else{
	for (int p_i = 0; p_i < matrix_dimension_size; p_i++) {
	  MPI_Send(&col_chunk[p_i*blocksize], blocksize, MPI_DOUBLE, 0, 255, MPI_COMM_WORLD);
	}
      }
    }
    
    for (int cur_block_iter=0; cur_block_iter < size; cur_block_iter++) {
      for(int j = 0; j < blocksize; j++){
	for (int k = 0; k < blocksize; k++) {
	  int row_idx = j;
	  int col_idx = (k+((rank+cur_block_iter)*blocksize))%matrix_dimension_size;
	  
	  results[((row_idx)*matrix_dimension_size) + col_idx] = dotProd(&row_chunk[j*matrix_dimension_size], // get the jth row in the block
									 &col_chunk[k], // get the kth column in the block
									 blocksize,
									 matrix_dimension_size);
	}
      }
      
      // once we are done with our local stuff we can shove this column chunk off to another node rq.
      MPI_Bsend(col_chunk, blocksize*matrix_dimension_size, MPI_DOUBLE, (rank+1) % size, 0, MPI_COMM_WORLD);
      MPI_Recv(col_chunk, blocksize*matrix_dimension_size, MPI_DOUBLE, (rank-1) < 0 ? size-1 : (rank-1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // get the chunk from the node before us

      MPI_Barrier(MPI_COMM_WORLD);
      
    }
    
    // buffer swap
    double* temp = results;
    results = row_chunk;
    row_chunk = temp;    
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
  else {

    // sum local elements
    double sum = 0.0;
    for(int i = 0; i < blocksize; i++) {
      for(int j = 0; j < matrix_dimension_size; j++) {
	sum += row_chunk[i * matrix_dimension_size + j];
      }
    }

    // odd send to even. then repeat.
    int nodes_this_step = size;
    for (int i = 0; nodes_this_step > 1; i++) {
      if(rank % (1<<i)) {
	break;
      }
      if(rank == size-(1<<i) || rank == size-(2*(1<<i))) {
	if(nodes_this_step % 2 == 1) {
	  if(rank == size-(1<<i)) {
	    MPI_Send(&sum, 1, MPI_DOUBLE, size-(2*(1<<i)), 10, MPI_COMM_WORLD);
	    break;
	  }
	  else {
	    double rec_sum;
	    MPI_Recv(&rec_sum, 1, MPI_DOUBLE, size-(1<<i), 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    sum += rec_sum;
	  }
	}
      }
      if(rank % (1<<(i+1)) == 0) {
	double rec_sum;
	MPI_Recv(&rec_sum, 1, MPI_DOUBLE, rank+(1<<i), 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	sum += rec_sum;
      } else {
	MPI_Send(&sum, 1, MPI_DOUBLE, rank-(1<<i), 10, MPI_COMM_WORLD);
      }
      nodes_this_step = nodes_this_step >> 1;
    }
    if(rank == 0) {
      printf("%f\n", sum);
    }
  }
  
  // MPI_Recv(void* data, int count, MPI_Datatype datatype, int from, int tag, MPI_Comm comm, MPI_Status* status);
  // MPI_Send(void* data, int count, MPI_Datatype datatype, int to,   int tag, MPI_Comm comm);

  buffer_res = MPI_Buffer_detach(send_buf, &send_buf_size);
  if(buffer_res) {
    printf("rank %d failed to detach the buffer.\n", rank);
    exit(-1);
  }
  
  MPI_Finalize();
  return 0;
}

