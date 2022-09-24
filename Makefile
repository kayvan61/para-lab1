test_mm: test_mm.c gen_matrix.c my_malloc.c gen_matrix.h my_malloc.h 
	gcc -g -DDEBUG test_mm.c gen_matrix.c my_malloc.c -o test_mm

cube_mm: gen_matrix.c my_malloc.c gen_matrix.h my_malloc.h cube_mm.c
	mpicc ./cube_mm.c gen_matrix.c my_malloc.c -o cube_mm

row_chu: gen_matrix.c my_malloc.c gen_matrix.h my_malloc.h row_col_chunks.c
	mpicc row_col_chunks.c gen_matrix.c my_malloc.c -o row_chu

run_debug:
	./test_mm 0 0 100

run_performance:
	./test_mm 1 0 100

clean:
	rm *~; rm *.exe
