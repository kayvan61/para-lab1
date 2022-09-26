test_mm: test_mm.c gen_matrix.c my_malloc.c gen_matrix.h my_malloc.h
	/scratch/04347/marui/cilk/build/bin/clang -fopencilk -DDEBUG -O3 -fno-tree-vectorize -mno-avx -mno-avx2 -mno-mmx -mno-fma -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 test_mm.c gen_matrix.c my_malloc.c -o test_mm

run_debug:
  #CILK_NWORKERS=1 time ./test_mm 0 0 1000
	time ./test_mm 0 0 1024

run_performance:
  #CILK_NWORKERS=1 time ./test_mm 1 0 1024
	time ./test_mm 1 0 1024

clean:
	rm *~; rm *.exe
