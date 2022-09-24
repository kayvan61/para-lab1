test_mm: test_mm.c gen_matrix.c my_malloc.c gen_matrix.h my_malloc.h
	/scratch/04347/marui/cilk/build/bin/clang -fopencilk -DDEBUG -O3 -fno-tree-vectorize -mno-avx -mno-avx2 -mno-mmx -mno-fma -mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -msoft-float -msse -msse2 -msse3 -msse4 test_mm.c gen_matrix.c my_malloc.c -o test_mm

run_debug:
	time ./test_mm 0 0 10000

run_performance:
	time ./test_mm 1 0 10000

clean:
	rm *~; rm *.exe
