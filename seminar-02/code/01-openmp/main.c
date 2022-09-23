#include<stdio.h>
#include<stdlib.h>
#include<omp.h>


double f(double x){
	return x*x + 2*x - 7;
}

int main(int argc, char* argv[]){

#ifdef _OPENMP
	printf("OpenMP is supported! %d \n", _OPENMP);
#endif

	int i = 0;
	int myid;
	long const N = 1e6;
	int num_threads, num_procs, max_thr;

	double integral, s, S, dx;
	double t_in, t_out;

	dx = 1 / (double)N;
	printf("dx = %e \n", dx);
	
	max_thr = omp_get_max_threads();
	num_procs = omp_get_num_procs();

	printf("Max threads = %d \n", max_thr);
	printf("Num of processors = %d \n", num_procs);

	printf("----------------------------\n");
	
	omp_set_num_threads(4);
	
	num_threads = omp_get_num_threads();  

	printf("Now num of threads = %d \n", num_threads);

	S = 0.0;
	integral = 0.0;
	
	myid = omp_get_thread_num();   
    	
	printf("Consecutive part 1, myid = %d\n", myid);

	t_in = omp_get_wtime();
	printf("----------------------------\n");

#pragma omp parallel shared (S), private(num_threads, myid, s, i)
	{
		float part = 0;
		myid = omp_get_thread_num();
		num_threads = omp_get_num_threads();
		
		s = 0.0;
			
		printf("Parallel part. I'm thread number %d, num_threads = %d \n", myid, num_threads);
		
		#pragma omp for 
			for (i = 1; i <= N; i++){
				part = 	( f(dx * i) + f(dx * (i - 1)) ) * dx / 2;
				s = s + part;
				integral = integral + part;
			}
		
		#pragma  omp  critical
		{
			S = S + s;
		}
	}
	t_out = omp_get_wtime();
	printf("Consecutive part 2 integral with critical section, S = %f \n", 4 * S);
	printf("Consecutive part 2 integral without critical section, S = %f \n", 4 * integral);
	
	printf("----------------------------\n");
	printf("Work time = %f \n", t_out - t_in);
	printf("----------------------------\n");
		
	myid = omp_get_thread_num();
	num_threads = omp_get_num_threads();
	printf("Consecutive part 2 (after parallel section), myid = %d, num_threads = %d \n", myid, num_threads);

	printf("\n");
	return 0;

}

