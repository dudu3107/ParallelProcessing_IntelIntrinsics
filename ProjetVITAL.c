// Student: Eduardo Vital Brasil; Mines Paristech HPC-AI

#include <time.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <sys/time.h> // for timing
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>

// COMPILING COMMAND FOR GCC
// gcc ProjetVITAL.c -o ProjetVITAL.out -mavx -mavx2 -lm -lpthread

#define N 1048576  // 1024*1024
double *rs;
pthread_mutex_t mutexsum;
double U2[N] __attribute__((aligned(32)));
double U3[N] __attribute__((aligned(32)));
double U4[N] __attribute__((aligned(32)));
double vec[N] __attribute__((aligned(32)));

double now(){
   struct timeval t; double f_t;
   gettimeofday(&t, NULL);
   f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
   return f_t;
}

double randfrom(double min, double max) // Function to generate random numbers within bounds
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

double norm4(double *U, double a, double b, double c, double d, int n) {
	double sum = 0;
	int i;

	for ( i = 0 ;  i < ((int)(n/4)); i++){
		sum +=  sqrt(a*a*U[4*i] + b*b*U[4*i+1] + c*c*U[4*i+2] + d*d*U[4*i+3]);
	};
	return sum;
};

double vect_norm4(double *U, double a, double b, double c, double d, int n) {  // Initial Simple Version - Time won't be printed
	double sum = 0;
	unsigned int i;
	
	int nb_iters = n / 4; 
	__m256d coefSqr = _mm256_set_pd(d * d, c * c, b * b, a * a);
	
	__m256d *mm_coefsSqr =  &coefSqr;
	__m256d *mm_vec = (__m256d*) vec;

	__m256d *mm_U = (__m256d*)U;

	for(i = 0; i < nb_iters; i++) {                                                                                                                                                            
		*mm_vec = _mm256_mul_pd(*mm_coefsSqr, mm_U[i]);
		sum += sqrt(vec[0] + vec[1] + vec[2] + vec[3]);
	};
	return sum;
};

double vect_norm4_2(double *U, double a, double b, double c, double d, int n) {  // Try with different version - Time won't be printed
	double sum = 0;
	unsigned int i;
	
	int nb_iters = n / 4; 
	__m256d coefSqr = _mm256_set_pd(d * d, c * c, b * b, a * a);
	
	__m256d *mm_coefsSqr =  &coefSqr;
	__m256d mm_mul; //, mm_res;
	__m128d vlow_1, vlow_2, vlow_3, vlow_4;
	__m256d *mm_U = (__m256d*)U;
	__m256d *mm_vec = (__m256d*) vec;

	for(i = 0; i < nb_iters; i+=4) {                                                                                                                                                            
		
		mm_mul = _mm256_mul_pd(*mm_coefsSqr, mm_U[i]);
		vlow_1 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul), _mm256_extractf128_pd(mm_mul, 1));
		
		mm_mul = _mm256_mul_pd(*mm_coefsSqr, mm_U[i+1]);
		vlow_2 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul), _mm256_extractf128_pd(mm_mul, 1));
		
		mm_mul = _mm256_mul_pd(*mm_coefsSqr, mm_U[i+2]);
		vlow_3 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul), _mm256_extractf128_pd(mm_mul, 1));
		
		mm_mul = _mm256_mul_pd(*mm_coefsSqr, mm_U[i+3]);
		vlow_4 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul), _mm256_extractf128_pd(mm_mul, 1));
		
		// mm_res =
		*mm_vec = _mm256_sqrt_pd(
						_mm256_set_m128d(
							_mm_move_sd(
									_mm_permute_pd(_mm_add_sd(vlow_1, _mm_unpackhi_pd(vlow_1, vlow_1)), 1), _mm_add_sd(vlow_2, _mm_unpackhi_pd(vlow_2, vlow_2))),
							_mm_move_sd(
									_mm_permute_pd(_mm_add_sd(vlow_3, _mm_unpackhi_pd(vlow_3, vlow_3)), 1), _mm_add_sd(vlow_4, _mm_unpackhi_pd(vlow_4, vlow_4)))
							)
						);
		//mm_res_low = _mm_add_pd(_mm256_castpd256_pd128(mm_res), _mm256_extractf128_pd(mm_res, 1));
		
		//sum += _mm_cvtsd_f64(_mm_add_sd(mm_res_low, _mm_unpackhi_pd(mm_res_low, mm_res_low)));
		sum += vec[0] + vec[1] + vec[2] + vec[3];
	};
	return sum;
};

double vect_norm4_3(double *U, double a, double b, double c, double d, int n) {  // Modified version 2 for improvement - Time won't be printed
	double sum = 0;
	unsigned int i;
	
	int nb_iters = n / 4; 
	__m256d mm_coefsSqr = {d * d, c * c, b * b, a * a};

	__m256d mm_vec, mm_mul1, mm_mul2, mm_mul3, mm_mul4;
	__m128d vlow_1, vlow_2, vlow_3, vlow_4;
	__m256d *mm_U = (__m256d*)U;

	for(i = 0; i < nb_iters; i+=4) {                                                                                                                                                            
		
		mm_mul1 = _mm256_mul_pd(mm_coefsSqr, mm_U[i]);
		vlow_1 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul1), _mm256_extractf128_pd(mm_mul1, 1));
		
		mm_mul2 = _mm256_mul_pd(mm_coefsSqr, mm_U[i+1]);
		vlow_2 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul2), _mm256_extractf128_pd(mm_mul2, 1));
		
		mm_mul3 = _mm256_mul_pd(mm_coefsSqr, mm_U[i+2]);
		vlow_3 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul3), _mm256_extractf128_pd(mm_mul3, 1));
		
		mm_mul4 = _mm256_mul_pd(mm_coefsSqr, mm_U[i+3]);
		vlow_4 = _mm_add_pd(_mm256_castpd256_pd128(mm_mul4), _mm256_extractf128_pd(mm_mul4, 1));
		
		mm_vec = _mm256_sqrt_pd(
						_mm256_set_m128d(
							_mm_move_sd(
									_mm_permute_pd(_mm_add_sd(vlow_1, _mm_unpackhi_pd(vlow_1, vlow_1)), 1), _mm_add_sd(vlow_2, _mm_unpackhi_pd(vlow_2, vlow_2))),
							_mm_move_sd(
									_mm_permute_pd(_mm_add_sd(vlow_3, _mm_unpackhi_pd(vlow_3, vlow_3)), 1), _mm_add_sd(vlow_4, _mm_unpackhi_pd(vlow_4, vlow_4)))
							)
						);
		sum += mm_vec[0] + mm_vec[1] + mm_vec[2] + mm_vec[3];
	};
	return sum;
};

double vect_norm4_4(double *U, double a, double b, double c, double d, int n) {  // Optimum version - Fastest of all
	double sum = 0;
	unsigned int i;
	
	int nb_iters = n / 4; 
	__m256d mm_coefsSqr = {a * a, b * b, c * c, d * d};

	__m256d mm_vec;
	__m256d *mm_U = (__m256d*)U;

	for(i = 0; i < nb_iters; i+=4) {                                                                                                                                                            
		
		mm_vec = _mm256_sqrt_pd(_mm256_hadd_pd(_mm256_permute4x64_pd(_mm256_hadd_pd(_mm256_mul_pd(mm_coefsSqr, mm_U[i]), _mm256_mul_pd(mm_coefsSqr, mm_U[i+1])), 0b11011000), _mm256_permute4x64_pd(_mm256_hadd_pd(_mm256_mul_pd(mm_coefsSqr, mm_U[i+2]), _mm256_mul_pd(mm_coefsSqr, mm_U[i+3])), 0b11011000)));

		sum += mm_vec[0] + mm_vec[1] + mm_vec[2] + mm_vec[3];
	};
	return sum;
};

struct thread_data{
  unsigned int thread_id;
  double *U;
  int begin_t;
  int end_t;
  double a2;
  double b2;
  double c2;
  double d2;
  unsigned char mode;
};

void *thread_function(void *threadarg){

  double s = 0;
  long i, id;

  double *U;
  int begin_t, end_t;
  double a2, b2, c2, d2;
  unsigned char mode;

 /* Association between shared variables and their correspondances */
  struct thread_data *thread_pointer_data;
  thread_pointer_data = (struct thread_data *)threadarg; 
  /* Shared variables */
  id = thread_pointer_data->thread_id;
  U = thread_pointer_data->U;
  begin_t = thread_pointer_data->begin_t;
  end_t = thread_pointer_data->end_t;
  a2 = thread_pointer_data->a2;
  b2 = thread_pointer_data->b2;
  c2 = thread_pointer_data->c2;
  d2 = thread_pointer_data->d2;
  mode = thread_pointer_data->mode;
 
  /* Body of the thread */
  
  if (mode == 0) {
	  for(i=begin_t;i<end_t;i=i+4) {
		s += sqrt(a2*U[i] + b2*U[i+1] + c2*U[i+2] + d2*U[i+3]);;
	  };
  }else {

		__m256d mm_coefsSqr = {a2, b2, c2, d2};
		__m256d mm_vec;
		__m256d *mm_U = (__m256d*)U2;

		for(i = begin_t/4; i < end_t/4; i+=4) {                                                                                                                                                            
			mm_vec = _mm256_sqrt_pd(_mm256_hadd_pd(_mm256_permute4x64_pd(_mm256_hadd_pd(_mm256_mul_pd(mm_coefsSqr, mm_U[i]), _mm256_mul_pd(mm_coefsSqr, mm_U[i+1])), 0b11011000), _mm256_permute4x64_pd(_mm256_hadd_pd(_mm256_mul_pd(mm_coefsSqr, mm_U[i+2]), _mm256_mul_pd(mm_coefsSqr, mm_U[i+3])), 0b11011000)));
			s += mm_vec[0] + mm_vec[1] + mm_vec[2] + mm_vec[3];
		};
	}
  
  // pthread_mutex_lock(&mutexsum);  // Another option to save thr results: a Global variable; proved itself slower
  // sumPar += s;
  // pthread_mutex_unlock(&mutexsum);
  rs[id] = s;

  pthread_exit(NULL);
  return 0;
}

double norm4Par(double *U, double a, double b, double c, double d, int n, int nb_threads, unsigned char mode){
	long i;
	double sumPar = 0;
	rs = malloc(nb_threads*sizeof(double)); // Store the results "s"
	struct thread_data thread_data_array[nb_threads];  // Vector with struct elements
	unsigned int thread_i;
	pthread_t thread_ptr[nb_threads];  // Vector with pthread_t elements
	
	for(i=0;i<nb_threads;i++){
		thread_i = i;
		/* Prepare data for this thread */
		thread_data_array[thread_i].thread_id = thread_i;
		thread_data_array[thread_i].U = U;
		thread_data_array[thread_i].begin_t = i*(n/nb_threads);
		thread_data_array[thread_i].end_t = (i+1)*(n/nb_threads);
		thread_data_array[thread_i].a2 = a*a;
		thread_data_array[thread_i].b2 = b*b;
		thread_data_array[thread_i].c2 = c*c;
		thread_data_array[thread_i].d2 = d*d;
		thread_data_array[thread_i].mode = mode;
		/* Create and launch this thread */
		pthread_create(&thread_ptr[thread_i], NULL, thread_function, (void *) &thread_data_array[thread_i]);
	}
	/* Wait for every thread to complete  */
	for(i=0;i<nb_threads;i++){
	  pthread_join(thread_ptr[i], NULL);
	}
	/* Calculate result from all threads  */
	for(i=0;i<nb_threads;i++) sumPar += rs[i];
	
	return sumPar;

};

int main(int argc, char *argv[]){
	long j;
	double t0, t1, t2, t3, t4, t5, t6, t7;
	int n = 1048576;  // 1024*1024
	int nb_threads = 8;
	int time_iters = 100;  // Number of callings of each function for better accracy of the acceleration result
	int a = 1;
	int b = 2;
	int c = 3;
	int d = 4;
	unsigned char mode; // (0 : scalaire et 1 : vectoriel
	double *U1, s;
	
	U1 = malloc(n*sizeof(double));
	
	//srand(20);
	srand (time (NULL));  // Will output a different result at each time
	for(j=0;j<n;j++) {
		 U1[j] = randfrom(0, 100);
		 U2[j] = randfrom(0, 100);
		 U3[j] = randfrom(0, 100);
		 U4[j] = randfrom(0, 100);
	 }
	
	 t0 = now();
	 for (int i=0; i < time_iters; i++) {
	 s = norm4(U1, a, b, c, d, n);
	 }
	 t1 = now();
	 printf("\nScalar = %8.6f   time: %8.6f\n", s, (t1 - t0)/time_iters);
	 
	t2 = now();
	for (int i=0; i < time_iters; i++) {
	s = vect_norm4_4(U2, a, b, c, d, n);
	}
	t3 = now();
	printf("Vectorial; result = %8.6f   time: %8.6f\n", s, (t3-t2)/time_iters);
	 
	mode=0;
	t4 = now();
	for (int i=0; i < time_iters; i++) {
	s = norm4Par(U3, a, b, c, d, n, nb_threads, mode);
	}
	t5 = now();
	printf("Multithread + Scalar; result = %8.6f   time: %8.6f\n", s, (t5-t4)/time_iters);

	mode=1;
	t6 = now();
	for (int i=0; i < time_iters; i++) {
	s = norm4Par(U4, a, b, c, d, n, nb_threads, mode);
	}
	t7 = now();
	printf("Multithread + Vectorial; result = %8.6f   time: %8.6f\n", s, (t7-t6)/time_iters);
	
	printf("\n --- Accelerations in comparison with scalar version --- \n");
	printf("Vectorial = %6.4f\n",(t1-t0)/(t3-t2));  
	printf("Multithread + Scalar = %6.4f\n",(t1-t0)/(t5-t4));
	printf("Multithread + Vectorial = %6.4f\n\n",(t1-t0)/(t7-t6));	
    
	return 0;
}


