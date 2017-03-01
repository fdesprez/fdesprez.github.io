#include <stdio.h>
#include <omp.h>

#include <x86intrin.h>

#define NBEXPERIMENTS    22
static long long unsigned int experiments [NBEXPERIMENTS] ;


#define N              512
#define TILE           16

typedef double vector [N] ;

typedef double matrix [N][N] ;

static vector a, b, c ;
static matrix M1, M2 ;

long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
    {
      s = s + exps [i] ;
    }

  return s / (NBEXPERIMENTS-2) ;
}


void init_vector (vector X, const double val)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    X [i] = val ;

  return ;
}

void init_matrix (matrix X, const double val)
{
  register unsigned int i, j;

  for (i = 0; i < N; i++)
    {
      for (j = 0 ;j < N; j++)
	{
	  X [i][j] = val ;
	}
    }
}

  
void print_vectors (vector X, vector Y)
{
  register unsigned int i ;

  for (i = 0 ; i < N; i++)
    printf (" X [%d] = %le Y [%d] = %le\n", i, X[i], i,Y [i]) ;

  return ;
}

void add_vectors1 (vector X, vector Y, vector Z)
{
  register unsigned int i ;

#pragma omp parallel for schedule(static)  
  for (i=0; i < N; i++)
    X[i] = Y[i] + Z[i];
  
  return ;
}

void add_vectors2 (vector X, vector Y, vector Z)
{
  register unsigned int i ;

#pragma omp parallel for schedule(dynamic)  
  for (i=0; i < N; i++)
    X[i] = Y[i] + Z[i];
  
  return ;
}

double dot1 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;

  
  dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction (+:dot)
  for (i=0; i < N; i++)
    dot += X [i] * Y [i];

  return dot ;
}

double dot2 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;


  dot = 0.0 ;
#pragma omp parallel for schedule(dynamic) reduction (+:dot)
  for (i=0; i < N; i++)
    dot += X [i] * Y [i];

  return dot ;
}

double dot3 (vector X, vector Y)
{
  register unsigned int i ;
  register double dot ;

  dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction (+:dot)
  for (i = 0; i < N; i = i + 8)
    {
    dot += X [i] * Y [i];
    dot += X [i + 1] * Y [i + 1];
    dot += X [i + 2] * Y [i + 2];
    dot += X [i + 3] * Y [i + 3];
    
    dot += X [i + 4] * Y [i + 4];
    dot += X [i + 5] * Y [i + 5];
    dot += X [i + 6] * Y [i + 6];
    dot += X [i + 7] * Y [i + 7];
    }

  return dot ;
}

void mult_mat_vect0 (matrix M, vector b, double *c)
{
  register unsigned int i ;
  register unsigned int j ;
  register double r ;
  
  for ( i = 0 ; i < N ; i = i + 1)
    {
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 1)
	{
	  r += M [i][j] * b [j] ;
	}
      c [i] = r ;
    }
  
  return ;
}

void mult_mat_vect1 (matrix M, vector b, vector c)
{
  register unsigned int i ;
  register unsigned int j ;
  register double r ;
  
#pragma omp parallel for private (i,j,r) schedule (static)  
  for ( i = 0 ; i < N ; i = i + 1)
    {
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 1)
	{
	  r += M [i][j] * b [j] ;
	}
      c [i] = r ;
    }
  
  return ;
}

void mult_mat_vect2 (matrix M, vector b, vector c)
{
  register unsigned int i ;
  register unsigned int j ;
  register double r ;
  
#pragma omp parallel for private (i,j,r) schedule (static)  
  for ( i = 0 ; i < N ; i = i + 1)
    {
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 8)
	{
	  r += M [i] [j]   * b [j] ;
	  r += M [i] [j+1] * b [j+1] ;
	  r += M [i] [j+2] * b [j+2] ;
	  r += M [i] [j+3] * b [j+3] ;
	  r += M [i] [j+4] * b [j+4] ;
	  r += M [i] [j+5] * b [j+5] ;
	  r += M [i] [j+6] * b [j+6] ;
	  r += M [i] [j+7] * b [j+7] ;
	}
      c [i] = r ;
    }
  
  return ;
}

void mult_mat_vect3 (matrix M, vector b, vector c)
{
  register unsigned int i ;
  register unsigned int j ;
  register double r ;
  
#pragma omp parallel for private (i,j,r) schedule (static)  
  for ( i = 0 ; i < N ; i = i + 4)
    {
      /* i */
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 8)
	{
	  r += M [i] [j]   * b [j] ;
	  r += M [i] [j+1] * b [j+1] ;
	  r += M [i] [j+2] * b [j+2] ;
	  r += M [i] [j+3] * b [j+3] ;
	  r += M [i] [j+4] * b [j+4] ;
	  r += M [i] [j+5] * b [j+5] ;
	  r += M [i] [j+6] * b [j+6] ;
	  r += M [i] [j+7] * b [j+7] ;
	}

      c [i] = r ;
      /* i + 1 */
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 8)
	{
	  r += M [i+1] [j]   * b [j] ;
	  r += M [i+1] [j+1] * b [j+1] ;
	  r += M [i+1] [j+2] * b [j+2] ;
	  r += M [i+1] [j+3] * b [j+3] ;
	  r += M [i+1] [j+4] * b [j+4] ;
	  r += M [i+1] [j+5] * b [j+5] ;
	  r += M [i+1] [j+6] * b [j+6] ;
	  r += M [i+1] [j+7] * b [j+7] ;
	}

      c [i+1] = r ;

      /* i + 2 */
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 8)
	{
	  r += M [i+2] [j]   * b [j] ;
	  r += M [i+2] [j+1] * b [j+1] ;
	  r += M [i+2] [j+2] * b [j+2] ;
	  r += M [i+2] [j+3] * b [j+3] ;
	  r += M [i+2] [j+4] * b [j+4] ;
	  r += M [i+2] [j+5] * b [j+5] ;
	  r += M [i+2] [j+6] * b [j+6] ;
	  r += M [i+2] [j+7] * b [j+7] ;
	}

      c [i+2] = r ;
      /* i + 3 */
      r = 0.0 ;
      for (j = 0 ; j < N ; j = j + 8)
	{
	  r += M [i+3] [j]   * b [j] ;
	  r += M [i+3] [j+1] * b [j+1] ;
	  r += M [i+3] [j+2] * b [j+2] ;
	  r += M [i+3] [j+3] * b [j+3] ;
	  r += M [i+3] [j+4] * b [j+4] ;
	  r += M [i+3] [j+5] * b [j+5] ;
	  r += M [i+3] [j+6] * b [j+6] ;
	  r += M [i+3] [j+7] * b [j+7] ;
	}
      c [i+3] = r ;
    }
  
  return ;
}


void mult_mat_mat0 (matrix A, matrix B, matrix C)
{
  register unsigned int i, j, k ;
  register double r ;
  
  for (i = 0 ; i < N; i = i + 1)
    {
      for (j = 0 ; j < N; j = j + 1)
	{
	  r = 0.0 ;
	  for (k =0 ; k < N; k = k + 1)
	    {
	      r += A [i][k] * B[k][j] ;
	    }
	  C [i][j] = r ;
	}
    }
  return ;
}


void mult_mat_mat1 (matrix A, matrix B, matrix C)
{
  register unsigned int i, j, k ;
  register double r ;
  
#pragma omp parallel for schedule (static) private (i,j,k,r)
  for (i = 0 ; i < N; i = i + 1)
    {
      for (j = 0 ; j < N; j = j + 1)
	{
	  r = 0.0 ;
	  for (k =0 ; k < N; k = k + 1)
	    {
	      r += A [i][k] * B[k][j] ;
	    }
	  C [i][j] = r ;
	}
    }
  return ;
}

void mult_mat_mat2 (matrix A, matrix B, matrix C)
{
  register unsigned int i, j, k ;
  register double r ;
  
#pragma  omp parallel for schedule (static) private (i,j,k,r)
  for (i = 0 ; i < N; i = i + 1)
    {
      for (j = 0 ; j < N; j = j + 1)
	{
          r = 0.0 ;	  
	  for (k =0 ; k < N; k = k + 8)
	    {
	      r += A [i][k]     * B [k][j] ;
	      r += A [i][k+1]   * B [k+1][j] ;
	      r += A [i][k+2]   * B [k+2][j] ;
	      r += A [i][k+3]   * B [k+3][j] ;
	      r += A [i][k+4]   * B [k+4][j] ;
	      r += A [i][k+5]   * B [k+5][j] ;
	      r += A [i][k+6]   * B [k+6][j] ;
	      r += A [i][k+7]   * B [k+7][j] ;
	    }
	  C [i][j] = r ;
	}
    }
  return ;
}

void mult_mat_mat3 (matrix A, matrix B, matrix C)
{
  register unsigned int i, j, k ;
  register unsigned int ip, jp ;
  register double r ;
  
#pragma omp parallel for schedule (static) private (i,j,k,ip,jp,r)
  for (i = 0 ; i < N; i = i + TILE)
    {
      for (j = 0 ; j < N; j = j + TILE)
	{
	  for (jp = j; jp < (j+TILE); jp = jp + 1)
	    {
	      for (ip = i; ip < (i+TILE); ip = ip +1)
		{
		  r = 0.0 ;
		  /* 
		     unrolling the loop with k 
		  */
	  
		  for (k =0 ; k < N; k = k + 8)
		    {
 		        r += A [ip][k] * B[k][jp] ;
		      
			r += A [ip][k+1] * B[k+1][jp] ;
			r += A [ip][k+2] * B[k+2][jp] ;
			r += A [ip][k+3] * B[k+3][jp] ;
			
			r += A [ip][k+4] * B[k+4][jp] ;
			r += A [ip][k+5] * B[k+5][jp] ;
			r += A [ip][k+6] * B[k+6][jp] ;
			r += A [ip][k+7] * B[k+7][jp] ; 
		      
		    }
		  C[ip][jp] = r ;
		}
	    }
	}
    }
  return ;
}




int main ()  
{
  int nthreads, maxnthreads ;
  
  int tid;

  unsigned long long int start, end ;
  unsigned long long int residu ;

  unsigned long long int av ;
  
  double r ;

  int exp ;
  
  /* 
     rdtsc: read the cycle counter 
  */
  
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;
  
  /* 
     Sequential code executed only by the master thread 
  */	
	
  nthreads = omp_get_num_threads();
  maxnthreads = omp_get_max_threads () ;
  printf("Sequential execution: \n# threads %d\nmax threads %d \n", nthreads, maxnthreads);
	
  /*
    Vector Initialization
  */

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

  /*
    print_vectors (a, b) ;
  */
    

  printf ("=============== ADD ==========================================\n") ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

  /*
    print_vectors (a, b) ;
  */
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         add_vectors1 (c, a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop:\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.6f GFLOP per second\n", (double) N / ((double) (av - residu) * (double) 0.38)) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
	
  /*
    print_vectors (a, b) ;
  */
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         add_vectors2 (c, a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP dynamic loop:\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.6f GFLOP per second\n", (double) N / ((double) (av - residu) * (double) 0.38)) ;

  printf ("==============================================================\n") ;

  printf ("====================DOT =====================================\n") ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;
  
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot1 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("dot OpenMP static loop:\t\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per seconde\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) 0.38)) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot2 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("dot OpenMP dynamic loop:\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per seconde\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) 0.38)) ;

  init_vector (a, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

         r = dot3 (a, b) ;
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("dot OpenMP static unrolled loop:\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per seconde\n", ((double) 2 * (double) N) / ((double) (av- residu) * (double) 0.38)) ;
  
  printf ("=============================================================\n") ;

  printf ("======================== Mult Mat Vector =====================================\n") ;
  
  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect0 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  
  printf ("Sequential matrice vector multiplication:\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per second\n", (((double) 2 * (double) N * (double) N)) / ((double) (av- residu) * (double) 0.38)) ;

  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect1 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
 
  printf ("OpenMP static loop MultMatVect1:\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per second\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;

  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect2 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop MultMatVect2:\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per second\n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;

  init_matrix (M1, 1.0) ;
  init_vector (b, 2.0) ;

   for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

           mult_mat_vect3 (M1, b, a) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
 
  printf ("OpenMP static loop MultMatVect3:\t\t %Ld cycles\n", av-residu) ;
  printf ("%3.3f GFLOP per second \n", (((double) 2 * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;

  printf ("===================================================================\n") ;

  printf ("======================== Mult Mat Mat =====================================\n") ;
  
  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat0 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("Sequential Matrix Matrix Multiplication:\t %Ld cycles\n", av-residu) ;
    printf ("%3.3f GFLOP per second \n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;
  
  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat1 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("OpenMP static loop matrix matrix multiplication: %Ld cycles\n", av-residu) ;
    printf ("%3.3f GFLOP per second \n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;

  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat2 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
    

  printf ("OpenMP unrolled loop matrix matrix multiplication: %Ld cycles\n", av-residu) ;
    printf ("%3.3f GFLOP per second \n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;

  init_matrix (M1, 1.0) ;
  init_matrix (M2, 2.0) ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      start = _rdtsc () ;

      mult_mat_mat3 (M1, M2, M2) ; 
     
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

    printf ("OpenMP Tiled loop MultMatMat3:\t\t\t %Ld cycles\n", av-residu) ;
    printf ("%3.3f GFLOP per second \n", (((double) 2 * (double) N * (double) N * (double) N)) / ((double) (av - residu) * (double) 0.38)) ;
  
  return 0;
  
}

