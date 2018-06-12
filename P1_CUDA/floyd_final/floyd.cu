#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blocksize1D 1024
#define blocksize2D 32
#define blocksizeReduction 1024

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************
__global__ void floyd_kernel_1D(int * M, const int nverts, const int k) 
{
  int ij = threadIdx.x + blockDim.x * blockIdx.x;
  if (ij < nverts * nverts) {
    int Mij= M[ij];
    int i= ij / nverts; 	// fila de la matriz
    int j= ij - i * nverts;	// columna de la matriz
    if (i != j && i != k && j != k){
	int Mikj = M[i*nverts+k] + M[k*nverts+j];
	Mij = (Mij > Mikj) ? Mikj : Mij;
	M[ij] = Mij;
    }
  }
}


//**************************************************************************
__global__ void floyd_kernel_2D(int * M, const int nverts, const int k) 
{
  int i = blockIdx.y * blockDim.y + threadIdx.y; // fila de la matriz
  int j = blockIdx.x * blockDim.x + threadIdx.x; // columna de la matriz

  if (i < nverts && j < nverts) {
     int ij = i*nverts+j; 	// indice de la matriz como array
     int Mij = M[ij];
     if (i != j && i != k && j != k){
	int Mikj = M[i*nverts+k] + M[k*nverts+j];
	Mij = (Mij > Mikj) ? Mikj : Mij;
	M[ij] = Mij; 
     }
  }
}

//***************************************************************************
// Se ha seguido el modelo de: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

// M es la matriz de entrada (la que resulta del algoritmo de floyd) en memoria global
// M_out es la matriz de salida en memoria global, donde se escribirán los resultados de la
// reducción por cada bloque
// elems es el numero de elementos de la matriz/vector a reducir
__global__ void floyd_reduction(int * M, int * M_out, int elems)
{
  __shared__ int sdata[blocksizeReduction];
  int ij = blockIdx.x * blockDim.x + threadIdx.x;
  if (ij < elems){
 
    // Copiamos a memoria compartida
    int tid = threadIdx.x;
    sdata[tid] = M[ij];
    __syncthreads();

    // fase de reducción
    int n; // primer valor a comparar
    int m; // segundo valor a comparar
    for (int s = blockDim.x/2; s>0; s >>= 1) {
     	if (tid < s) {
          n = sdata[tid];
          m = sdata[tid+s];
		  sdata[tid] = (n > m) ? n : m;
       	}
     	__syncthreads(); // Para que la siguiente iteración no empiece a calcular si alguna 
		         // hebra no ha terminado su reducción
    }
  }

  // escribimos la reduccion final del bloque
  if (threadIdx.x == 0) M_out[blockIdx.x] = sdata[0]; 

}


int main (int argc, char *argv[]) {

  if (argc != 2) {
	cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
	return(-1);
  }

  // This will pick the best possible CUDA capable device
  // int devID = findCudaDevice(argc, (const char **)argv);

  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess)
	cout << "CUDA GET DEVICE ERROR" << endl;

  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n\n", devID, props.name, props.major, props.minor);

  Graph G;
  G.lee(argv[1]);// Read the Graph

  //cout << "EL Grafo de entrada es:"<<endl;
  //G.imprime();

  const int nverts  = G.vertices;
  const int niters  = nverts;
  const int nverts2 = nverts * nverts;
  
  // Variables para floyd
  int *h_Out_M_1D = new int[nverts2];
  int *h_Out_M_2D = new int[nverts2];
  int size        = nverts2*sizeof(int);
  int *d_In_M_1D  = NULL;
  int *d_In_M_2D  = NULL;

  // Variables para la reducción
  int nBlocksReduction = ceil(float(nverts2)/blocksizeReduction);
  int * h_Out_M_reduction = new int[nBlocksReduction];
  int * d_In_M_reduction  = NULL;
  int * d_Out_M_reduction = NULL;

  err = cudaMalloc((void **) &d_In_M_1D, size);
  if (err != cudaSuccess) cout << "ERROR RESERVA 1D" << endl;
  
  err = cudaMalloc((void **) &d_In_M_2D, size);
  if (err != cudaSuccess) cout << "ERROR RESERVA 1D" << endl;

  err = cudaMalloc((void **) &d_In_M_reduction, size);  
  if (err != cudaSuccess) cout << "ERROR RESERVA REDUCTION (IN)" << endl;
  
  err = cudaMalloc((void **) &d_Out_M_reduction, nBlocksReduction*sizeof(int));
  if (err != cudaSuccess) cout << "ERROR RESERVA REDUCTION (OUT)" << endl;
  
  int *A = G.Get_Matrix();





  /********************** GPU phase (1D) ************************/
  double t1 = cpuSecond();

  // Copia de host a device
  err = cudaMemcpy(d_In_M_1D, A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) cout << "ERROR COPIA 1D" << endl;

  for(int k = 0; k < niters; k++) {
  	//printf("CUDA kernel 1D launch \n");
  	int threadsPerBlock = blocksize1D;
  	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;
  
  	floyd_kernel_1D<<<blocksPerGrid,threadsPerBlock>>>(d_In_M_1D, nverts, k);
  	err = cudaGetLastError();

  	if (err != cudaSuccess) {
  		fprintf(stderr, "Failed to launch 1D kernel!\n");
  		exit(EXIT_FAILURE);
	}
  }
  
  // copia de device a host del resultado del algoritmo de floyd
  cudaMemcpy(h_Out_M_1D, d_In_M_1D, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double Tgpu_1D = cpuSecond()-t1;

  cout << "Tiempo gastado Floyd GPU (1D) = " << Tgpu_1D << endl;





  /************************ GPU phase (2D) ********************/

  t1 = cpuSecond();

  // Copia de host a device
  err = cudaMemcpy(d_In_M_2D, A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) cout << "ERROR COPIA 2D" << endl;

  dim3 threadsPerBlock2D(blocksize2D, blocksize2D);
  dim3 blocksPerGrid2D(ceil((float)nverts/blocksize2D),
		       ceil((float)nverts/blocksize2D) );

  for(int k = 0; k < niters; k++) {
  	//printf("CUDA kernel 2D launch \n");
  	floyd_kernel_2D<<<blocksPerGrid2D,threadsPerBlock2D>>>(d_In_M_2D, nverts, k);
  	err = cudaGetLastError();

  	if (err != cudaSuccess) {
  		fprintf(stderr, "Failed to launch 2D kernel!\n");
  		exit(EXIT_FAILURE);
	}
  }
  
  // copia de device a host del resultado del algoritmo de floyd
  cudaMemcpy(h_Out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double Tgpu_2D = cpuSecond()-t1;
  cout << "Tiempo gastado Floyd GPU (2D) = " << Tgpu_2D << endl;







  /************************** CPU Floyd phase ***********************/
  t1 = cpuSecond();

  // BUCLE PPAL DEL ALGORITMO
  int inj, in, kn;
  for(int k = 0; k < niters; k++) {
	kn = k * nverts;
	for(int i=0;i<nverts;i++) {
		in = i * nverts;
		for(int j = 0; j < nverts; j++)
	    		if (i!=j && i!=k && j!=k){
		 	    inj = in + j;
		 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	        	}
	}
  }

  double t2 = cpuSecond() - t1;
  cout << "Tiempo gastado Floyd CPU = " << t2 << endl << endl;






  /************************ GPU Reduction Phase ********************/
  // Copia de host a device
  t1 = cpuSecond();

  err = cudaMemcpy(d_In_M_reduction, h_Out_M_2D, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) cout << "ERROR COPIA REDUCTION" << endl;
  
  floyd_reduction<<<nBlocksReduction, blocksizeReduction>>>(d_In_M_reduction, 
	d_Out_M_reduction, nverts*nverts);
  err = cudaGetLastError();

  if (err != cudaSuccess){
     fprintf(stderr, "Failed to launch reduction kernel!\n");
     exit(EXIT_FAILURE);
  }
  

  // Copia de device a host del resultado de la reducción
  cudaMemcpy(h_Out_M_reduction, d_Out_M_reduction, nBlocksReduction*sizeof(int), 
	cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int mayor_gpu = h_Out_M_reduction[0];
  for (int i = 1; i < nBlocksReduction; i++){
	int val = h_Out_M_reduction[i];
	mayor_gpu = (val > mayor_gpu) ? val : mayor_gpu;
  }
  double Tgpu_red = cpuSecond() - t1;
  cout << "Tiempo gastado reducción GPU = " << Tgpu_red << ". Resultado: " << mayor_gpu << endl;



  /*********************** CPU Reduction phase *********************/
  
  t1 = cpuSecond();

  int mayor_cpu = h_Out_M_2D[0];
  for (int i = 0; i < nverts; i++)
    for (int j = 0; j < nverts; j++){
	int val = h_Out_M_2D[i*nverts+j];
	mayor_cpu = (val > mayor_cpu) ? val : mayor_cpu;
    }
  double Tcpu_red = cpuSecond() - t1;
  cout << "Tiempo gastado reducción CPU = " << Tcpu_red << ". Resultado: " << mayor_cpu << endl;     







  /************************ Resultados ************************/
  cout << endl << "Ganancia Floyd 1D= " << t2 / Tgpu_1D << endl;
  cout << "Ganancia Floyd 2D= " << t2 / Tgpu_2D << endl;
  cout << "Ganancia Reduccion 1D= " << Tcpu_red / Tgpu_red << endl;
  for(int i = 0; i < nverts; i++)
    for(int j = 0; j < nverts; j++)
       if (abs(h_Out_M_1D[i*nverts+j] - G.arista(i,j)) > 0)
	 cout << "1D[" << i*nverts+j << "]=" << h_Out_M_1D[i*nverts+j] << ", 2D[" << i*nverts+j << "]=" << h_Out_M_2D[i*nverts+j] << endl;
  
  for(int i = 0; i < nverts; i++)
    for(int j = 0; j < nverts; j++)
       if (abs(h_Out_M_2D[i*nverts+j] - G.arista(i,j)) > 0)
         cout << "Error 2D (" << i << "," << j << ")   " << h_Out_M_2D[i*nverts+j] << "..." << G.arista(i,j) << endl;

  // free host memory
  delete(h_Out_M_1D);
  delete(h_Out_M_2D);
  delete(h_Out_M_reduction);

}
