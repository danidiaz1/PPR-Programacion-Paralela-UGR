#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <sys/time.h>

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************
__global__ void transformacion_kernel_global(float * A, float * B, float * C, 
      float * D, float * mx)
{

  int tid = threadIdx.x;
  int Bsize = blockDim.x;
  int i= tid + Bsize * blockIdx.x;
  float c = 0.0; // valor a calcular
  
  extern __shared__ float sdata[]; // memoria compartida
  float *sdata_A = sdata; 	   // Puntero al primer valor de A
  float *sdata_B = sdata+Bsize;    // Puntero al primer valor de B
  float *sdata_C = sdata+Bsize*2;  // Puntero al primer valor de C
  float *sdata_C2 = sdata+Bsize*3; // Puntero al primer valor de una copia de C
  

  // Paso a memoria compartida de A y B
  *(sdata_A+tid) = A[i];
  *(sdata_B+tid) = B[i];

  __syncthreads();
  
  /***** Fase del calculo de C (memoria global)  *****/

  int jinicio = blockIdx.x * Bsize;
  int jfin = jinicio + Bsize;
  for (int j = jinicio; j < jfin; j++){ 
     float a = A[j] * i;
     int signo = int(ceil(a))%2 == 0 ? 1 : -1;
     c += a + B[j] * signo;
  }

  C[i] = c;
  *(sdata_C+tid) = c;
  *(sdata_C2+tid) = c;

  __syncthreads();


  /***** Fase del calculo de D (reduccion suma) y mx (reduccion >) *****/
  float n, m;
  for (int s=blockDim.x/2; s>0; s>>=1){
     if (tid < s){
        *(sdata_C+tid) += *(sdata_C+tid+s);
        n = *(sdata_C2+tid);
        m = *(sdata_C2+tid+s);
        *(sdata_C2+tid) = (n > m) ? n : m;
     }
     __syncthreads();
  }

  if (tid == 0){
     D[blockIdx.x] = *(sdata_C);
     mx[blockIdx.x] = *(sdata_C2);
  }
  
}


//**************************************************************************
__global__ void transformacion_kernel_shared(float * A, float * B, float * C,
      float * D, float * mx)
{

  int tid = threadIdx.x;
  int Bsize = blockDim.x;
  int i= tid + Bsize * blockIdx.x;
  float c = 0.0; // valor a calcular

  extern __shared__ float sdata[]; // memoria compartida
  float *sdata_A = sdata;          // Puntero al primer valor de A
  float *sdata_B = sdata+Bsize;    // Puntero al primer valor de B
  float *sdata_C = sdata+Bsize*2;  // Puntero al primer valor de C
  float *sdata_C2 = sdata+Bsize*3; // Puntero al primer valor de una copia de C


  // Paso a memoria compartida de A y B
  *(sdata_A+tid) = A[i];
  *(sdata_B+tid) = B[i];

  __syncthreads();

  /***** Fase del calculo de C (memoria compartida)  *****/

  for (int j = 0; j < Bsize; j++){
     float a =  *(sdata_A+j) * i;
     int signo = int(ceil(a))%2 == 0 ? 1 : -1;
     c += a + *(sdata_B+j) * signo;
  }

  C[i] = c;
  *(sdata_C+tid) = c;
  *(sdata_C2+tid) = c;

  __syncthreads();


  /***** Fase del calculo de D (reduccion suma) y mx (reduccion >) *****/
  float n, m;
  for (int s=blockDim.x/2; s>0; s>>=1){
     if (tid < s){
        *(sdata_C+tid) += *(sdata_C+tid+s);
        n = *(sdata_C2+tid);
        m = *(sdata_C2+tid+s);
        *(sdata_C2+tid) = (n > m) ? n : m;
     }
     __syncthreads();
  }

  if (tid == 0){
     D[blockIdx.x] = *(sdata_C);
     mx[blockIdx.x] = *(sdata_C2);
  }

}

//**************************************************************************
int main(int argc, char *argv[])
//**************************************************************************
{
  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess)
	cout << "CUDA GET DEVICE ERROR" << endl;

  cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n\n", devID, props.name, props.major, props.minor);

  int Bsize, NBlocks;
  if (argc != 3){ 
	cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
	return(0);
  }else{
	NBlocks = atoi(argv[1]);
   	Bsize= atoi(argv[2]);
  }

  const int N=Bsize*NBlocks;

  // Pointers to memory
  float *h_A, *h_B, *h_C, *h_D, *h_D_global, *h_mx_global, h_mx, *h_D_shared, *h_mx_shared; //host
  float *d_A, *d_B, *d_C, *d_D_global, *d_mx_global, *d_D_shared, *d_mx_shared; // device

  // Allocate arrays a, b, c and d on host
  h_A = new float[N];
  h_B = new float[N];
  h_C = new float[N];
  h_D = new float[NBlocks];
  
  // resultados del kernel de memoria global
  h_D_global = new float[NBlocks];
  h_mx_global= new float[NBlocks];

  // resultados del kernel de memoria compartida
  h_D_shared = new float[NBlocks];
  h_mx_shared = new float[NBlocks];

  // Allocate device memory
  int sizeABC = N*sizeof(float);
  int sizeD = NBlocks*sizeof(float);

  d_A = NULL;
  err = cudaMalloc((void **) &d_A, sizeABC);
  if (err != cudaSuccess) cout << "ERROR RESERVA A" << endl;
  
  d_B = NULL;
  err = cudaMalloc((void **) &d_B, sizeABC);
  if (err != cudaSuccess) cout << "ERROR RESERVA B" << endl;

  d_C = NULL;
  err = cudaMalloc((void **) &d_C, sizeABC);
  if (err != cudaSuccess) cout << "ERROR RESERVA C" << endl;

  d_D_global = NULL;
  err = cudaMalloc((void **) &d_D_global, sizeD);
  if (err != cudaSuccess) cout << "ERROR RESERVA D (GLOBAL)" << endl;

  d_mx_global = NULL; // array with the maximum of each block of C (device)
  err = cudaMalloc((void **) &d_mx_global, sizeD);
  if (err != cudaSuccess) cout << "ERROR RESERVA MX (GLOBAL)" << endl;

  d_D_shared = NULL; 
  err = cudaMalloc((void **) &d_D_shared, sizeD);
  if (err != cudaSuccess) cout << "ERROR RESERVA D (SHARED)" << endl;

  d_mx_shared = NULL;
  err = cudaMalloc((void **) &d_mx_shared, sizeD);
  if (err != cudaSuccess) cout << "ERROR RESERVA MX (SHARED)" << endl;

  //* Initialize arrays */
  for (int i=0; i<N; i++){ 
	h_A[i]= (float) (1  -(i%100)*0.001);
	h_B[i]= (float) (0.5+(i%10) *0.1  );    
  }

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);






  /*********************** GPU Phase (global memory) ************************/
  double t1 = cpuSecond();

  // copy A and B to device
  err = cudaMemcpy(d_A, h_A, sizeABC, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) cout << "ERROR COPIA A" << endl;

  err = cudaMemcpy(d_B, h_B, sizeABC, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) cout << "ERROR COPIA B" << endl;
  

  dim3 threadsPerBlock(Bsize, 1);
  dim3 numBlocks(NBlocks, 1);
   
  int smemSize = Bsize*4*sizeof(float);
  transformacion_kernel_global<<<numBlocks, threadsPerBlock, smemSize>>>(
       d_A, d_B, d_C, d_D_global, d_mx_global);

  cudaMemcpy(h_D_global, d_D_global, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mx_global, d_mx_global, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

  float mx_global_final = h_mx_global[0];
  cudaDeviceSynchronize();

  // final reduction on CPU
  for (int k = 1; k<NBlocks; k++)
     mx_global_final = (mx_global_final > h_mx_global[k]) ? mx_global_final : h_mx_global[k];

  double tgpu_global=cpuSecond()-t1;
  






  /********************** GPU Phase (shared memory) **********************/

  t1 = cpuSecond();

  // copy A and B to device
  //err = cudaMemcpy(d_A, h_A, sizeABC, cudaMemcpyHostToDevice);
  //if (err != cudaSuccess) cout << "ERROR COPIA A" << endl;

  //err = cudaMemcpy(d_B, h_B, sizeABC, cudaMemcpyHostToDevice);
  //if (err != cudaSuccess) cout << "ERROR COPIA B" << endl;

  smemSize = Bsize*4*sizeof(float);
  transformacion_kernel_shared<<<numBlocks, threadsPerBlock, smemSize>>>(
       d_A, d_B, d_C, d_D_shared, d_mx_shared);

  cudaMemcpy(h_D_shared, d_D_shared, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mx_shared, d_mx_shared, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

  float mx_shared_final = h_mx_shared[0];
  cudaDeviceSynchronize();

  // final reduction on CPU
  for (int k = 1; k<NBlocks; k++)
     mx_shared_final = (mx_shared_final > h_mx_shared[k]) ? mx_shared_final : h_mx_shared[k];

  double tgpu_shared=cpuSecond()-t1;









  /******************************* CPU Phase *****************************/
  // Time measurement  
  t1=cpuSecond();
  
  // Compute C[i], d[K] and mx
  for (int k=0; k<NBlocks;k++)
  { 
	int istart=k*Bsize;
  	int iend  =istart+Bsize;
  	h_D[k]=0.0;
  	for (int i=istart; i<iend;i++){ 
	   h_C[i]=0.0;
    	   for (int j=istart; j<iend;j++){
	   	float a=h_A[j]*i;
       		if ((int)ceil(a) % 2 ==0)
		   h_C[i]+= a + h_B[j];
       		else
 	   	   h_C[i]+= a - h_B[j];
     	   }

   	   h_D[k]+=h_C[i];
   	   h_mx= (i==1) ? h_C[0] : max(h_C[i],h_mx);
  	}
  }

  double tsec=cpuSecond()-t1;
  



  /********************** RESULTADOS ***************************/

  //for (int i=0; i<N;i++)   cout<<"C["<<i<<"]="<<h_C[i]<<endl;
  /*cout<<"................................."<<endl;
  for (int k=0; k<NBlocks;k++){
	cout<<"D["<<k<<"]="<<h_D[k]<<endl;
	cout<<"D_global["<<k<<"]="<<h_D_global[k]<<endl;
	cout<<"D_shared["<<k<<"]="<<h_D_shared[k]<<endl;
  }*/
  cout<<"................................."<<endl<<"El valor máximo en C (sec) es: "<<h_mx<<endl;
  cout<<"................................."<<endl<<"El valor máximo en C (gpu global) es: "<<mx_global_final<<endl;
  cout<<"................................."<<endl<<"El valor máximo en C (gpu shared) es: "<<mx_shared_final << endl;
  cout << endl << "N=" << N << "= " << Bsize << "*" << NBlocks << endl;
  cout << "Tiempo gastado CPU= " << tsec << endl;
  cout << "Tiempo gastado GPU (mem global)= " << tgpu_global << endl;
  cout << "Tiempo gastado GPU (mem compartida)= " << tgpu_shared << endl;

  cout << endl << "Ganancia mem global = " << tsec/tgpu_global << endl;
  cout << "Ganancia mem compartida = " << tsec/tgpu_shared << endl;

  // Free host memory
  delete [] h_A; 
  delete [] h_B; 
  delete [] h_C;
  delete [] h_D;
  delete [] h_D_global;
  delete [] h_mx_global;
  delete [] h_D_shared;
  delete [] h_mx_shared;

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_mx_global);
  cudaFree(d_mx_shared);
  cudaFree(d_D_global);
  cudaFree(d_D_shared);
}
