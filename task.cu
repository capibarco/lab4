#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void calc(double* matrixOld, double* matrixNew, int n)
{
    size_t i = blockIdx.x;
	size_t j = threadIdx.x;

	if((i > 0 && i < n-1) && (j > 0 && j < n-1))
	matrixNew[i * n + j] = 0.25 * (
					matrixOld[i * n + j - 1] +
					matrixOld[(i - 1) * n + j] +
					matrixOld[(i + 1) * n + j] +
					matrixOld[i * n + j + 1]);
}
__global__ void findError(double* matrixOld, double* matrixNew, double* matrixTmp, size_t n)
{
    size_t i = blockIdx.x;
    size_t j = threadIdx.x;
	if((i > 0 && i < n-1) && (j > 0 && j < n-1))
	{
		size_t idx = i * blockDim.x + j;
		matrixTmp[idx] = matrixNew[idx] - matrixOld[idx];
	}
}
int main(int argc, char** argv)
{
	int cornerUL = 10;
	int cornerUR = 20;
	int cornerBR = 30;
	int cornerBL = 20;

	char* eptr;
	const double maxError = strtod((argv[1]), &eptr);
	const int size = atoi(argv[2]);
	const int maxIteration = atoi(argv[3]);
	const int toPrint = argc > 4 ? 1 : 0;

	int totalSize = size * size;

	double* matrixOld = (double*)calloc(totalSize, sizeof(double));
	double* matrixNew = (double*)calloc(totalSize, sizeof(double));
	double* matrixTmp = (double*)calloc(totalSize, sizeof(double));
const double fraction = 10.0 / (size - 1);
for (int i = 0; i < size; i++)
	{
		matrixOld[i] = cornerUL + i * fraction;
		matrixOld[i * size] = cornerUL + i * fraction;
		matrixOld[size * i + size - 1] = cornerUR + i * fraction;
		matrixOld[size * (size - 1) + i] = cornerUR + i * fraction;

		matrixNew[i] = matrixOld[i];
		matrixNew[i * size] = matrixOld[i * size];
		matrixNew[size * i + size - 1] = matrixOld[size * i + size - 1];
		matrixNew[size * (size - 1) + i] = matrixOld[size * (size - 1) + i];
	}

    double* matrixOldD;
    double* matrixNewD;
    double* matrixTmpD;

	cudaMalloc((void **)&matrixOldD, sizeof(double)*totalSize);
    cudaMalloc((void **)&matrixNewD, sizeof(double)*totalSize);
    cudaMalloc((void **)&matrixTmpD, sizeof(double)*totalSize);

	cudaMemcpy(matrixOldD, matrixOld, sizeof(double)*totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixNewD, matrixNew, sizeof(double)*totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixTmpD, matrixTmp, sizeof(double)*totalSize, cudaMemcpyHostToDevice);


	  int blockSize =32;
	 int gridSize = (size + 31) / 32;


    double* max_error, *store=0;
    cudaMalloc(&max_error, sizeof(double));

    size_t tempsize  = 0;
    cub::DeviceReduce::Max(store, tempsize, matrixTmpD, max_error, totalSize);
	cudaMalloc((void**)&store, tempsize);



	
	double errorNow = 1.0;
	int iterNow = 0;

	int result = 0;
	const double minus = -1;

	clock_t begin = clock();
if (toPrint)
	{
		#pragma acc kernels loop seq
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
				printf("%lf\t",matrixOld[size * i + j]);
			printf("\n");				  
		}
		printf("\n");
	}
while (errorNow > maxError && iterNow < maxIteration)
{
calc<<<gridSize, blockSize>>>(matrixOldD, matrixNewD, size);
iterNow++;
		if (iterNow % 100 == 0){
			findError<<<gridSize, blockSize>>>(matrixOldD, matrixNewD, matrixTmpD, size);

			 cub::DeviceReduce::Max(store, tempsize, matrixTmpD, max_error, totalSize);
        cudaMemcpy(&errorNow, max_error, sizeof(double), cudaMemcpyDeviceToHost);
		}

		double* t = matrixOldD;
		matrixOldD = matrixNewD;
		matrixNewD = t;

		
}
	clock_t end = clock();
	free(matrixOld);
	free(matrixNew);
	free(matrixTmp);
	cudaFree(matrixOldD);
	cudaFree(matrixNewD);
	cudaFree(matrixTmpD);
	printf("iterations = %d, error = %lf, time = %lf\n", iterNow, errorNow, (double)(end - begin) / CLOCKS_PER_SEC);

	return 0;
}
