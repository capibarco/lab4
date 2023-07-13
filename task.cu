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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//Индекс для обращения к элементу массива 
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;//Высчитывается из логики - Номер блока на размер блока(кол-во поток) плюс номер потока

    if((i > 0 && i < n-1) && (j > 0 && j < n-1))
        matrixNew[i * n + j] = 0.25 * (
                matrixOld[i * n + j - 1] +
                matrixOld[(i - 1) * n + j] +
                matrixOld[(i + 1) * n + j] +
                matrixOld[i * n + j + 1]);
}
__global__ void findError(double* matrixOld, double* matrixNew, double* matrixTmp, size_t n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//Индекс для обращения к элементу массива 
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;//Высчитывается из логики - Номер блока на размер блока(кол-во поток) плюс номер потока
    if((i > 0 && i < n-1) && (j > 0 && j < n-1))
    {
        size_t idx = i * blockDim.x + j;
        matrixTmp[idx] = matrixNew[idx] - matrixOld[idx];
    }
}
__global__ void fill(double* arr, double* new_arr, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    arr[0] = new_arr[0] = 10;
    arr[n - 1]= new_arr[n - 1] = 20;
    arr[n * n - 1] = new_arr[n * n - 1] = 30;
    arr[n * (n - 1)] = new_arr[n * (n - 1)] = 20;
    if(i > 0 && i < n-1)
    {
        arr[i] = new_arr[i] = arr[0] + (10.0 / (n-1)) * i;
        arr[n*(n-1) + i] = new_arr[n*(n-1) + i] = arr[n - 1] + 10.0 / (n-1) * i;
        arr[n*i]= new_arr[n*i] = arr[0] + 10.0 / (n-1) * i;
        arr[n*i + n - 1] = new_arr[n*i + n - 1] = arr[n-1] + 10.0 / (n-1) * i;
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
/*
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
*/

    double* matrixOldD = 0;
    double* matrixNewD = 0;
    double* matrixTmpD = 0;
    double* errorD =0;
    double* store = 0;

    cudaMalloc((void **)&matrixOldD, sizeof(double)*totalSize);
    cudaMalloc((void **)&matrixNewD, sizeof(double)*totalSize);
    cudaMalloc((void **)&matrixTmpD, sizeof(double)*totalSize);
    cudaMalloc((void **)&errorD, sizeof(double));
    size_t tempsize  = 0;
    cub::DeviceReduce::Max(store, tempsize, matrixTmpD, errorD, totalSize);
    cudaMalloc((void**)&store, tempsize);


    cudaMemcpy(matrixOldD, matrixOld, sizeof(double)*totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixNewD, matrixNew, sizeof(double)*totalSize, cudaMemcpyHostToDevice);


    int blockS, minGridSize = 128;
    int maxSize = size;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockS, calc, 0, totalSize);
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize  = dim3((size + 31) / 32, (size + 31) / 32);

    double errorNow = 1.0;
    int iterNow = 0;

    
    clock_t begin = clock();
 fill<<<gridSize, blockSize>>>(matrixOldD, matrixNewD, size);
    while (errorNow > maxError && iterNow < maxIteration)
    {
        calc<<<gridSize, blockSize>>>(matrixOldD, matrixNewD, size);

        if (iterNow % 100 == 0)
        {
            findError<<<gridSize, blockSize>>>(matrixOldD, matrixNewD, matrixTmpD, size);

            cub::DeviceReduce::Max(store, tempsize, matrixTmpD, errorD, totalSize);
            cudaMemcpy(&errorNow, errorD, sizeof(double), cudaMemcpyDeviceToHost);
        }
        double* t = matrixOldD;
        matrixOldD = matrixNewD;
        matrixNewD = t;


        iterNow++;


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
