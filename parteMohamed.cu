#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16
#define ITERACIONES 10


__global__ void kernel(int maximos[N][N], int minimos[N][N], int matriz[N][N])
{
	
}

void imprimirMatriz(int matriz[N][N])
{
	for (int i = 0; i < N; i++)
	{
		printf("{");
		for (int j = 0; j < N; j++)
		{
			if (j < N - 1)
			{
				printf("%2d,", matriz[i][j]);
			}
			else
			{
				printf("%2d", matriz[i][j]);
			}
		}
		printf("}\n");
	}
}

void generarAuxiliar(int matriz[N][N])
{
	int variaciones[] = { -2,-1,0,1,2 };
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			int n = rand() % 5;
			matriz[i][j] += variaciones[n];
		}
	}
}

int main(void)
{
	//Declare all variables
	int matriz_h[N][N];
	int maximos_h[N][N];
	int minimos_h[N][N];

	int(*matriz_d)[N];
	int(*maximos_d)[N];
	int(*minimos_d)[N];
	
	int size = sizeof(matriz_h);

	// Dynamically allocate device memory for GPU results.
	cudaMalloc((void**)&matriz_d, size);
	cudaMalloc((void**)&maximos_d, size);
	cudaMalloc((void**)&minimos_d, size);

	// Copy host memory to device memory

	//rellenamos matriz inicial
	for (int  i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i == 0 || j == 0 || i == N-1 || j == N-1)
			{
				matriz_h[i][j] = 80;
			}
			else
			{
				matriz_h[i][j] = (rand() % 15) + 80;
			}
		}
	}
	printf("Matriz Inicial: \n\n");
	imprimirMatriz(matriz_h);

	generarAuxiliar(matriz_h);
	printf("\nMatriz Auxiliar\n\n");
	imprimirMatriz(matriz_h);

	dim3 dimBlock(16, 16);
	dim3 dimGrid(N/16,N/16);

	for (int i = 0; i < ITERACIONES; i++)
	{
		cudaMemcpy(matriz_d, matriz_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(maximos_d, maximos_h, size, cudaMemcpyHostToDevice);
		cudaMemcpy(minimos_d, minimos_h, size, cudaMemcpyHostToDevice);

		kernel << < dimGrid, dimBlock >> > (maximos_d, minimos_d, matriz_d);

		cudaMemcpy(matriz_h, matriz_d, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(maximos_h, maximos_d, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(minimos_h, minimos_d, size, cudaMemcpyDeviceToHost);

		generarAuxiliar(matriz_h);
	}

	//TODO: densidad

	//TODO: calculo calor

	//TODO:imprimir todas las matrices

	// Free dynamically−allocated device memory
	cudaFree(matriz_d);
	cudaFree(maximos_d);
	cudaFree(minimos_d);

	return 0;
}