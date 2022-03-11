#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define N 32
#define ITERACIONES 10
#define DIFUSIVIDAD 8.418e-5


__global__ void matrizDensidad(int a[N][N], int b[N][N])
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;  //We obtain the id of the x thread, so we can use it to iterate through the rows of the matrix.  
	int y = blockIdx.y * blockDim.y + threadIdx.y;  //We obtain the id of the y thread, so we can use it to iterate through the columns of the matrix.
	if ((80 <= a[x][y]) && (a[x][y] < 88))
	{
		b[x][y] = 0; //Color amarilo = 0
	}
	else if ((88 <= a[x][y]) && (a[x][y] < 91))
	{
		b[x][y] = 1; //Color narankja = 1
	}
	else if ((91 <= a[x][y]) && (a[x][y] <= 95))
	{
		b[x][y] = 2; //Color rojo = 2
	}
}

__global__ void conduccion_calor(int r[N][N], int a[N][N])
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
	double temp = 0;
	//temp = a[pos_x + 1][pos_y] - 2 * a[pos_x][pos_y] + a[pos_x - 1][pos_y] + a[pos_x][pos_y + 1] - 2 * a[pos_x][pos_y] + a[pos_x][pos_y - 1];
	if (pos_x - 1 > 0)
	{
		temp += a[pos_x - 1][pos_y];
	}
	if (pos_x + 1 < N)
	{
		temp += a[pos_x + 1][pos_y];
	}
	if (pos_y - 1 > 0)
	{
		temp += a[pos_x][pos_y - 1];
	}
	if (pos_y + 1 < N)
	{
		temp += a[pos_x][pos_y + 1];
	}
	temp += -4 * a[pos_x][pos_y];
	r[pos_x][pos_y] = a[pos_x][pos_y] + DIFUSIVIDAD * temp;
}

__global__ void actualizacion_temperatura(int a[N][N])
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
	int media = 0;
	int contador = 0;
	if (pos_x - 1 > 0)
	{
		media += a[pos_x - 1][pos_y];
		contador++;
		if (pos_y - 1 > 0)
		{
			media += a[pos_x - 1][pos_y - 1];
			contador++;
		}
		if (pos_y + 1 < N)
		{
			media += a[pos_x - 1][pos_y - 1];
			contador++;
		}
	}
	if (pos_x + 1 < N)
	{
		media += a[pos_x + 1][pos_y];
		contador++;
		if (pos_y + 1 < N)
		{
			media += a[pos_x + 1][pos_y + 1];
			contador++;
		}
		if (pos_y - 1 > 0)
		{
			media += a[pos_x + 1][pos_y - 1];
			contador++;
		}
	}
	if (pos_y - 1 > 0)
	{
		media += a[pos_x][pos_y - 1];
		contador++;
	}
	if (pos_y + 1 < N)
	{
		media += a[pos_x][pos_y + 1];
		contador++;
	}
	a[pos_x][pos_y] = media / contador;
}

__global__ void  compararMatrices(int mmax[N][N], int mmin[N][N], int mactual[N][N])
{ //Funcion que va cambiando los valores de las matrices de minimos y maximos comparando cada posicion con la de la matriz de una iteracion derterminada

	int pos_x = blockDim.x * blockIdx.x + threadIdx.x;
	int pos_y = blockDim.y * blockIdx.y + threadIdx.y;


	if (mactual[pos_x][pos_y] > mmax[pos_x][pos_y]) {  //Si la posicion es mayor que esa misma posicion en la matriz de maximos, actualizamos el valor.
		mmax[pos_x][pos_y] = (mactual[pos_x][pos_y]);
	}
	if (mactual[pos_x][pos_y] < mmin[pos_x][pos_y]) {  //Si la posicion es menor que esa misma posicion en la matriz de minimos, actualizamos el valor.
		mmin[pos_x][pos_y] = (mactual[pos_x][pos_y]);
	}
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
	int densidad_h[N][N];
	int matrizConduccion_h[N][N];

	int(*matriz_d)[N];
	int(*maximos_d)[N];
	int(*minimos_d)[N];
	int(*densidad_d)[N];
	int(*matrizConduccion_d)[N];
	
	int size = sizeof(matriz_h);

	// Dynamically allocate device memory for GPU results.
	cudaMalloc((void**)&matriz_d, size);
	cudaMalloc((void**)&maximos_d, size);
	cudaMalloc((void**)&minimos_d, size);
	cudaMalloc((void**)&densidad_d, size);
	cudaMalloc((void**)&matrizConduccion_d, size);


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

	//inicializar maximos y minimos
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			maximos_h[i][j] = matriz_h[i][j];
			minimos_h[i][j] = matriz_h[i][j];
			densidad_h[i][j] = 0;
		}
	}

	printf("Matriz Inicial: \n\n");
	imprimirMatriz(matriz_h);

	dim3 dimBlock(16, 16);	//usaremos teselas de 16x16
	dim3 dimGrid(N/16,N/16);	//calculamos tamaño de grid

	cudaMemcpy(maximos_d, maximos_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(minimos_d, minimos_h, size, cudaMemcpyHostToDevice);

	for (int i = 0; i < ITERACIONES; i++)
	{
		//generamos matriz auxiliar
		generarAuxiliar(matriz_h);

		//copiamos datos de CPU a GPU
		cudaMemcpy(matriz_d, matriz_h, size, cudaMemcpyHostToDevice);

		actualizacion_temperatura << < dimGrid, dimBlock >> > (matriz_d);
		compararMatrices << < dimGrid, dimBlock >> > (maximos_d, minimos_d, matriz_d);

		//copiamos datos de GPU a CPU
		cudaMemcpy(matriz_h, matriz_d, size, cudaMemcpyDeviceToHost);

	}

	cudaMemcpy(maximos_h, maximos_d, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(minimos_h, minimos_d, size, cudaMemcpyDeviceToHost);

	//densidad
	cudaMemcpy(densidad_d, densidad_h, size, cudaMemcpyHostToDevice);
	matrizDensidad << < dimGrid, dimBlock >> > (matriz_d, densidad_d);
	cudaMemcpy(densidad_h, densidad_d, size, cudaMemcpyDeviceToHost);

	//calculo calor
	cudaMemcpy(matrizConduccion_d, matrizConduccion_h, size, cudaMemcpyHostToDevice);
	conduccion_calor << < dimGrid, dimBlock >> > (matrizConduccion_d, matriz_d);
	cudaMemcpy(matrizConduccion_h, matrizConduccion_d, size, cudaMemcpyDeviceToHost);

	//TODO:imprimir todas las matrices
	printf("\nMatriz final\n");
	imprimirMatriz(matriz_h);
	printf("\nMatriz maximos\n");
	imprimirMatriz(maximos_h);
	printf("\nMatriz minimos\n");
	imprimirMatriz(minimos_h);
	printf("\nMatriz conduccion\n");
	imprimirMatriz(matrizConduccion_h);
	printf("\nMatriz densidad\n");
	imprimirMatriz(densidad_h);

	// Free dynamically−allocated device memory
	cudaFree(matriz_d);
	cudaFree(maximos_d);
	cudaFree(minimos_d);
	cudaFree(densidad_d);
	cudaFree(matrizConduccion_d);

	return 0;
}