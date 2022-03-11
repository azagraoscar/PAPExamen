
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

const int N = 16;  //tamaño de la matriz A , min y max


__global__ void  compararMatrices(int mmax[N][N], int mmin[N][N],int mactual[N][N])
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



void imprimir_matriz(int m[N][N], int width) //función auxiliar para imprimir matrices por pantalla
{
	printf("Matrix con Datos entre 80 y 95:\n");
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < width; j++)
		{
			printf("%d  ", m[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
int main()
{
	//Declaración de variables
	
	//Variables del host
	int a_h[N][N];
	int b_h[N][N];
	int c_h[N][N];

	//Generamos las matrices con valores aleatorios
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a_h[i][j] = rand() % (95 - 80 + 1) + 80;
		}
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			b_h[i][j] = rand() % (95 - 80 + 1) + 80;
		}
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			c_h[i][j] = rand() % (95 - 80 + 1) + 80;
		}
	}
	
	int size = sizeof(a_h);


	imprimir_matriz(a_h, N);
	imprimir_matriz(b_h, N);
	imprimir_matriz(c_h, N);

	//Variables del device
	int(*a_d)[N];
	int(*b_d)[N];
	int(*c_d)[N];


	//Usamos un tamaño de bloques de 16 porque es el óptimo. El número de bloques entonces, al trabajar con una matriz de tamaño igual a la matriz A + 2, necesitaremos 2x2 bloques
	dim3 blockSize(16, 16);
	dim3 grid(2, 2);


	//Gestión de memoria

	cudaMalloc((void**)&a_d, size);
	cudaMalloc((void**)&b_d, size);
	cudaMalloc((void**)&c_d, size);


	//Movemos las variables al dispositivo
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice);



	//Realizamos las operaciones
	compararMatrices<< <grid, blockSize >> > (a_d, b_d, c_d);
	cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

	//liberamos memoria
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	//Imprimimos el resultado por pantalla
	printf("RESULTADO: \n");
	imprimir_matriz(a_h, N);
	imprimir_matriz(b_h, N);
}
