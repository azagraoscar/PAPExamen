
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 16
const double DIFUSIVIDAD = 8.418e-5;
const int ITERACIONES = 10;
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

__global__ void actualizacion_temperatura(int r[N][N], int a[N][N])
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
			media += a[pos_x-1][pos_y - 1];
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
	r[pos_x][pos_y] = media/contador;
}
int main()
{
	
}