
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//Function
__global__ void crearMatrizDensidad(int a[16][16], int b[16][16])
{
    int x = threadIdx.x;  //We obtain the id of the x thread, so we can use it to iterate through the rows of the matrix.  
    int y = threadIdx.y;  //We obtain the id of the y thread, so we can use it to iterate through the columns of the matrix.
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

int main()
{
    //Declare all variables.
        //CPU

    //aD = matriz 16x16 con valores entre 80 y 95.
    //bD = matriz 16x16 calculada sobre aD, valores entre 80 y 87 dan aqui un valor 0, valores entre 88 y 90 dan valores 1, y valores entre 91 y 95 dan valores 2.

    const int arraySize = 16; //Size of matrix on each axis
    const int upper = 95;
    const int lower = 80;
    int aH[arraySize][arraySize]; //First matrix in CPU
    int bH[arraySize][arraySize]; //Second matrix in CPU

        //GPU
    int (* aD)[arraySize]; //Pointer to an array of arraySize for the matrix A.
    int (* bD)[arraySize]; //Pointer to an array of arraySize for the matrix B

    //Generamos las matrices con valores aleatorios
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
        {
            aH[i][j] = rand() % (upper-lower+1)+lower;
        }
    }

    // Display the matrix A
    printf("Matrix con Datos entre 80 y 95:\n");
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
        {
            printf("%d  ", aH[i][j]);
        }
        printf("\n");
    }

    //Allocate memory    
    cudaMalloc((void**)&aD, (arraySize * sizeof(int) * arraySize)); //For the matrix A, allocate memory for the whole structure.
    cudaMalloc((void**)&bD, (arraySize * sizeof(int) * arraySize)); //For the matrix B, allocate memory for the whole structure.

    //Write and do things
    //Copy each row of the matrix A and B from CPU to each variable of GPU respectively.
    for (int i = 0; i < arraySize; i++)
    {
        cudaMemcpy(&aD[i], aH[i], (arraySize * sizeof(int)), cudaMemcpyHostToDevice); //First matrix from CPU to GPU
        cudaMemcpy(&bD[i], bH[i], (arraySize * sizeof(int)), cudaMemcpyHostToDevice); //Second matrix from CPU to GPU
    }

    //Define a matrix of 16x16x1 = 256 threads for a block, and invoke the function with one block and the arrangement of the threads in the block.
    dim3 threadsPerBlock(16, 16, 1);
    crearMatrizDensidad <<<1, threadsPerBlock >>> (aD, bD); //aD = Matriz con datos de 80 a 95, bD es matriz con 0s, 1s o 2s

    //Retrieve values of the resulting matrix from GPU to CPU.
    for (int i = 0; i < arraySize; i++)
        cudaMemcpy(bH[i], &bD[i], (arraySize * sizeof(int)), cudaMemcpyDeviceToHost);

    //Print resulting matrix
    printf("\nMatriz de Densidad (0s, 1s, 2s en funcion de 80<=valor<=95):\n");
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
        {
            printf("%d  ", bH[i][j]);
        }
        printf("\n");
    }

    //Free allocated memory (with malloc()/cudamalloc())
    cudaFree(aD);
    cudaFree(bD);

    return 0;
}

