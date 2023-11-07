#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel para sumar elementos de dos vectores
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        c[index] = a[index] + b[index];
        printf("Thread #%d, sumando %f + %f = %f\n", index, a[index], b[index], c[index]);
    }
}

// Función para inicializar los vectores con valores aleatorios entre 0 y 1
void initializeVectors(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Función para imprimir los vectores
void printVector(float *vec, int n, const char *name) {
    printf("Vector %s: ", name);
    for (int i = 0; i < n; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

// Función principal
int main() {
    int n = 10; // Por ejemplo, tamaño de los vectores
    float *a, *b, *c; // vectores en el host
    float *d_a, *d_b, *d_c; // vectores en el device
    int size = n * sizeof(float);

    // Asignación de memoria en el host
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Inicializar la semilla de aleatoriedad
    srand(time(NULL));

    // Inicializar los vectores de entrada con valores aleatorios
    initializeVectors(a, n);
    initializeVectors(b, n);

    // Imprimir los vectores inicializados
    printVector(a, n, "A");
    printVector(b, n, "B");

    // Asignación de memoria en el device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copiar los vectores de entrada al device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Lanzar el kernel en el device con un thread por cada elemento
    vectorAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);

    // Copiar el vector de resultado de vuelta al host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Limpieza
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    // Verificación de errores de CUDA
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // Finalización exitosa
    return 0;
}
