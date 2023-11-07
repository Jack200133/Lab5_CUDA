#include <stdio.h>

// CUDA kernel para sumar elementos de dos vectores
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Asegúrate de no salir del rango de los vectores
    if (index < n) {
        c[index] = a[index] + b[index];
        printf("Thread #%d, sumando %d + %d = %d\n", index, a[index], b[index], c[index]);
    }
}

// Función principal
int main() {
    int n = 10; // Por ejemplo, tamaño de los vectores
    int *a, *b, *c; // vectores en el host
    int *d_a, *d_b, *d_c; // vectores en el device
    int size = n * sizeof(int);

    // Asignación de memoria en el host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Inicializar los vectores de entrada con valores
    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

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
