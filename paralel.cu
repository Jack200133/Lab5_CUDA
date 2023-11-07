#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1000000; // Tamaño grande para medir tiempo significativo
    size_t size = n * sizeof(float);

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Calcular e imprimir la suma de todos los valores del vector resultante
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += c[i];
    }
    printf("Suma de todos los valores en C: %f\n", sum);


    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Tiempo de ejecución de CUDA: %f milisegundos\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
