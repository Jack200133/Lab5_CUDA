#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vectorAddCPU(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int n = 1000000; // Tamaño grande para medir tiempo significativo
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *c = malloc(n * sizeof(float));

    // Inicialización de vectores con valores aleatorios
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    clock_t start = clock();
    vectorAddCPU(a, b, c, n);
    // Calcular e imprimir la suma de todos los valores del vector resultante
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += c[i];
    }
    printf("Suma de todos los valores en C: %f\n", sum);

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución secuencial: %f segundos\n", time_spent);

    free(a);
    free(b);
    free(c);

    return 0;
}
